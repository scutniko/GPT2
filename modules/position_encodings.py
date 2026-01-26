"""
各种位置编码实现
包含可学习位置编码、ALiBi、RoPE、正弦位置编码等
"""

import math
import torch
import torch.nn as nn


class LearnedPositionEncoding(nn.Module):
    """
    可学习的位置编码（GPT-2 baseline）
    使用Embedding层来学习位置信息
    """
    
    def __init__(self, block_size, n_embd):
        super().__init__()
        self.wpe = nn.Embedding(block_size, n_embd)
    
    def forward(self, idx):
        """
        Args:
            idx: token indices [B, T]
        Returns:
            pos_emb: position embeddings [T, n_embd]
        """
        T = idx.size(1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.wpe(pos)  # shape (T, n_embd)
        return pos_emb


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi)
    在attention score上添加与位置距离成比例的负偏置
    """
    
    def __init__(self, n_head, max_seq_len=2048):
        super().__init__()
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        
        # 预计算每个head的斜率 (slopes)
        # 公式: m_h = 2^(-8/n_head * h) for h in [1, n_head]
        slopes = torch.tensor(self._get_slopes(n_head))
        self.register_buffer("slopes", slopes)
        
        # 预计算偏置矩阵（用于加速）
        self._init_cache(max_seq_len)
    
    @staticmethod
    def _get_slopes(n_head):
        """计算每个head的斜率"""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        # 如果head数是2的幂次
        if math.log2(n_head).is_integer():
            return get_slopes_power_of_2(n_head)
        else:
            # 如果不是2的幂次，取最接近的2的幂次，然后插值
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = ALiBi._get_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][:n_head - closest_power_of_2]
            return slopes_a + slopes_b
    
    def _init_cache(self, seq_len):
        """
        预计算ALiBi偏置矩阵
        
        偏置矩阵 shape: [n_head, seq_len, seq_len]
        bias[h, i, j] = -slopes[h] * |i - j|
        """
        # 创建位置矩阵 [seq_len, seq_len]
        # 其中 relative_position[i, j] = i - j
        context_position = torch.arange(seq_len)[:, None]
        memory_position = torch.arange(seq_len)[None, :]
        relative_position = memory_position - context_position  # [seq_len, seq_len]
        
        # 对于因果注意力，只关注过去的位置（j <= i）
        # 未来的位置设为0（之后会被mask掉）
        relative_position = torch.abs(relative_position)
        
        # 应用每个head的斜率: [n_head, 1, 1] * [1, seq_len, seq_len]
        # 结果: [n_head, seq_len, seq_len]
        alibi_bias = -self.slopes[:, None, None] * relative_position[None, :, :]
        
        # 添加batch维度: [1, n_head, seq_len, seq_len]
        self.register_buffer("alibi_bias", alibi_bias[None, :, :, :], persistent=False)
    
    def forward(self, query_len, key_len=None):
        """
        返回ALiBi偏置矩阵
        
        Args:
            query_len: 查询长度
            key_len: 键值长度（可选，默认与query_len相同）
            
        Returns:
            bias: [1, n_head, query_len, key_len]
        """
        if key_len is None:
            key_len = query_len
        
        # 如果序列长度超过缓存，重新计算
        if key_len > self.alibi_bias.shape[2]:
            self._init_cache(key_len)
        
        # 返回对应长度的偏置（取最后query_len行）
        return self.alibi_bias[:, :, key_len - query_len:key_len, :key_len]


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    通过旋转变换来编码位置信息
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算cos和sin缓存（用于加速）
        self._init_cache(max_seq_len)
    
    def _init_cache(self, seq_len):
        """预计算cos和sin值"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, q, k, start_pos=0):
        """
        对query和key应用旋转位置编码
        
        Args:
            q, k: [B, n_head, T, head_dim]
            start_pos: 起始位置偏移（用于KV cache）
            
        Returns:
            q_rot, k_rot: 应用RoPE后的q和k
        """
        seq_len = q.shape[2]
        end_pos = start_pos + seq_len
        
        # 如果序列长度超过缓存，重新计算
        if end_pos > self.cos_cached.shape[2]:
            self._init_cache(end_pos)
        
        # 获取对应长度的cos和sin
        cos = self.cos_cached[:, :, start_pos:end_pos, :]
        sin = self.sin_cached[:, :, start_pos:end_pos, :]
        
        # 应用旋转
        q_rot = self.apply_rotary_emb(q, cos, sin)
        k_rot = self.apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    @staticmethod
    def apply_rotary_emb(x, cos, sin):
        """
        应用旋转变换
        
        Args:
            x: [B, n_head, T, head_dim]
            cos, sin: [1, 1, T, head_dim]
            
        Returns:
            rotated_x: 应用旋转后的tensor
        """
        # 将x分成两半
        x1 = x[..., : x.shape[-1] // 2]  # 前半部分
        x2 = x[..., x.shape[-1] // 2 :]  # 后半部分
        
        # 对cos和sin也分成两半
        cos1 = cos[..., : cos.shape[-1] // 2]
        cos2 = cos[..., cos.shape[-1] // 2 :]
        sin1 = sin[..., : sin.shape[-1] // 2]
        sin2 = sin[..., sin.shape[-1] // 2 :]
        
        # 应用旋转公式：
        # [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos1 - x2 * sin1
        rotated_x2 = x1 * sin2 + x2 * cos2
        
        # 拼接回来
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码（固定，不可学习）
    """
    
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self._init_cache(max_seq_len)

    def _init_cache(self, max_seq_len):
        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, self.d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，不参与梯度更新
        # shape: [1, max_seq_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    
    def ensure_cache(self, max_seq_len, device=None, dtype=None):
        """
        预先扩展位置编码缓存，避免逐步扩展带来的开销
        """
        if max_seq_len <= self.pe.size(1):
            return
        self._init_cache(max_seq_len)
        if device is not None or dtype is not None:
            self.pe = self.pe.to(device=device, dtype=dtype)
    
    def forward(self, x, start_pos=0):
        """
        Args:
            x: [B, T, d_model]
            start_pos: 起始位置偏移（用于KV cache）
            
        Returns:
            pe: [B, T, d_model] 位置编码
        """
        # 返回对应序列长度的位置编码
        end_pos = start_pos + x.size(1)
        if end_pos > self.pe.size(1):
            self._init_cache(end_pos)
            self.pe = self.pe.to(x.device, dtype=x.dtype)
        return self.pe[:, start_pos:end_pos, :]

