"""
各种注意力机制实现
包含标准MHA、MQA、GQA、MLA等变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    """
    标准的多头自注意力机制（GPT-2 baseline）
    """
    
    def __init__(self, config, rope=None, alibi=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 在一个batch中，对全部的head进行query, key, value投影
        # B: batch size
        # T: sequence length
        # C: embedding dimension

        # [B, T, C] -> [B, T, 3 * C]
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # [B, T, 3 * C] -> [B, T, C]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # n_head: number of heads
        # n_embd: embedding dimension
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # 位置编码
        self.rope = rope
        self.alibi = alibi

    def forward(self, x, past_kv=None, use_cache=False):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        # C是n_head * head_size
        # GPT-2 (124M), n_head=12, head_size=64, 所以 n_head * head_size=C=768 即Transformer的通道数
        B, T, C = x.size() 
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        
        # 对输入x进行线性变换，得到[B, T, 3 * C]
        qkv = self.c_attn(x)

        # [B, T, 3 * C] -> [B, T, C], [B, T, C], [B, T, C]
        # 将[B, T, 3 * C]沿着第2维（feature维）按照每份n_embd的大小分成3份，也就是q、k、v三个张量。
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 将qkv分别转换为[B, T, n_head, head_size]的形状，再转置得到[B, n_head, T, head_size]
        # 之所以要转置，是会把batch与时间步维度放到前面，方便后续的计算。
        # [B, T, C] -> [B, n_head, T, head_size]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 应用 RoPE（如果有）
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=past_len)
        
        # 拼接KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        # 应用 ALiBi（如果有）
        attn_bias = None
        if self.alibi is not None:
            attn_bias = self.alibi(T, k.size(2))  # [1, n_head, T, K]
        
        # 使用flash attention计算注意力，得到[B, n_head, T, head_size]
        # is_causal=True表示是因果注意力，即只能看到前面的token，不能看到后面的token。
        # 返回的y的shape是[B, n_head, T, head_size]
        use_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=use_causal) # flash attention

        # 将[B, n_head, T, head_size]转置回[B, T, C]，再拼接起来
        # 其中contiguous()是为了确保tensor在内存中是连续的，因为transpose会破坏连续性，所以需要重新排列内存，方便后续的计算。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # 对输出y进行线性变换，shape不变，还是[B, T, C]
        y = self.c_proj(y)
        if use_cache:
            present_kv = (k, v)
            return y, present_kv
        return y


class MQAAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    Query有多个头，Key和Value只有一个头（所有query头共享）
    """
    
    def __init__(self, config, rope=None, alibi=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MQA (Multi-Query Attention): Query有多个头，Key和Value只有一个头
        # B: batch size
        # T: sequence length
        # C: embedding dimension
        # head_dim: 每个头的维度

        head_dim = config.n_embd // config.n_head
        
        # Query: [B, T, C] -> [B, T, C] (多头)
        self.c_q = nn.Linear(config.n_embd, config.n_embd)
        
        # Key, Value: [B, T, C] -> [B, T, head_dim] (单头，所有query头共享)
        self.c_kv = nn.Linear(config.n_embd, 2 * head_dim)
        
        # 输出投影: [B, T, C] -> [B, T, C]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # n_head: number of heads
        # n_embd: embedding dimension
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = head_dim
        
        # 位置编码
        self.rope = rope
        self.alibi = alibi

    def forward(self, x, past_kv=None, use_cache=False):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        # MQA: Query有多个头，Key和Value只有一个头（所有query头共享）
        B, T, C = x.size() 
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        
        # 生成多头Query: [B, T, C]
        q = self.c_q(x)
        # 将Query转换为[B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 生成单头Key和Value: [B, T, 2 * head_dim]
        kv = self.c_kv(x)
        # 分离Key和Value: 每个都是[B, T, head_dim]
        k, v = kv.split(self.head_dim, dim=2)
        
        # 将Key和Value转换为[B, 1, T, head_dim]，然后broadcast到所有头
        # 添加head维度，使其可以与多头Query进行attention计算
        k = k.unsqueeze(1)  # [B, 1, T, head_dim]
        v = v.unsqueeze(1)  # [B, 1, T, head_dim]
        
        # 应用 RoPE（如果有）
        if self.rope is not None:
            # RoPE 需要所有头的 k，所以先 broadcast
            k_expanded = k.expand(B, self.n_head, T, self.head_dim)
            q, k_expanded = self.rope(q, k_expanded, start_pos=past_len)
            k = k_expanded[:, :1, :, :]  # 取回第一个头（所有头相同）
        
        # 拼接KV cache（单头）
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        # 应用 ALiBi（如果有）
        attn_bias = None
        if self.alibi is not None:
            attn_bias = self.alibi(T, k.size(2))  # [1, n_head, T, K]
        
        # 使用flash attention计算注意力
        # q: [B, n_head, T, head_dim]
        # k: [B, 1, T, head_dim] -> 自动broadcast到 [B, n_head, T, head_dim]
        # v: [B, 1, T, head_dim] -> 自动broadcast到 [B, n_head, T, head_dim]
        # 返回的y的shape是[B, n_head, T, head_dim]
        use_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=use_causal) # flash attention

        # 将[B, n_head, T, head_dim]转置回[B, T, C]，再拼接起来
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 对输出y进行线性变换，shape不变，还是[B, T, C]
        y = self.c_proj(y)
        if use_cache:
            present_kv = (k, v)
            return y, present_kv
        return y


class GQAAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    Q 有 n_head 个头，K 和 V 有 n_kv_head 个头 (n_kv_head < n_head)
    多个 Q 头共享一个 KV 头，减少 KV cache 大小
    """
    
    def __init__(self, config, rope=None, alibi=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # GQA: Grouped Query Attention
        # Q 有 n_head 个头，K 和 V 有 n_kv_head 个头 (n_kv_head < n_head)
        # 多个 Q 头共享一个 KV 头，减少 KV cache 大小
        # B: batch size
        # T: sequence length
        # C: embedding dimension

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head  # KV 头的数量
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # 计算每个 KV 头对应多少个 Q 头
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_rep = config.n_head // config.n_kv_head
        
        # Q 投影: [B, T, C] -> [B, T, C]
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # K, V 投影: [B, T, C] -> [B, T, n_kv_head * head_dim]
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        
        # 输出投影: [B, T, C] -> [B, T, C]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # 位置编码
        self.rope = rope
        self.alibi = alibi

    def forward(self, x, past_kv=None, use_cache=False):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        B, T, C = x.size() 
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        
        # Q 投影: [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # K, V 投影: [B, T, C] -> [B, T, n_kv_head, head_dim] -> [B, n_kv_head, T, head_dim]
        k = self.k_proj(x)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        v = self.v_proj(x)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE（如果有）- 在重复之前应用
        if self.rope is not None:
            # 对于 GQA，先对原始的 q 和 k 应用 RoPE
            # q: [B, n_head, T, head_dim]
            # k: [B, n_kv_head, T, head_dim]
            # 需要将 k 扩展到 n_head 来应用 RoPE，然后再压缩回 n_kv_head
            k_expanded = k.repeat_interleave(self.n_rep, dim=1)
            q, k_expanded = self.rope(q, k_expanded, start_pos=past_len)
            # 将 k 压缩回 n_kv_head（取每组的第一个）
            k = k_expanded[:, ::self.n_rep, :, :]
        
        # 拼接KV cache（按KV头存储）
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        k_kv = k
        v_kv = v
        
        # 重复 K 和 V 来匹配 Q 的头数
        # [B, n_kv_head, T, head_dim] -> [B, n_head, T, head_dim]
        if self.n_rep > 1:
            # 每个 KV 头重复 n_rep 次
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # 应用 ALiBi（如果有）
        attn_bias = None
        if self.alibi is not None:
            attn_bias = self.alibi(T, k.size(2))  # [1, n_head, T, K]

        # 使用 flash attention 计算注意力
        # [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]
        use_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=use_causal)

        # 将多头输出拼接回原始维度
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.c_proj(y)
        if use_cache:
            present_kv = (k_kv, v_kv)
            return y, present_kv
        return y


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) - DeepSeek-V2
    通过低秩压缩大幅减少 KV cache，同时保持模型性能
    核心思想：将 KV 投影到低维潜在空间，减少存储和计算开销
    """
    
    def __init__(self, config, rope=None, alibi=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # MLA: Multi-head Latent Attention (DeepSeek-V2)
        # 通过低秩压缩大幅减少 KV cache，同时保持模型性能
        # 核心思想：将 KV 投影到低维潜在空间，减少存储和计算开销
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # MLA 的关键参数：潜在维度（低维压缩空间）
        # 通常设置为 n_embd 的 1/4 到 1/2，这里使用 1/4 以获得更好的压缩比
        self.kv_lora_rank = config.kv_lora_rank  # KV 低秩维度
        self.q_lora_rank = config.q_lora_rank    # Q 低秩维度
        
        # === MLA 的三阶段投影 ===
        
        # 阶段1: 压缩投影 - 将输入压缩到低维潜在空间
        # [B, T, C] -> [B, T, kv_lora_rank + q_lora_rank]
        self.down_proj = nn.Linear(config.n_embd, self.kv_lora_rank + self.q_lora_rank, bias=False)
        
        # 阶段2: KV 解压投影 - 从低维潜在空间解压到多头 KV 空间
        # [B, T, kv_lora_rank] -> [B, T, 2 * n_embd]
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, 2 * config.n_embd, bias=False)
        
        # 阶段3: Q 解压投影 - 从低维潜在空间解压到多头 Q 空间
        # [B, T, q_lora_rank] -> [B, T, n_embd]
        self.q_up_proj = nn.Linear(self.q_lora_rank, config.n_embd, bias=False)
        
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # 位置编码
        self.rope = rope
        self.alibi = alibi

    def forward(self, x, past_kv=None, use_cache=False):
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        B, T, C = x.size()
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        
        # === MLA 前向传播 ===
        
        # 步骤1: 压缩到低维潜在空间
        # [B, T, C] -> [B, T, kv_lora_rank + q_lora_rank]
        latent = self.down_proj(x)
        
        # 分离 KV 和 Q 的潜在表示
        kv_latent = latent[:, :, :self.kv_lora_rank]  # [B, T, kv_lora_rank]
        q_latent = latent[:, :, self.kv_lora_rank:]   # [B, T, q_lora_rank]
        
        # 步骤2: 从潜在空间解压 KV
        # [B, T, kv_lora_rank] -> [B, T, 2 * C]
        kv = self.kv_up_proj(kv_latent)
        k, v = kv.split(self.n_embd, dim=2)  # 分离 K 和 V，各为 [B, T, C]
        
        # 步骤3: 从潜在空间解压 Q
        # [B, T, q_lora_rank] -> [B, T, C]
        q = self.q_up_proj(q_latent)
        
        # 步骤4: 重塑为多头格式
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE（如果有）
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=past_len)
        
        # 拼接KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        # 应用 ALiBi（如果有）
        attn_bias = None
        if self.alibi is not None:
            attn_bias = self.alibi(T, k.size(2))  # [1, n_head, T, K]
        
        # 步骤5: 标准的多头注意力计算
        # [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]
        use_causal = past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=use_causal)
        
        # 步骤6: 合并多头输出
        # [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 步骤7: 输出投影
        y = self.c_proj(y)
        if use_cache:
            present_kv = (k, v)
            return y, present_kv
        return y

