"""
MLP (多层感知机) 模块
支持多种激活函数：GELU, ReLU, SiLU/Swish, SwiGLU, GeGLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    标准的MLP模块（GPT-2）
    包含两个线性层和GELU激活函数
    """
    
    def __init__(self, config):
        super().__init__()
        # 全连接层，将[B, T, C] -> [B, T, 4 * C]
        # 在GPT-2中，MLP的中间层是4倍于输入层的大小。
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU激活函数
        self.gelu = nn.GELU(approximate='tanh')
        # 全连接层，将[B, T, 4 * C] -> [B, T, C]
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # 初始化权重，使得输出方差与输入方差一致
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
            
        Returns:
            x: [B, T, C]
        """
        # 输入x的shape是[B, T, C]，最终输出y的shape是[B, T, C]
        # C是n_embd，即Transformer的通道数
        # C=n_head * head_size
        # GPT-2 (124M), n_head=12, head_size=64, 所以 n_head * head_size=C=768 即Transformer的通道数
        x = self.c_fc(x)
        # 对x进行线性变换，得到[B, T, 4 * C]
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class ReLUMLP(nn.Module):
    """
    使用ReLU激活函数的MLP
    ReLU是最经典的激活函数，计算简单但可能存在"神经元死亡"问题
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
        Returns:
            x: [B, T, C]
        """
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x


class SiLUMLP(nn.Module):
    """
    使用SiLU/Swish激活函数的MLP
    SiLU(x) = x * sigmoid(x)，平滑且非单调，被许多现代模型采用
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.silu = nn.SiLU()  # Swish
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
        Returns:
            x: [B, T, C]
        """
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        return x


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP (Swish-Gated Linear Unit)
    使用门控机制，被 PaLM、LLaMA 等模型采用
    参考: https://arxiv.org/abs/2002.05202
    
    SwiGLU(x) = SiLU(W_gate * x) ⊙ (W_up * x) * W_down
    其中 ⊙ 表示逐元素乘法（门控机制）
    """
    
    def __init__(self, config):
        super().__init__()
        # SwiGLU 使用 2/3 * 4 * d 的中间维度以保持参数量接近标准MLP
        # 标准MLP: 2 * (d * 4d) = 8d^2
        # SwiGLU: 2 * (d * hidden_dim) + hidden_dim * d = 3 * d * hidden_dim
        # 令 3 * d * hidden_dim ≈ 8d^2，得 hidden_dim ≈ 8d/3
        hidden_dim = int(8 * config.n_embd / 3)
        # 确保是8的倍数，有利于GPU计算效率
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        
        # 门控投影（用于计算门控值）
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        # 上投影（用于计算特征）
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        # 下投影
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
        Returns:
            x: [B, T, C]
        """
        # 门控机制：SwiGLU(x) = SiLU(W_gate * x) ⊙ (W_up * x)
        gate = F.silu(self.w_gate(x))  # [B, T, hidden_dim]
        up = self.w_up(x)              # [B, T, hidden_dim]
        x = gate * up                   # 逐元素乘法，实现门控
        x = self.w_down(x)              # [B, T, C]
        return x


class GeGLUMLP(nn.Module):
    """
    GeGLU MLP (GELU-Gated Linear Unit)
    使用 GELU 作为门控激活函数的门控MLP
    参考: https://arxiv.org/abs/2002.05202
    
    GeGLU(x) = GELU(W_gate * x) ⊙ (W_up * x) * W_down
    """
    
    def __init__(self, config):
        super().__init__()
        # GeGLU 同样使用 8d/3 的中间维度以保持参数量接近
        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        
        # 门控投影
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        # 上投影
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        # 下投影
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1
        
        # GELU 激活
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
        Returns:
            x: [B, T, C]
        """
        # 门控机制：GeGLU(x) = GELU(W_gate * x) ⊙ (W_up * x)
        gate = self.gelu(self.w_gate(x))  # [B, T, hidden_dim]
        up = self.w_up(x)                  # [B, T, hidden_dim]
        x = gate * up                       # 逐元素乘法，实现门控
        x = self.w_down(x)                  # [B, T, C]
        return x
