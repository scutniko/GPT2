"""
MLP (多层感知机) 模块
"""

import torch.nn as nn


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

