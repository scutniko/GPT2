"""
各种归一化层实现
包含 LayerNorm, RMSNorm 等
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相比 LayerNorm 更简单高效，被 LLaMA 等模型采用
    
    RMSNorm 只做缩放归一化，不做平移，计算公式：
    RMSNorm(x) = x / RMS(x) * γ
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: 归一化的维度
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 γ
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, dim]
        Returns:
            normalized_x: [B, T, dim]
        """
        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并应用可学习的缩放参数
        x_normalized = x / rms
        
        return self.weight * x_normalized


# 方便使用的别名
LayerNorm = nn.LayerNorm

