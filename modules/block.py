"""
Transformer Block模块
支持可配置的注意力机制、MLP和归一化层
"""

import torch.nn as nn

from modules.attentions import BaseAttention
from modules.mlp import MLP


class Block(nn.Module):
    """
    Transformer Block
    包含LayerNorm、Attention、MLP等组件
    支持可配置的归一化层和MLP类型
    """
    
    def __init__(self, config, attention_class=None, mlp_class=None, 
                 norm_class=None, rope=None, alibi=None):
        """
        Args:
            config: 模型配置
            attention_class: 注意力机制类（如果为None，则使用BaseAttention）
            mlp_class: MLP类（如果为None，则使用标准MLP）
            norm_class: 归一化层类（如果为None，则使用LayerNorm）
            rope: RoPE位置编码对象（可选）
            alibi: ALiBi位置编码对象（可选）
        """
        super().__init__()
        
        # 选择归一化层类型
        if norm_class is None:
            norm_class = nn.LayerNorm
        
        # LayerNorm/RMSNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_1 = norm_class(config.n_embd)
        
        # CausalSelfAttention，将[B, T, C] -> [B, T, C]，不改变shape
        # 如果传入了attention_class，使用传入的；否则使用BaseAttention
        if attention_class is None:
            attention_class = BaseAttention
        self.attn = attention_class(config, rope=rope, alibi=alibi)
        
        # LayerNorm/RMSNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_2 = norm_class(config.n_embd)
        
        # 选择MLP类型
        if mlp_class is None:
            mlp_class = MLP
        
        # MLP，将[B, T, C] -> [B, T, C]，不改变shape
        self.mlp = mlp_class(config)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, T, C]
            
        Returns:
            x: [B, T, C]
        """
        # 将x与self.attn(self.ln_1(x))相加，得到[B, T, C]
        x = x + self.attn(self.ln_1(x))
        # 将x与self.mlp(self.ln_2(x))相加，得到[B, T, C]
        x = x + self.mlp(self.ln_2(x))
        return x

