"""
Transformer Block模块
"""

import torch.nn as nn


class Block(nn.Module):
    """
    Transformer Block
    包含LayerNorm、Attention、MLP等组件
    """
    
    def __init__(self, config, attention_class=None):
        """
        Args:
            config: 模型配置
            attention_class: 注意力机制类（如果为None，则从config.attention_class获取）
        """
        super().__init__()
        # LayerNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # CausalSelfAttention，将[B, T, C] -> [B, T, C]，不改变shape
        # 如果传入了attention_class，使用传入的；否则从config获取
        if attention_class is None:
            # 默认导入BaseAttention
            from GPT2.modules.attentions import BaseAttention
            attention_class = BaseAttention
        self.attn = attention_class(config)
        
        # LayerNorm, 将[B, T, C] -> [B, T, C]，不改变shape
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # MLP，将[B, T, C] -> [B, T, C]，不改变shape
        from GPT2.modules.mlp import MLP
        self.mlp = MLP(config)
    
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

