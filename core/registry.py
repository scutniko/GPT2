"""
组件注册表：将 YAML 中的字符串键映射到具体实现类。
"""

import torch.nn as nn

from modules.attentions import BaseAttention, GQAAttention, MLAAttention, MQAAttention
from modules.mlp import GeGLUMLP, MLP, ReLUMLP, SiLUMLP, SwiGLUMLP
from modules.moe_mlp import MoEMLP
from modules.normalizations import RMSNorm
from modules.position_encodings import ALiBi, LearnedPositionEncoding, RoPE, SinusoidalPositionalEncoding


COMPONENT_REGISTRY = {
    "attention": {
        "base": BaseAttention,
        "mqa": MQAAttention,
        "gqa": GQAAttention,
        "mla": MLAAttention,
    },
    "position_encoding": {
        "learned": LearnedPositionEncoding,
        "alibi": ALiBi,
        "rope": RoPE,
        "sine": SinusoidalPositionalEncoding,
        "sinusoidal": SinusoidalPositionalEncoding,
    },
    "mlp": {
        "default": MLP,
        "mlp": MLP,
        "relu": ReLUMLP,
        "silu": SiLUMLP,
        "swiglu": SwiGLUMLP,
        "geglu": GeGLUMLP,
        "moe": MoEMLP,
    },
    "norm": {
        "default": nn.LayerNorm,
        "layernorm": nn.LayerNorm,
        "rmsnorm": RMSNorm,
    },
}


def resolve_component(component_type, key):
    """
    将组件类型与键解析为实现类。
    """
    if component_type not in COMPONENT_REGISTRY:
        raise ValueError(f"未知组件类型: {component_type}")
    normalized_key = str(key).strip().lower()
    mapping = COMPONENT_REGISTRY[component_type]
    if normalized_key not in mapping:
        candidates = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"组件 {component_type} 不支持 '{key}'，可选值: {candidates}"
        )
    return mapping[normalized_key]

