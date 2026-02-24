"""
组件注册表
将字符串名称映射到具体实现类
"""

import torch.nn as nn

from modules.attentions import BaseAttention, MQAAttention, GQAAttention, MLAAttention
from modules.position_encodings import (
    LearnedPositionEncoding,
    ALiBi,
    RoPE,
    SinusoidalPositionalEncoding,
)
from modules.mlp import MLP, ReLUMLP, SiLUMLP, SwiGLUMLP, GeGLUMLP
from modules.moe_mlp import MoEMLP
from modules.normalizations import RMSNorm


ATTENTION_REGISTRY = {
    "base": BaseAttention,
    "mha": BaseAttention,
    "mqa": MQAAttention,
    "gqa": GQAAttention,
    "mla": MLAAttention,
}

POSITION_REGISTRY = {
    "learned": LearnedPositionEncoding,
    "alibi": ALiBi,
    "rope": RoPE,
    "sine": SinusoidalPositionalEncoding,
    "sinusoidal": SinusoidalPositionalEncoding,
}

MLP_REGISTRY = {
    "mlp": MLP,
    "relu": ReLUMLP,
    "silu": SiLUMLP,
    "swiglu": SwiGLUMLP,
    "geglu": GeGLUMLP,
    "moe": MoEMLP,
}

NORM_REGISTRY = {
    "layernorm": nn.LayerNorm,
    "ln": nn.LayerNorm,
    "rmsnorm": RMSNorm,
}


def _resolve_component(name, registry, kind, allow_none=False):
    if name is None:
        if allow_none:
            return None
        raise ValueError(f"{kind} 不能为空")
    key = str(name).lower()
    if key not in registry:
        choices = ", ".join(sorted(registry.keys()))
        raise ValueError(f"未知 {kind}: {name}，可选值: {choices}")
    return registry[key]


def resolve_attention(name):
    return _resolve_component(name, ATTENTION_REGISTRY, "attention")


def resolve_position(name):
    return _resolve_component(name, POSITION_REGISTRY, "position_encoding")


def resolve_mlp(name):
    return _resolve_component(name, MLP_REGISTRY, "mlp", allow_none=True)


def resolve_norm(name):
    return _resolve_component(name, NORM_REGISTRY, "norm", allow_none=True)
