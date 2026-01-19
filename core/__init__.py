"""
核心公共代码模块
包含配置、数据加载、训练工具等
"""

from .config import GPTConfig
from .data_loader import DataLoaderLite, load_tokens
from .training_utils import get_lr, get_most_likely_row

__all__ = [
    'GPTConfig',
    'DataLoaderLite',
    'load_tokens',
    'get_lr',
    'get_most_likely_row',
]

