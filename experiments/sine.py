"""
Sinusoidal Position Encoding实验配置
使用正弦位置编码的GPT-2模型
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GPTConfig
from modules.attentions import BaseAttention
from modules.position_encodings import SinusoidalPositionalEncoding


# 实验名称
EXPERIMENT_NAME = "sine"

# 模型配置
MODEL_CONFIG = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
)

# 注意力机制
ATTENTION_CLASS = BaseAttention

# 位置编码 - Sinusoidal
POSITION_ENCODING_CLASS = SinusoidalPositionalEncoding

# 训练配置
TRAINING_CONFIG = {
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 715,
    "max_steps": 19073,
    "weight_decay": 0.1,
    "total_batch_size": 524288,
    "micro_batch_size": 8,
    "sequence_length": 1024,
    "log_dir": os.path.join(project_root, "log_train", EXPERIMENT_NAME, "log"),
}

