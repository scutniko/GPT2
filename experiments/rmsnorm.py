"""
RMSNorm归一化层实验配置
测试使用RMSNorm替代LayerNorm的效果
被LLaMA等模型采用
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
from modules.position_encodings import LearnedPositionEncoding
from modules.normalizations import RMSNorm


# 实验名称
EXPERIMENT_NAME = "rmsnorm"

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

# 位置编码
POSITION_ENCODING_CLASS = LearnedPositionEncoding

# 归一化层 - 使用RMSNorm替代LayerNorm
NORM_CLASS = RMSNorm

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
    "data_root": os.environ.get("DATASET_ROOT", "/opt/train/data/nanogpt/edu_fineweb10B"),
    "log_dir": os.path.join(project_root, "log_train", EXPERIMENT_NAME, "log"),
}

