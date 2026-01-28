"""
MQA实验配置
使用Multi-Query Attention的GPT-2模型
"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GPTConfig
from modules.attentions import MQAAttention
from modules.position_encodings import LearnedPositionEncoding


# 实验名称
EXPERIMENT_NAME = "mqa"

# 模型配置
MODEL_CONFIG = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
)

# 注意力机制 - MQA
ATTENTION_CLASS = MQAAttention

# 位置编码
POSITION_ENCODING_CLASS = LearnedPositionEncoding

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

