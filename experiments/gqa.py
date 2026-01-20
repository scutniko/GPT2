"""
GQA实验配置
使用Grouped Query Attention的GPT-2模型
"""

import sys
import os
from dataclasses import dataclass

# 添加GPT2父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
gpt2_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(gpt2_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from GPT2.core.config import GPTConfig
from GPT2.modules.attentions import GQAAttention
from GPT2.modules.position_encodings import LearnedPositionEncoding


# 实验名称
EXPERIMENT_NAME = "gqa"

# GQA需要额外的n_kv_head参数
@dataclass
class GQAConfig(GPTConfig):
    n_kv_head: int = 4  # KV头数量，12个Q头共享4个KV头

# 模型配置
MODEL_CONFIG = GQAConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_kv_head=4,  # 4 KV heads for 12 Q heads
)

# 注意力机制 - GQA
ATTENTION_CLASS = GQAAttention

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
    "log_dir": "/opt/train/data/LLM_Training/GPT2/log_train/gqa/log",
}

