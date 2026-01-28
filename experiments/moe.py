"""
MoE实验配置
Top-k Switch/Router MoE，作为MLP的替代
"""

import sys
import os
from dataclasses import dataclass

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import GPTConfig
from modules.attentions import BaseAttention
from modules.position_encodings import LearnedPositionEncoding
from modules.moe_mlp import MoEMLP


# 实验名称
EXPERIMENT_NAME = "moe"


@dataclass
class MoEConfig(GPTConfig):
    # MoE 参数
    n_experts: int = 4
    moe_top_k: int = 1
    moe_capacity_factor: float = 1.0
    moe_router_noise: float = 0.0
    moe_expert_type: str = "mlp"


# 模型配置
MODEL_CONFIG = MoEConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_experts=4,
    moe_top_k=1,
    moe_capacity_factor=1.0,
    moe_router_noise=0.0,
    moe_expert_type="mlp",
)

# 注意力机制
ATTENTION_CLASS = BaseAttention

# 位置编码
POSITION_ENCODING_CLASS = LearnedPositionEncoding

# MLP 替换为 MoE
MLP_CLASS = MoEMLP

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
    "moe_aux_weight": 0.01,
}
