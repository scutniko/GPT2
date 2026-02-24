"""
运行时环境管理
包含设备选择、DDP初始化与随机种子设置
"""

from dataclasses import dataclass
import os

import torch
from torch.distributed import init_process_group, destroy_process_group


@dataclass
class RuntimeContext:
    ddp: bool
    ddp_rank: int
    ddp_local_rank: int
    ddp_world_size: int
    device: str
    device_type: str
    master_process: bool


def setup_runtime():
    """初始化运行时上下文（单卡/DDP）。"""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP需要CUDA支持"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"使用设备: {device}")

    # 与原行为保持一致：除CUDA外统一按cpu autocast分支处理
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return RuntimeContext(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        device_type=device_type,
        master_process=master_process,
    )


def teardown_runtime(runtime):
    """销毁DDP进程组。"""
    if runtime.ddp:
        destroy_process_group()


def seed_everything(seed):
    """设置随机种子。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
