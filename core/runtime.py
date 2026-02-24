"""
运行时环境初始化（设备、DDP、随机种子）
"""

import os
from dataclasses import dataclass

import torch
from torch.distributed import destroy_process_group, init_process_group


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
    """
    初始化分布式上下文与设备信息。
    """
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


def set_seed(seed=1337):
    """设置随机种子。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def cleanup_runtime(ctx):
    """释放分布式资源。"""
    if ctx.ddp:
        destroy_process_group()

