"""
运行时环境初始化（设备、DDP、随机种子）
"""

import os
from contextlib import nullcontext
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


def infer_device_type(device):
    """根据设备字符串返回 device_type（cuda/mps/cpu）。"""
    if str(device).startswith("cuda"):
        return "cuda"
    if str(device).startswith("mps"):
        return "mps"
    return "cpu"


def get_autocast_context(device_type):
    """
    返回与设备匹配的 autocast 上下文。
    - cuda: bfloat16
    - mps: float16（不支持时降级为 nullcontext）
    - cpu: bfloat16（不支持时降级为 nullcontext）
    """
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device_type == "mps":
        try:
            return torch.autocast(device_type="mps", dtype=torch.float16)
        except (TypeError, RuntimeError):
            return nullcontext()
    if device_type == "cpu":
        try:
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        except (TypeError, RuntimeError):
            return nullcontext()
    return nullcontext()


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

    device_type = infer_device_type(device)
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
