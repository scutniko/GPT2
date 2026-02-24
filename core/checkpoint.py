"""
checkpoint工具
包含低峰值加载与优化器状态迁移
"""

import torch


def load_checkpoint_low_mem(path, map_location="cpu"):
    """
    低峰值内存加载checkpoint：
    - 优先使用 mmap 降低恢复内存峰值
    - 兼容旧版本 PyTorch（无 mmap 参数）
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def optimizer_to_device(optimizer, device):
    """将优化器状态中的Tensor迁移到指定设备。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)
