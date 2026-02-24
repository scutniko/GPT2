"""
检查点读写与恢复逻辑
"""

import gc
import os

import torch

from core.data_loader import load_tokens


def load_checkpoint_low_mem(path, map_location="cpu"):
    """
    低峰值内存加载checkpoint：
    - 优先使用 mmap 减少恢复时内存峰值
    - 兼容旧版本 PyTorch（无 mmap 参数）
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def optimizer_to_device(optimizer, device):
    """将优化器状态中的Tensor迁移到指定设备。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def load_model_for_train(model, resume_path=None, init_from_path=None, master_process=True):
    """
    按训练模式加载权重/恢复状态。

    Returns:
        start_step, resume_optimizer_state, resume_loader_state
    """
    start_step = 0
    resume_optimizer_state = None
    resume_loader_state = None

    if resume_path:
        ckpt = load_checkpoint_low_mem(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
        resume_optimizer_state = ckpt.get("optimizer", None)
        resume_loader_state = ckpt.get("train_loader_state", None)
        del ckpt
        gc.collect()
        if master_process:
            print(f"✓ 从 {resume_path} 恢复训练 (第 {start_step} 步开始)")
    elif init_from_path:
        ckpt = load_checkpoint_low_mem(init_from_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        del ckpt
        gc.collect()
        if master_process:
            print(f"✓ 已加载模型权重（不恢复优化器/step）: {init_from_path}")

    return start_step, resume_optimizer_state, resume_loader_state


def restore_optimizer_state(optimizer, resume_optimizer_state, device, master_process=True):
    """恢复优化器状态。"""
    if resume_optimizer_state is None:
        return
    optimizer.load_state_dict(resume_optimizer_state)
    optimizer_to_device(optimizer, device)
    del resume_optimizer_state
    gc.collect()
    if master_process:
        print("✓ 恢复优化器状态")


def ensure_log_file(log_file, resume=False):
    """在非恢复模式下清空日志文件。"""
    if not resume:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("")


def restore_loader_state(train_loader, resume_loader_state, start_step, log_file, master_process=True):
    """恢复数据加载器状态并裁剪日志。"""
    if resume_loader_state is None:
        return

    train_loader.current_shard = resume_loader_state["current_shard"]
    train_loader.current_position = resume_loader_state["current_position"]
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])

    if not master_process:
        return

    print(
        f"✓ 恢复数据加载器状态: shard {train_loader.current_shard}, "
        f"position {train_loader.current_position}"
    )

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(log_file, "w", encoding="utf-8") as f:
            for line in lines:
                try:
                    step_in_line = int(line.split()[0])
                    if step_in_line < start_step:
                        f.write(line)
                except (ValueError, IndexError):
                    f.write(line)
        print(f"✓ 日志文件已截断至第 {start_step} 步之前")


def save_checkpoint(path, raw_model, optimizer, train_loader, step, val_loss):
    """保存训练检查点。"""
    checkpoint = {
        "model": raw_model.state_dict(),
        "config": raw_model.config,
        "step": step,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict(),
        "train_loader_state": {
            "current_shard": train_loader.current_shard,
            "current_position": train_loader.current_position,
        },
    }
    torch.save(checkpoint, path)

