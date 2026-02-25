"""
检查点读写与恢复逻辑

注意：当前仅支持 schema_version=2 的 checkpoint。
"""

import gc
import os
import tempfile
from dataclasses import fields

import torch

from core.config import GPTConfig
from core.data_loader import load_tokens

CHECKPOINT_SCHEMA_VERSION = 2


def _torch_load_with_fallback(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint_low_mem(path, map_location="cpu"):
    """
    低峰值内存加载checkpoint：
    - 优先使用 mmap 减少恢复时内存峰值
    - 兼容旧版本 PyTorch（无 mmap 参数）
    """
    return _torch_load_with_fallback(path, map_location=map_location)


def validate_checkpoint_v2(ckpt, required_fields=None):
    """校验 checkpoint 为 v2，并可选校验必填字段。"""
    schema_version = get_checkpoint_schema_version(ckpt)
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的 checkpoint schema_version={schema_version}，"
            f"当前仅支持 {CHECKPOINT_SCHEMA_VERSION}"
        )
    if required_fields:
        missing = [k for k in required_fields if k not in ckpt]
        if missing:
            raise KeyError(f"checkpoint 缺少字段: {', '.join(missing)}")


def get_checkpoint_schema_version(ckpt):
    """返回 checkpoint schema 版本。"""
    schema_version = ckpt.get("schema_version", None)
    if not isinstance(schema_version, int):
        raise KeyError("checkpoint 缺少 schema_version")
    return schema_version


def serialize_config(config):
    """将配置对象序列化为普通 dict。"""
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"不支持的配置类型: {type(config)}")


def deserialize_config(config_dict):
    """从配置 dict 反序列化为 GPTConfig（保留扩展字段）。"""
    if not isinstance(config_dict, dict):
        raise TypeError(f"config_dict 必须是 dict，实际是: {type(config_dict)}")
    core_field_names = {f.name for f in fields(GPTConfig)}
    core_kwargs = {}
    extra_kwargs = {}
    for key, value in config_dict.items():
        if key in core_field_names:
            core_kwargs[key] = value
        else:
            extra_kwargs[key] = value
    cfg = GPTConfig(**core_kwargs)
    for key, value in extra_kwargs.items():
        setattr(cfg, key, value)
    return cfg


def extract_config_from_checkpoint(ckpt):
    """
    从 checkpoint 中提取配置对象（仅支持 v2）。
    """
    validate_checkpoint_v2(ckpt, required_fields=("config_dict",))
    return deserialize_config(ckpt["config_dict"])


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
        validate_checkpoint_v2(ckpt, required_fields=("model", "step"))
        if not isinstance(ckpt["step"], int):
            raise TypeError(f"checkpoint.step 必须是 int，实际是: {type(ckpt['step'])}")
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
        validate_checkpoint_v2(ckpt, required_fields=("model",))
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

    if "current_shard" not in resume_loader_state or "current_position" not in resume_loader_state:
        raise KeyError("train_loader_state 缺少 current_shard/current_position")

    current_shard = int(resume_loader_state["current_shard"])
    rank0_position = int(resume_loader_state["current_position"])
    rank_offset = train_loader.B * train_loader.T * train_loader.process_rank
    restored_position = rank0_position + rank_offset

    if current_shard < 0 or current_shard >= len(train_loader.shards):
        raise ValueError(f"非法 current_shard={current_shard}")
    if rank0_position < 0:
        raise ValueError(f"非法 current_position={rank0_position}")

    train_loader.current_shard = current_shard
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    if restored_position + (train_loader.B * train_loader.T + 1) > len(train_loader.tokens):
        raise ValueError(
            "恢复的数据加载器位置越界，checkpoint 可能损坏或与当前数据分片不匹配: "
            f"shard={train_loader.current_shard}, position={restored_position}, "
            f"tokens={len(train_loader.tokens)}"
        )
    train_loader.current_position = restored_position

    if not master_process:
        return

    print(
        f"✓ 恢复数据加载器状态: shard {train_loader.current_shard}, "
        f"rank0_position {rank0_position}, 当前rank_position {train_loader.current_position}"
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


def save_checkpoint(
    path,
    raw_model,
    optimizer,
    train_loader,
    step,
    val_loss,
    experiment_name=None,
    config_ref=None,
):
    """保存训练检查点。"""
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": raw_model.state_dict(),
        "config_dict": serialize_config(raw_model.config),
        "step": step,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict(),
        "train_loader_state": {
            "current_shard": train_loader.current_shard,
            "current_position": train_loader.current_position,
        },
    }
    if experiment_name is not None:
        checkpoint["experiment_name"] = experiment_name
    if config_ref is not None:
        checkpoint["config_ref"] = config_ref
    _atomic_torch_save(checkpoint, path)


def _atomic_torch_save(obj, path):
    """
    原子写入 checkpoint：
    先写临时文件，再通过 os.replace 原子替换目标文件。
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pt", dir=dir_path)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
