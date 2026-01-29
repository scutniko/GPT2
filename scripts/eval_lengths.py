"""
Evaluate a saved checkpoint on multiple sequence lengths.
"""

import argparse
import contextlib
import importlib
import math
import os
import sys
import types

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.data_loader import DataLoaderLite
from models.gpt import GPT
from modules.position_encodings import LearnedPositionEncoding, SinusoidalPositionalEncoding


def _ensure_checkpoint_module_alias():
    # Old checkpoints may reference the project root as "GPT2.*"
    if "GPT2" in sys.modules:
        return
    pkg = types.ModuleType("GPT2")
    sys.modules["GPT2"] = pkg
    try:
        sys.modules["GPT2.core"] = importlib.import_module("core")
        sys.modules["GPT2.core.config"] = importlib.import_module("core.config")
        sys.modules["GPT2.core.data_loader"] = importlib.import_module("core.data_loader")
        sys.modules["GPT2.models"] = importlib.import_module("models")
        sys.modules["GPT2.models.gpt"] = importlib.import_module("models.gpt")
        sys.modules["GPT2.modules"] = importlib.import_module("modules")
        sys.modules["GPT2.modules.position_encodings"] = importlib.import_module(
            "modules.position_encodings"
        )
    except Exception:
        # Best-effort alias; torch.load will raise if something is still missing.
        pass


def parse_lengths(s):
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("序列长度列表为空")
    return items


def pick_device(name):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="多序列长度验证评估")
    parser.add_argument("--experiment", required=True, type=str, help="实验名，如 baseline/rope/alibi/sine")
    parser.add_argument("--checkpoint", required=True, type=str, help="checkpoint 路径")
    parser.add_argument("--lengths", required=True, type=str, help="逗号分隔的序列长度列表，如 512,1024,2048")
    parser.add_argument("--data_root", required=True, type=str, help="离线 token shard 目录（包含 train/val 的 .npy）")
    parser.add_argument("--val_steps", type=int, default=20, help="每个长度评估的 batch 数")
    parser.add_argument("--batch_size", type=int, default=None, help="评估 batch size，默认用训练配置的 micro_batch_size")
    parser.add_argument("--device", type=str, default=None, help="cuda/mps/cpu，默认自动选择")
    args = parser.parse_args()

    lengths = parse_lengths(args.lengths)
    max_len = max(lengths)

    try:
        exp_module = importlib.import_module(f"experiments.{args.experiment}")
    except ImportError as e:
        print(f"错误：找不到实验配置 '{args.experiment}'")
        raise e

    attention_class = exp_module.ATTENTION_CLASS
    position_encoding_class = exp_module.POSITION_ENCODING_CLASS
    mlp_class = getattr(exp_module, "MLP_CLASS", None)
    norm_class = getattr(exp_module, "NORM_CLASS", None)
    train_config = exp_module.TRAINING_CONFIG

    _ensure_checkpoint_module_alias()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    # Decide whether we can evaluate longer than training length
    drop_sine_buffer = False
    if position_encoding_class == LearnedPositionEncoding:
        if max_len > config.block_size:
            raise ValueError(
                f"Learned 位置编码不能外推：请求长度 {max_len} > 训练 block_size {config.block_size}"
            )
    elif position_encoding_class == SinusoidalPositionalEncoding:
        if max_len > config.block_size:
            config.block_size = max_len
            drop_sine_buffer = True
    else:
        if max_len > config.block_size:
            config.block_size = max_len

    model = GPT(
        config,
        attention_class=attention_class,
        position_encoding_class=position_encoding_class,
        mlp_class=mlp_class,
        norm_class=norm_class,
    )

    state_dict = ckpt["model"]
    if drop_sine_buffer:
        state_dict = {k: v for k, v in state_dict.items() if k != "pos_encoder.pe"}

    if drop_sine_buffer:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if k != "pos_encoder.pe"]
        if missing or unexpected:
            raise RuntimeError(f"state_dict 不匹配：missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(state_dict, strict=True)

    device = pick_device(args.device)
    model.to(device)
    model.eval()

    batch_size = args.batch_size if args.batch_size is not None else train_config["micro_batch_size"]
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else contextlib.nullcontext()
    )

    print(f"实验: {args.experiment}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"设备: {device}")
    print(f"评估 batch_size: {batch_size}, val_steps: {args.val_steps}")
    print("-" * 60)

    for T in lengths:
        if T > config.block_size:
            raise ValueError(f"请求长度 {T} 超过当前 block_size {config.block_size}")

        val_loader = DataLoaderLite(
            B=batch_size,
            T=T,
            process_rank=0,
            num_processes=1,
            split="val",
            master_process=True,
            data_root=args.data_root,
        )
        val_loader.reset()

        loss_accum = 0.0
        with torch.no_grad():
            for _ in range(args.val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with amp_ctx:
                    _, loss = model(x, y)
                loss_accum += loss.detach().float().item()

        loss_avg = loss_accum / args.val_steps
        ppl = math.exp(loss_avg)
        print(f"T={T:5d} | val_loss={loss_avg:.6f} | ppl={ppl:.4f}")


if __name__ == "__main__":
    main()
