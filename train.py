"""
统一的训练入口
支持两种配置来源：
1) 兼容旧版 --experiment（experiments/*.py）
2) 新增 --config（YAML）
"""

import argparse
import gc
import os
import sys

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.checkpoint import load_checkpoint_low_mem, optimizer_to_device
from core.config_loader import load_experiment_spec
from core.data_loader import DataLoaderLite
from core.evaluator import generate_samples
from core.runtime import setup_runtime, teardown_runtime, seed_everything
from core.trainer import Trainer
from models.gpt import GPT


def _build_parser():
    parser = argparse.ArgumentParser(description="GPT-2 Training with Ablation Studies")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="实验名称（兼容模式）：baseline, alibi, rope, sine, mqa, gqa, mla ...",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML配置路径（推荐），如 configs/experiments/baseline.yaml",
    )
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--init_from", type=str, default=None, help="仅加载模型权重，不恢复优化器和step")
    parser.add_argument("--inference", type=str, default=None, help="推理模式：加载检查点生成文本")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="离线token shard目录（训练模式必填）",
    )
    parser.add_argument(
        "--log_subdir",
        type=str,
        default=None,
        help="日志与checkpoint子目录（训练模式必填）",
    )
    return parser


def _validate_args(parser, args):
    if args.experiment and args.config:
        parser.error("--experiment 与 --config 不能同时使用")
    if not args.experiment and not args.config:
        parser.error("必须提供 --experiment 或 --config 其中之一")
    if args.resume and args.init_from:
        parser.error("--resume 与 --init_from 不能同时使用")

    is_training_mode = args.inference is None
    if is_training_mode:
        if not args.data_root:
            parser.error("训练模式必须提供 --data_root")
        if not args.log_subdir:
            parser.error("训练模式必须提供 --log_subdir")


def _print_experiment(spec):
    print("=" * 60)
    print(f"实验: {spec.experiment_name}")
    print(f"注意力机制: {spec.attention_class.__name__}")
    print(f"位置编码: {spec.position_encoding_class.__name__}")
    if spec.mlp_class is not None:
        print(f"MLP类型: {spec.mlp_class.__name__}")
    if spec.norm_class is not None:
        print(f"归一化层: {spec.norm_class.__name__}")
    print("=" * 60)


def _validate_train_config(train_config):
    required_keys = [
        "max_lr",
        "min_lr",
        "warmup_steps",
        "max_steps",
        "weight_decay",
        "total_batch_size",
        "micro_batch_size",
        "sequence_length",
    ]
    missing = [key for key in required_keys if key not in train_config]
    if missing:
        raise ValueError(f"训练配置缺少字段: {', '.join(missing)}")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    try:
        spec = load_experiment_spec(experiment=args.experiment, config_path=args.config)
    except ImportError:
        print(f"错误: 找不到实验配置 '{args.experiment}'")
        print("可用的实验:")
        print("  位置编码: baseline, alibi, rope, sine")
        print("  注意力机制: mqa, gqa, mla")
        print("  激活函数: relu, silu, swiglu, geglu")
        print("  MoE: moe")
        print("  归一化层: rmsnorm")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载配置失败 - {e}")
        sys.exit(1)

    runtime = setup_runtime()
    try:
        _print_experiment(spec)
        seed_everything(1337)
        enc = tiktoken.get_encoding("gpt2")

        model = GPT(
            spec.model_config,
            spec.attention_class,
            spec.position_encoding_class,
            mlp_class=spec.mlp_class,
            norm_class=spec.norm_class,
        )

        start_step = 0
        resume_optimizer_state = None
        resume_loader_state = None
        if args.resume:
            ckpt = load_checkpoint_low_mem(args.resume, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            start_step = ckpt["step"] + 1
            resume_optimizer_state = ckpt.get("optimizer", None)
            resume_loader_state = ckpt.get("train_loader_state", None)
            del ckpt
            gc.collect()
            if runtime.master_process:
                print(f"✓ 从 {args.resume} 恢复训练 (第 {start_step} 步开始)")
        elif args.init_from:
            ckpt = load_checkpoint_low_mem(args.init_from, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            del ckpt
            gc.collect()
            if runtime.master_process:
                print(f"✓ 已加载模型权重（不恢复优化器/step）: {args.init_from}")

        # 推理模式不再强制 data_root/log_subdir
        if args.inference:
            if runtime.master_process:
                print(f"✓ 推理模式：加载模型权重 {args.inference}")
            ckpt = load_checkpoint_low_mem(args.inference, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            del ckpt
            gc.collect()

            model.to(runtime.device)
            outputs = generate_samples(
                model=model,
                enc=enc,
                device=runtime.device,
                device_type=runtime.device_type,
                prompt="Hello, I'm a language model,",
                num_return_sequences=5,
                max_length=32,
                seed=42,
            )
            if runtime.master_process:
                print(f"\n{'=' * 60}")
                print("提示词: Hello, I'm a language model,")
                print(f"{'=' * 60}\n")
                for i, decoded in enumerate(outputs):
                    print(f"生成 {i + 1}: {decoded}\n")
            return

        train_config = spec.train_config
        _validate_train_config(train_config)

        total_batch_size = train_config["total_batch_size"]
        B = train_config["micro_batch_size"]
        T = train_config["sequence_length"]
        assert total_batch_size % (B * T * runtime.ddp_world_size) == 0, (
            "total_batch_size必须能被B * T * ddp_world_size整除"
        )
        grad_accum_steps = total_batch_size // (B * T * runtime.ddp_world_size)

        if runtime.master_process:
            print(f"总batch size: {total_batch_size}")
            print(f"=> 梯度累积步数: {grad_accum_steps}")

        train_loader = DataLoaderLite(
            B=B,
            T=T,
            process_rank=runtime.ddp_rank,
            num_processes=runtime.ddp_world_size,
            split="train",
            master_process=runtime.master_process,
            data_root=args.data_root,
        )
        val_loader = DataLoaderLite(
            B=B,
            T=T,
            process_rank=runtime.ddp_rank,
            num_processes=runtime.ddp_world_size,
            split="val",
            master_process=runtime.master_process,
            data_root=args.data_root,
        )

        torch.set_float32_matmul_precision("high")
        model.to(runtime.device)

        use_compile = bool(train_config.get("use_compile", False))
        if use_compile:
            model = torch.compile(model)
        if runtime.ddp:
            find_unused = bool(train_config.get("ddp_find_unused_parameters", True))
            model = DDP(model, device_ids=[runtime.ddp_local_rank], find_unused_parameters=find_unused)
        raw_model = model.module if runtime.ddp else model

        max_lr = train_config["max_lr"]
        weight_decay = train_config["weight_decay"]

        optimizer = raw_model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=max_lr,
            device_type=runtime.device_type,
            master_process=runtime.master_process,
        )

        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
            optimizer_to_device(optimizer, runtime.device)
            del resume_optimizer_state
            gc.collect()
            if runtime.master_process:
                print("✓ 恢复优化器状态")

        log_dir = os.path.join(current_dir, "log_train", spec.experiment_name, args.log_subdir)
        os.makedirs(log_dir, exist_ok=True)
        log_name = train_config.get("log_name", "log.txt")
        log_file = os.path.join(log_dir, log_name)

        if not args.resume:
            with open(log_file, "w", encoding="utf-8"):
                pass

        trainer = Trainer(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            runtime=runtime,
            train_config=train_config,
            log_dir=log_dir,
            log_file=log_file,
            enc=enc,
        )
        trainer.restore_train_loader_state(resume_loader_state, start_step)
        trainer.fit(start_step=start_step, grad_accum_steps=grad_accum_steps)
    finally:
        teardown_runtime(runtime)


if __name__ == "__main__":
    main()
