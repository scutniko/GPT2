"""
统一的训练入口
支持所有消融实验的训练、恢复和推理
"""

import gc
import os
import sys

import tiktoken
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.checkpoint import (
    extract_config_from_checkpoint,
    ensure_log_file,
    load_checkpoint_low_mem,
    load_model_for_train,
    restore_loader_state,
    restore_optimizer_state,
    validate_checkpoint_v2,
)
from core.cli import parse_args
from core.data_loader import DataLoaderLite, validate_shard_lengths
from core.evaluator import run_inference
from core.experiment import load_experiment_spec
from core.runtime import cleanup_runtime, set_seed, setup_runtime
from core.trainer import run_train_loop
from models.gpt import GPT


def print_experiment_info(spec):
    """打印实验配置信息。"""
    print("=" * 60)
    print(f"实验: {spec.experiment_name}")
    print(f"注意力机制: {spec.attention_class.__name__}")
    print(f"位置编码: {spec.position_encoding_class.__name__}")
    if spec.mlp_class is not None:
        print(f"MLP类型: {spec.mlp_class.__name__}")
    if spec.norm_class is not None:
        print(f"归一化层: {spec.norm_class.__name__}")
    print("=" * 60)


def load_experiment_spec_or_exit(config_ref):
    """加载实验配置，失败则退出。"""
    try:
        return load_experiment_spec(config_ref)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)


def build_model_from_spec(spec, model_config=None):
    """按实验规范构建模型。"""
    cfg = model_config if model_config is not None else spec.model_config
    return GPT(
        cfg,
        attention_class=spec.attention_class,
        position_encoding_class=spec.position_encoding_class,
        mlp_class=spec.mlp_class,
        norm_class=spec.norm_class,
    )


def run_train_mode(args):
    """训练模式入口。"""
    spec = load_experiment_spec_or_exit(args.config)
    print_experiment_info(spec)

    runtime_ctx = setup_runtime()
    set_seed(1337)
    try:
        enc = tiktoken.get_encoding("gpt2")

        train_config = spec.train_config
        total_batch_size = train_config["total_batch_size"]
        B = train_config["micro_batch_size"]
        T = train_config["sequence_length"]
        assert total_batch_size % (B * T * runtime_ctx.ddp_world_size) == 0, (
            "total_batch_size必须能被B * T * ddp_world_size整除"
        )
        grad_accum_steps = total_batch_size // (B * T * runtime_ctx.ddp_world_size)

        if runtime_ctx.master_process:
            print(f"总batch size: {total_batch_size}")
            print(f"=> 梯度累积步数: {grad_accum_steps}")

        min_tokens_required = B * T * runtime_ctx.ddp_world_size + 1
        train_shard_stats = validate_shard_lengths(
            data_root=args.data_root,
            split="train",
            min_tokens_required=min_tokens_required,
        )
        val_shard_stats = validate_shard_lengths(
            data_root=args.data_root,
            split="val",
            min_tokens_required=min_tokens_required,
        )
        if runtime_ctx.master_process:
            print(
                "shard长度检查通过: "
                f"required_tokens>={min_tokens_required}; "
                f"train(num={train_shard_stats['num_shards']}, "
                f"min={train_shard_stats['min_tokens']}, max={train_shard_stats['max_tokens']}), "
                f"val(num={val_shard_stats['num_shards']}, "
                f"min={val_shard_stats['min_tokens']}, max={val_shard_stats['max_tokens']})"
            )

        train_loader = DataLoaderLite(
            B=B,
            T=T,
            process_rank=runtime_ctx.ddp_rank,
            num_processes=runtime_ctx.ddp_world_size,
            split="train",
            master_process=runtime_ctx.master_process,
            data_root=args.data_root,
        )
        val_loader = DataLoaderLite(
            B=B,
            T=T,
            process_rank=runtime_ctx.ddp_rank,
            num_processes=runtime_ctx.ddp_world_size,
            split="val",
            master_process=runtime_ctx.master_process,
            data_root=args.data_root,
        )

        torch.set_float32_matmul_precision("high")

        model = build_model_from_spec(spec)

        start_step, resume_optimizer_state, resume_loader_state = load_model_for_train(
            model=model,
            resume_path=args.resume,
            init_from_path=args.init_from,
            master_process=runtime_ctx.master_process,
        )

        model.to(runtime_ctx.device)
        use_compile = False  # torch.compile与HellaSwag评估和生成有冲突
        if use_compile:
            model = torch.compile(model)
        if runtime_ctx.ddp:
            model = DDP(model, device_ids=[runtime_ctx.ddp_local_rank], find_unused_parameters=True)
        raw_model = model.module if runtime_ctx.ddp else model

        optimizer = raw_model.configure_optimizers(
            weight_decay=train_config["weight_decay"],
            learning_rate=train_config["max_lr"],
            device_type=runtime_ctx.device_type,
            master_process=runtime_ctx.master_process,
        )

        restore_optimizer_state(
            optimizer=optimizer,
            resume_optimizer_state=resume_optimizer_state,
            device=runtime_ctx.device,
            master_process=runtime_ctx.master_process,
        )

        log_dir = os.path.join(current_dir, "log_train", spec.experiment_name, args.log_subdir)
        os.makedirs(log_dir, exist_ok=True)
        log_name = train_config.get("log_name", "log.txt")
        log_file = os.path.join(log_dir, log_name)
        ensure_log_file(log_file, resume=bool(args.resume))

        restore_loader_state(
            train_loader=train_loader,
            resume_loader_state=resume_loader_state,
            start_step=start_step,
            log_file=log_file,
            master_process=runtime_ctx.master_process,
        )

        run_train_loop(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            runtime_ctx=runtime_ctx,
            train_config=train_config,
            enc=enc,
            log_file=log_file,
            log_dir=log_dir,
            start_step=start_step,
            grad_accum_steps=grad_accum_steps,
            experiment_name=spec.experiment_name,
            config_ref=args.config,
            use_compile=use_compile,
        )
    finally:
        cleanup_runtime(runtime_ctx)


def run_infer_mode(args):
    """推理模式入口。"""
    runtime_ctx = setup_runtime()
    set_seed(args.seed)
    try:
        enc = tiktoken.get_encoding("gpt2")
        ckpt = load_checkpoint_low_mem(args.checkpoint, map_location="cpu")
        validate_checkpoint_v2(ckpt, required_fields=("model", "config_dict"))

        config_ref = args.config or ckpt.get("config_ref") or ckpt.get("experiment_name")
        if not config_ref:
            print("错误: infer 模式需要 --config，或 checkpoint 中包含 config_ref/experiment_name")
            sys.exit(1)

        spec = load_experiment_spec_or_exit(config_ref)
        if runtime_ctx.master_process:
            print_experiment_info(spec)
            if args.config is None:
                print(f"推理自动使用配置引用: {config_ref}")

        model_config = extract_config_from_checkpoint(ckpt)
        model = build_model_from_spec(spec, model_config=model_config)
        model.load_state_dict(ckpt["model"], strict=True)
        del ckpt
        gc.collect()

        run_inference(
            model=model,
            device=runtime_ctx.device,
            device_type=runtime_ctx.device_type,
            enc=enc,
            prompt=args.prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=args.seed,
            master_process=runtime_ctx.master_process,
        )
    finally:
        cleanup_runtime(runtime_ctx)


def main():
    args = parse_args()
    if args.command == "train":
        run_train_mode(args)
        return
    if args.command == "infer":
        run_infer_mode(args)
        return
    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
