"""
KV cache 推理评估脚本
对比不使用KV cache与使用KV cache的生成速度与输出一致性
"""

import os
import glob
import time
import argparse
import importlib
import sys
import types

import torch
import torch.nn.functional as F
import tiktoken

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.gpt import GPT

from modules.position_encodings import LearnedPositionEncoding, SinusoidalPositionalEncoding, RoPE, ALiBi
from core.config import GPTConfig

def _sync_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _find_latest_checkpoint(log_dir):
    pattern = os.path.join(log_dir, "model_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    def _step_from_name(path):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        step_str = stem.split("_")[-1]
        try:
            return int(step_str)
        except ValueError:
            return -1
    return max(candidates, key=_step_from_name)


@torch.no_grad()
def generate_no_cache(model, input_ids, max_length, top_k, seed, device, use_autocast, block_size=None):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = input_ids
    _sync_if_cuda(device)
    t0 = time.time()
    while x.size(1) < max_length:
        if block_size is not None and x.size(1) > block_size:
            x_ctx = x[:, -block_size:]
        else:
            x_ctx = x
        if use_autocast:
            with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                logits, _ = model(x_ctx)
        else:
            logits, _ = model(x_ctx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=generator)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
    _sync_if_cuda(device)
    t1 = time.time()
    return x, t1 - t0


@torch.no_grad()
def generate_with_cache(model, input_ids, max_length, top_k, seed, device, use_autocast):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = input_ids
    past_kv = None
    _sync_if_cuda(device)
    t0 = time.time()
    while x.size(1) < max_length:
        if past_kv is None:
            if use_autocast:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                    logits, _, past_kv = model(x, past_kv=None, use_cache=True)
            else:
                logits, _, past_kv = model(x, past_kv=None, use_cache=True)
        else:
            if use_autocast:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                    logits, _, past_kv = model(x[:, -1:], past_kv=past_kv, use_cache=True)
            else:
                logits, _, past_kv = model(x[:, -1:], past_kv=past_kv, use_cache=True)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=generator)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
    _sync_if_cuda(device)
    t1 = time.time()
    return x, t1 - t0


def _reset_cuda_peak_memory(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _get_cuda_peak_memory_mb(device):
    if str(device).startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def main():
    parser = argparse.ArgumentParser(description="KV cache 推理评估")
    parser.add_argument("--experiment", type=str, required=True, help="实验名称")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径（可选，默认自动找最新）")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,", help="提示词")
    parser.add_argument("--max_length", type=int, default=64, help="生成最大长度")
    parser.add_argument("--top_k", type=int, default=50, help="top-k 采样")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="生成条数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="推理精度")
    parser.add_argument("--allow_long", action="store_true", help="允许生成长度超过训练长度（仅RoPE/ALiBi/正弦）")
    args = parser.parse_args()

    try:
        exp_module = importlib.import_module(f"experiments.{args.experiment}")
    except ImportError:
        print(f"错误: 找不到实验配置 '{args.experiment}'")
        return

    experiment_name = exp_module.EXPERIMENT_NAME
    model_config = exp_module.MODEL_CONFIG
    attention_class = exp_module.ATTENTION_CLASS
    position_encoding_class = exp_module.POSITION_ENCODING_CLASS
    mlp_class = getattr(exp_module, "MLP_CLASS", None)
    norm_class = getattr(exp_module, "NORM_CLASS", None)
    train_config = exp_module.TRAINING_CONFIG

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"实验: {experiment_name}")
    print(f"注意力机制: {attention_class.__name__}")
    print(f"位置编码: {position_encoding_class.__name__}")
    print(f"使用设备: {device}")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        log_dir = train_config["log_dir"]
        checkpoint_path = _find_latest_checkpoint(log_dir)
        if checkpoint_path is None:
            print(f"错误: 在 {log_dir} 未找到任何检查点")
            return
        print(f"未指定检查点，自动使用最新检查点: {checkpoint_path}")
    else:
        print(f"使用检查点: {checkpoint_path}")

    model = GPT(model_config, attention_class, position_encoding_class,
                mlp_class=mlp_class, norm_class=norm_class)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"提示: 缺失参数 {missing}")
    if unexpected:
        print(f"提示: 多余参数 {unexpected}")
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(args.num_return_sequences, 1)
    block_size = model_config.block_size
    prompt_len = tokens.size(1)

    if args.max_length > block_size:
        if not args.allow_long:
            print(f"错误: max_length={args.max_length} 超过训练长度 block_size={block_size}")
            print("可加 --allow_long 尝试长序列推理（仅RoPE/ALiBi/正弦）")
            return
        if position_encoding_class == LearnedPositionEncoding:
            print("错误: LearnedPositionEncoding 不支持超过训练长度的推理")
            return
        if position_encoding_class not in {RoPE, ALiBi, SinusoidalPositionalEncoding}:
            print("错误: 该位置编码不支持超过训练长度的推理")
            return
        if prompt_len > block_size:
            print(f"错误: 提示词长度 {prompt_len} 超过 block_size={block_size}")
            return
        print(f"提示: 将生成超过训练长度的序列 (max_length={args.max_length}, block_size={block_size})")

    if position_encoding_class == SinusoidalPositionalEncoding:
        model.pos_encoder.ensure_cache(
            max_seq_len=args.max_length,
            device=tokens.device,
            dtype=torch.float32,
        )

    use_autocast = args.dtype == "bfloat16" and device != "cpu"
    if use_autocast:
        print("推理精度: bfloat16")
    else:
        print("推理精度: float32")

    run_no_cache = args.max_length <= block_size
    out_no_cache = None
    time_no_cache = None
    peak_no_cache = None
    if run_no_cache:
        _reset_cuda_peak_memory(device)
        out_no_cache, time_no_cache = generate_no_cache(
            model, tokens, args.max_length, args.top_k, args.seed, device, use_autocast, block_size=block_size
        )
        peak_no_cache = _get_cuda_peak_memory_mb(device)
    else:
        print("提示: 超过训练长度时仅使用KV cache路径生成，不运行无cache路径")

    _reset_cuda_peak_memory(device)
    out_cache, time_cache = generate_with_cache(
        model, tokens, args.max_length, args.top_k, args.seed, device, use_autocast
    )
    peak_cache = _get_cuda_peak_memory_mb(device)

    prompt_len = tokens.size(1)
    gen_tokens = (args.max_length - prompt_len) * args.num_return_sequences
    tok_per_sec_no_cache = None
    if run_no_cache:
        tok_per_sec_no_cache = gen_tokens / max(time_no_cache, 1e-9)
    tok_per_sec_cache = gen_tokens / max(time_cache, 1e-9)

    print("\n==== 速度对比 ====")
    if run_no_cache:
        print(f"不使用KV cache: {time_no_cache:.4f}s, {tok_per_sec_no_cache:.2f} tok/s")
    else:
        print("不使用KV cache: 跳过（超出训练长度）")
    print(f"使用KV cache:  {time_cache:.4f}s, {tok_per_sec_cache:.2f} tok/s")
    if run_no_cache and time_cache > 0:
        print(f"速度提升: {time_no_cache / time_cache:.2f}x")

    print("\n==== 一致性检查 ====")
    if run_no_cache:
        same = torch.equal(out_no_cache, out_cache)
        print(f"逐token输出一致: {'是' if same else '否'}")
        if not same:
            mismatch = (out_no_cache != out_cache).nonzero(as_tuple=False)
            first = mismatch[0].tolist()
            print(f"首个不一致位置: batch={first[0]}, pos={first[1]}")
    else:
        print("逐token输出一致: 不检查（仅运行KV cache路径）")

    print("\n==== 显存占用（CUDA） ====")
    if str(device).startswith("cuda"):
        if run_no_cache:
            print(f"不使用KV cache: 峰值 {peak_no_cache:.2f} MB")
        else:
            print("不使用KV cache: 跳过")
        print(f"使用KV cache:  峰值 {peak_cache:.2f} MB")
    else:
        print("当前设备非CUDA，无法统计显存")

    if run_no_cache:
        print("\n==== 生成结果（不使用KV cache） ====")
        for i in range(args.num_return_sequences):
            decoded = enc.decode(out_no_cache[i].tolist())
            print(f"生成 {i + 1}: {decoded}")

    print("\n==== 生成结果（使用KV cache） ====")
    for i in range(args.num_return_sequences):
        decoded = enc.decode(out_cache[i].tolist())
        print(f"生成 {i + 1}: {decoded}")


if __name__ == "__main__":
    main()
