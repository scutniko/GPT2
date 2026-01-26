"""
KV cache 推理评估脚本
对比不使用KV cache与使用KV cache的生成速度与输出一致性
"""

import os
import glob
import time
import argparse
import importlib

import torch
import torch.nn.functional as F
import tiktoken

from models.gpt import GPT


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
def generate_no_cache(model, input_ids, max_length, top_k, seed, device, use_autocast):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = input_ids
    _sync_if_cuda(device)
    t0 = time.time()
    while x.size(1) < max_length:
        if use_autocast:
            with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
                logits, _ = model(x)
        else:
            logits, _ = model(x)
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
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(args.num_return_sequences, 1)

    use_autocast = args.dtype == "bfloat16" and device != "cpu"
    if use_autocast:
        print("推理精度: bfloat16")
    else:
        print("推理精度: float32")

    out_no_cache, time_no_cache = generate_no_cache(
        model, tokens, args.max_length, args.top_k, args.seed, device, use_autocast
    )
    out_cache, time_cache = generate_with_cache(
        model, tokens, args.max_length, args.top_k, args.seed, device, use_autocast
    )

    prompt_len = tokens.size(1)
    gen_tokens = (args.max_length - prompt_len) * args.num_return_sequences
    tok_per_sec_no_cache = gen_tokens / max(time_no_cache, 1e-9)
    tok_per_sec_cache = gen_tokens / max(time_cache, 1e-9)

    print("\n==== 速度对比 ====")
    print(f"不使用KV cache: {time_no_cache:.4f}s, {tok_per_sec_no_cache:.2f} tok/s")
    print(f"使用KV cache:  {time_cache:.4f}s, {tok_per_sec_cache:.2f} tok/s")
    if time_cache > 0:
        print(f"速度提升: {time_no_cache / time_cache:.2f}x")

    same = torch.equal(out_no_cache, out_cache)
    print("\n==== 一致性检查 ====")
    print(f"逐token输出一致: {'是' if same else '否'}")
    if not same:
        mismatch = (out_no_cache != out_cache).nonzero(as_tuple=False)
        first = mismatch[0].tolist()
        print(f"首个不一致位置: batch={first[0]}, pos={first[1]}")

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
