"""
评估与采样逻辑
"""

import torch
import torch.nn.functional as F

from core.training_utils import get_most_likely_row
from hellaswag import iterate_examples, render_example


def run_inference(
    model,
    device,
    device_type,
    enc,
    prompt="Hello, I'm a language model,",
    max_length=32,
    num_return_sequences=5,
    top_k=50,
    temperature=1.0,
    seed=42,
    master_process=True,
):
    """执行文本生成推理（模型权重需已加载）。"""
    model.to(device)
    model.eval()

    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

    if master_process:
        print(f"\n{'='*60}")
        print(f"提示词: {prompt}")
        print(f"{'='*60}\n")

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            safe_top_k = min(top_k, probs.size(-1))
            topk_probs, topk_indices = torch.topk(probs, safe_top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    if master_process:
        for i in range(num_return_sequences):
            out_tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(out_tokens)
            print(f"生成 {i+1}: {decoded}\n")


def evaluate_validation_loss(model, val_loader, device, device_type, val_loss_steps=20):
    """计算验证集loss（单进程本地结果）。"""
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    return val_loss_accum


def evaluate_hellaswag_local(model, device, device_type, ddp_rank, ddp_world_size):
    """计算本进程负责切片上的 HellaSwag 正确数与总数。"""
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % ddp_world_size != ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    return num_correct_norm, num_total


def generate_samples_for_rank(
    model,
    enc,
    device,
    device_type,
    ddp_rank,
    num_return_sequences=4,
    max_length=32,
    prompt="Hello, I'm a language model,",
):
    """按 rank 生成若干样本文本。"""
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    outputs = []
    for i in range(num_return_sequences):
        out_tokens = xgen[i, :max_length].tolist()
        outputs.append((i, enc.decode(out_tokens)))
    return outputs
