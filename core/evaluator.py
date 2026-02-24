"""
评估与采样工具
"""

import torch
import torch.nn.functional as F

from hellaswag import render_example, iterate_examples
from core.training_utils import get_most_likely_row


def evaluate_val_loss(model, val_loader, device, device_type, val_loss_steps=20):
    """
    计算验证集loss（返回张量，便于DDP all_reduce）。
    """
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = torch.zeros((), device=device)
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_loss_accum += loss.detach() / val_loss_steps
    return val_loss_accum


def evaluate_hellaswag(model, device, device_type, ddp_rank=0, ddp_world_size=1):
    """
    评估 HellaSwag，返回 (num_correct_norm, num_total)。
    """
    num_correct_norm = 0
    num_total = 0
    model.eval()
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


def generate_samples(
    model,
    enc,
    device,
    device_type,
    prompt,
    num_return_sequences=4,
    max_length=32,
    seed=42,
):
    """
    top-k 采样文本，返回解码后的字符串列表。
    """
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    xgen = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)

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
        decoded = enc.decode(xgen[i, :max_length].tolist())
        outputs.append(decoded)
    return outputs
