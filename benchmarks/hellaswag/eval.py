"""
HellaSwag 评估逻辑。
"""

import torch
import torch.nn.functional as F

from benchmarks.hellaswag.data import iterate_examples, render_example


@torch.no_grad()
def evaluate_hf_model(model_type, device):
    """
    用 HuggingFace GPT2LMHeadModel 在 HellaSwag 上评估。
    """
    from transformers import GPT2LMHeadModel

    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")

        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    return {
        "num_total": num_total,
        "num_correct": num_correct,
        "num_correct_norm": num_correct_norm,
        "acc": num_correct / num_total if num_total > 0 else 0.0,
        "acc_norm": num_correct_norm / num_total if num_total > 0 else 0.0,
    }

