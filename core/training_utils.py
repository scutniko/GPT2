"""
训练工具函数
包含学习率调度、HellaSwag评估等
"""

import math
import torch
import torch.nn.functional as F


def get_lr(it, warmup_steps, max_steps, max_lr=6e-4, min_lr=6e-5):
    """
    学习率调度函数（带warmup的cosine decay）
    
    Args:
        it: 当前迭代步数
        warmup_steps: warmup步数
        max_steps: 最大步数
        max_lr: 最大学习率
        min_lr: 最小学习率
        
    Returns:
        当前步数的学习率
    """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def get_most_likely_row(tokens, mask, logits):
    """
    HellaSwag评估辅助函数
    计算所有位置的自回归损失，返回损失最小的completion索引
    
    Args:
        tokens: token序列 [batch_size, seq_len]
        mask: mask标记 [batch_size, seq_len]
        logits: 模型输出logits [batch_size, seq_len, vocab_size]
        
    Returns:
        pred_norm: 最可能的completion索引
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

