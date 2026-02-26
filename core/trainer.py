"""
训练循环
"""

import os
import time

import torch
import torch.distributed as dist

from core.checkpoint import save_checkpoint
from core.evaluator import (
    evaluate_hellaswag_local,
    evaluate_validation_loss,
    generate_samples_for_rank,
)
from core.runtime import get_autocast_context
from core.training_utils import get_lr


def run_train_loop(
    model,
    raw_model,
    optimizer,
    train_loader,
    val_loader,
    runtime_ctx,
    train_config,
    enc,
    log_file,
    log_dir,
    start_step,
    grad_accum_steps,
    experiment_name=None,
    config_ref=None,
    use_compile=False,
):
    """执行训练主循环。"""
    max_lr = train_config["max_lr"]
    min_lr = train_config["min_lr"]
    warmup_steps = train_config["warmup_steps"]
    max_steps = train_config["max_steps"]
    moe_aux_weight = float(train_config.get("moe_aux_weight", 0.0))

    ddp = runtime_ctx.ddp
    ddp_rank = runtime_ctx.ddp_rank
    ddp_world_size = runtime_ctx.ddp_world_size
    device = runtime_ctx.device
    device_type = runtime_ctx.device_type
    master_process = runtime_ctx.master_process

    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = step == (max_steps - 1)
        should_eval = step % 250 == 0 or last_step
        should_sample = ((step > 0 and step % 250 == 0) or last_step) and (not use_compile)

        model.train()
        optimizer.zero_grad()
        total_loss_accum = torch.zeros((), device=device)
        ce_loss_accum = torch.zeros((), device=device)
        aux_loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            with get_autocast_context(device_type):
                _, ce_loss = model(x, y)
            aux_loss_term = torch.zeros_like(ce_loss)
            if moe_aux_weight > 0:
                aux_loss = raw_model.get_moe_aux_loss() if hasattr(raw_model, "get_moe_aux_loss") else None
                if aux_loss is not None:
                    aux_loss_term = moe_aux_weight * aux_loss
            total_loss = ce_loss + aux_loss_term
            total_loss = total_loss / grad_accum_steps
            total_loss_accum += total_loss.detach()
            ce_loss_accum += (ce_loss / grad_accum_steps).detach()
            aux_loss_accum += (aux_loss_term / grad_accum_steps).detach()
            total_loss.backward()
        if ddp:
            dist.all_reduce(total_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(ce_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(aux_loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {total_loss_accum.item():.6f} | "
                f"ce: {ce_loss_accum.item():.6f} | aux: {aux_loss_accum.item():.6f} | "
                f"lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | "
                f"tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{step} train {total_loss_accum.item():.6f} "
                    f"ce={ce_loss_accum.item():.6f} aux={aux_loss_accum.item():.6f}\n"
                )

        if should_eval:
            model.eval()

            val_loss_accum = evaluate_validation_loss(
                model=model,
                val_loader=val_loader,
                device=device,
                device_type=device_type,
                val_loss_steps=20,
            )
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0:
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    save_checkpoint(
                        path=checkpoint_path,
                        raw_model=raw_model,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        step=step,
                        val_loss=val_loss_accum.item(),
                        experiment_name=experiment_name,
                        config_ref=config_ref,
                    )

            if not use_compile:
                num_correct_norm, num_total = evaluate_hellaswag_local(
                    model=model,
                    device=device,
                    device_type=device_type,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"{step} hella {acc_norm:.4f}\n")

        if should_sample:
            samples = generate_samples_for_rank(
                model=model,
                enc=enc,
                device=device,
                device_type=device_type,
                ddp_rank=ddp_rank,
                num_return_sequences=4,
                max_length=32,
                prompt="Hello, I'm a language model,",
            )
            for i, decoded in samples:
                print(f"rank {ddp_rank} sample {i}: {decoded}")
