"""
训练器
封装训练主循环、评估、采样与checkpoint保存
"""

import os
import time

import torch
import torch.distributed as dist

from core.data_loader import load_tokens
from core.evaluator import evaluate_hellaswag, evaluate_val_loss, generate_samples
from core.training_utils import get_lr


class Trainer:
    def __init__(
        self,
        model,
        raw_model,
        optimizer,
        train_loader,
        val_loader,
        runtime,
        train_config,
        log_dir,
        log_file,
        enc,
    ):
        self.model = model
        self.raw_model = raw_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.runtime = runtime
        self.train_config = train_config
        self.log_dir = log_dir
        self.log_file = log_file
        self.enc = enc

        self.max_lr = train_config["max_lr"]
        self.min_lr = train_config["min_lr"]
        self.warmup_steps = train_config["warmup_steps"]
        self.max_steps = train_config["max_steps"]
        self.eval_interval = int(train_config.get("eval_interval", 250))
        self.val_loss_steps = int(train_config.get("val_loss_steps", 20))
        self.moe_aux_weight = float(train_config.get("moe_aux_weight", 0.0))
        self.use_compile = bool(train_config.get("use_compile", False))

    def _truncate_log_file_for_resume(self, start_step):
        if not os.path.exists(self.log_file):
            return
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(self.log_file, "w", encoding="utf-8") as f:
            for line in lines:
                try:
                    step_in_line = int(line.split()[0])
                    if step_in_line < start_step:
                        f.write(line)
                except (ValueError, IndexError):
                    f.write(line)

    def restore_train_loader_state(self, resume_loader_state, start_step):
        if resume_loader_state is None:
            return
        self.train_loader.current_shard = resume_loader_state["current_shard"]
        self.train_loader.current_position = resume_loader_state["current_position"]
        self.train_loader.tokens = load_tokens(self.train_loader.shards[self.train_loader.current_shard])
        self.train_loader._ensure_current_shard_has_batch()
        if self.runtime.master_process:
            print(
                f"✓ 恢复数据加载器状态: shard {self.train_loader.current_shard}, "
                f"position {self.train_loader.current_position}"
            )
            self._truncate_log_file_for_resume(start_step)
            print(f"✓ 日志文件已截断至第 {start_step} 步之前")

    def _save_checkpoint(self, step, val_loss):
        checkpoint_path = os.path.join(self.log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "config": self.raw_model.config,
            "step": step,
            "val_loss": val_loss,
            "optimizer": self.optimizer.state_dict(),
            "train_loader_state": {
                "current_shard": self.train_loader.current_shard,
                "current_position": self.train_loader.current_position,
            },
        }
        torch.save(checkpoint, checkpoint_path)

    def fit(self, start_step, grad_accum_steps):
        for step in range(start_step, self.max_steps):
            t0 = time.time()
            last_step = step == self.max_steps - 1

            if step % self.eval_interval == 0 or last_step:
                val_loss_accum = evaluate_val_loss(
                    model=self.model,
                    val_loader=self.val_loader,
                    device=self.runtime.device,
                    device_type=self.runtime.device_type,
                    val_loss_steps=self.val_loss_steps,
                )
                if self.runtime.ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if self.runtime.master_process:
                    val_loss = val_loss_accum.item()
                    print(f"validation loss: {val_loss:.4f}")
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"{step} val {val_loss:.4f}\n")
                    if step > 0 and (step % self.eval_interval == 0 or last_step):
                        self._save_checkpoint(step, val_loss)

            if (step % self.eval_interval == 0 or last_step) and (not self.use_compile):
                num_correct_norm, num_total = evaluate_hellaswag(
                    model=self.model,
                    device=self.runtime.device,
                    device_type=self.runtime.device_type,
                    ddp_rank=self.runtime.ddp_rank,
                    ddp_world_size=self.runtime.ddp_world_size,
                )
                if self.runtime.ddp:
                    num_total_tensor = torch.tensor(num_total, dtype=torch.long, device=self.runtime.device)
                    num_correct_tensor = torch.tensor(
                        num_correct_norm, dtype=torch.long, device=self.runtime.device
                    )
                    dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_tensor, op=dist.ReduceOp.SUM)
                    num_total = num_total_tensor.item()
                    num_correct_norm = num_correct_tensor.item()

                acc_norm = num_correct_norm / num_total
                if self.runtime.master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"{step} hella {acc_norm:.4f}\n")

            if ((step > 0 and step % self.eval_interval == 0) or last_step) and (not self.use_compile):
                samples = generate_samples(
                    model=self.model,
                    enc=self.enc,
                    device=self.runtime.device,
                    device_type=self.runtime.device_type,
                    prompt="Hello, I'm a language model,",
                    num_return_sequences=4,
                    max_length=32,
                    seed=42 + self.runtime.ddp_rank,
                )
                for i, decoded in enumerate(samples):
                    print(f"rank {self.runtime.ddp_rank} sample {i}: {decoded}")

            self.model.train()
            self.optimizer.zero_grad()
            loss_accum = torch.zeros((), device=self.runtime.device)
            for micro_step in range(grad_accum_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.runtime.device), y.to(self.runtime.device)
                if self.runtime.ddp:
                    self.model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                with torch.autocast(device_type=self.runtime.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, y)
                if self.moe_aux_weight > 0:
                    aux_loss = (
                        self.raw_model.get_moe_aux_loss() if hasattr(self.raw_model, "get_moe_aux_loss") else None
                    )
                    if aux_loss is not None:
                        loss = loss + self.moe_aux_weight * aux_loss
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

            if self.runtime.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            lr = get_lr(step, self.warmup_steps, self.max_steps, self.max_lr, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()

            if self.runtime.device_type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = (
                self.train_loader.B * self.train_loader.T * grad_accum_steps * self.runtime.ddp_world_size
            )
            tokens_per_sec = tokens_processed / dt
            if self.runtime.master_process:
                print(
                    f"step {step:5d} | loss: {loss_accum.item():.6f} | "
                    f"lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | "
                    f"tok/sec: {tokens_per_sec:.2f}"
                )
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")
