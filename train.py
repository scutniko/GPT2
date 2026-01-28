"""
统一的训练入口
支持所有消融实验的训练、恢复和推理
"""

import os
import sys
import time
import argparse
import importlib
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import tiktoken
from hellaswag import render_example, iterate_examples
from core.data_loader import DataLoaderLite, load_tokens
from core.training_utils import get_lr, get_most_likely_row
from models.gpt import GPT

from core.config import GPTConfig

def main():
    """主训练函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GPT-2 Training with Ablation Studies')
    parser.add_argument('--experiment', type=str, required=True,
                        help='实验名称 (baseline, alibi, rope, sine, mqa, gqa, mla)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练 (e.g., log/model_15000.pt)')
    parser.add_argument('--inference', type=str, default=None,
                        help='推理模式：加载检查点生成文本 (e.g., log/model_15000.pt)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='离线token shard目录（包含train/val切分的.npy文件）')
    args = parser.parse_args()

    # 动态加载实验配置
    try:
        exp_module = importlib.import_module(f'experiments.{args.experiment}')
    except ImportError:
        print(f"错误: 找不到实验配置 '{args.experiment}'")
        print(f"可用的实验:")
        print(f"  位置编码: baseline, alibi, rope, sine")
        print(f"  注意力机制: mqa, gqa, mla")
        print(f"  激活函数: relu, silu, swiglu, geglu")
        print(f"  MoE: moe")
        print(f"  归一化层: rmsnorm")
        sys.exit(1)
    
    # 获取配置
    experiment_name = exp_module.EXPERIMENT_NAME
    model_config = exp_module.MODEL_CONFIG
    attention_class = exp_module.ATTENTION_CLASS
    position_encoding_class = exp_module.POSITION_ENCODING_CLASS
    train_config = exp_module.TRAINING_CONFIG
    
    # 获取可选的MLP和归一化层配置（如果实验配置中没有定义，则为None）
    mlp_class = getattr(exp_module, 'MLP_CLASS', None)
    norm_class = getattr(exp_module, 'NORM_CLASS', None)
    data_root = args.data_root
    if data_root is None:
        data_root = train_config.get("data_root")
    
    print(f"=" * 60)
    print(f"实验: {experiment_name}")
    print(f"注意力机制: {attention_class.__name__}")
    print(f"位置编码: {position_encoding_class.__name__}")
    if mlp_class is not None:
        print(f"MLP类型: {mlp_class.__name__}")
    if norm_class is not None:
        print(f"归一化层: {norm_class.__name__}")
    print(f"=" * 60)
    
    # 设置DDP (分布式数据并行)
    ddp = int(os.environ.get('RANK', -1)) != -1  # 是否为DDP运行
    if ddp:
        # DDP需要CUDA
        assert torch.cuda.is_available(), "DDP需要CUDA支持"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # 主进程负责日志和保存
    else:
        # 单卡训练
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # 自动检测设备
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"使用设备: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # 设置随机种子
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # 训练配置
    total_batch_size = train_config["total_batch_size"]
    B = train_config["micro_batch_size"]
    T = train_config["sequence_length"]
    assert total_batch_size % (B * T * ddp_world_size) == 0, \
        "total_batch_size必须能被B * T * ddp_world_size整除"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"总batch size: {total_batch_size}")
        print(f"=> 梯度累积步数: {grad_accum_steps}")

    # 创建数据加载器
    train_loader = DataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size, 
        split="train", 
        master_process=master_process,
        data_root=data_root
    )
    val_loader = DataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size, 
        split="val", 
        master_process=master_process,
        data_root=data_root
    )

    torch.set_float32_matmul_precision('high')

    # 创建模型
    model = GPT(model_config, attention_class, position_encoding_class,
                mlp_class=mlp_class, norm_class=norm_class)
    
    # 检查点恢复（在DDP包装之前）
    start_step = 0
    resume_checkpoint = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt['step'] + 1
        resume_checkpoint = ckpt
        if master_process:
            print(f"✓ 从 {args.resume} 恢复训练 (第 {start_step} 步开始)")

    # 推理模式
    if args.inference:
        if master_process:
            print(f"✓ 推理模式：加载模型权重 {args.inference}")
        ckpt = torch.load(args.inference, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        
        # 生成文本5次
        num_return_sequences = 5
        max_length = 32
        prompt = "Hello, I'm a language model,"
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        
        if master_process:
            print(f"\n{'='*60}")
            print(f"提示词: {prompt}")
            print(f"{'='*60}\n")
        
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        
        if master_process:
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"生成 {i+1}: {decoded}\n")
        
        sys.exit(0)
    
    model.to(device)
    use_compile = False  # torch.compile与HellaSwag评估和生成有冲突
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    raw_model = model.module if ddp else model  # 获取原始模型

    # 训练超参数
    max_lr = train_config["max_lr"]
    min_lr = train_config["min_lr"]
    warmup_steps = train_config["warmup_steps"]
    max_steps = train_config["max_steps"]
    weight_decay = train_config["weight_decay"]
    moe_aux_weight = float(train_config.get("moe_aux_weight", 0.0))

    # 创建优化器
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        device_type=device_type,
        master_process=master_process
    )

    # 恢复优化器状态
    if resume_checkpoint is not None and 'optimizer' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        if master_process:
            print(f"✓ 恢复优化器状态")

    # 创建日志目录
    log_dir = train_config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    
    # 只在非恢复模式下清空日志文件
    if not args.resume:
        with open(log_file, "w") as f:
            pass

    # 恢复数据加载器状态
    if resume_checkpoint is not None and 'train_loader_state' in resume_checkpoint:
        train_loader.current_shard = resume_checkpoint['train_loader_state']['current_shard']
        train_loader.current_position = resume_checkpoint['train_loader_state']['current_position']
        train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
        if master_process:
            print(f"✓ 恢复数据加载器状态: shard {train_loader.current_shard}, position {train_loader.current_position}")
            
            # 截断日志文件
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                with open(log_file, "w") as f:
                    for line in lines:
                        try:
                            step_in_line = int(line.split()[0])
                            if step_in_line < start_step:
                                f.write(line)
                        except (ValueError, IndexError):
                            f.write(line)
                print(f"✓ 日志文件已截断至第 {start_step} 步之前")

    # 训练循环
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 验证评估
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 250 == 0 or last_step):
                    # 保存检查点
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict(),
                        'train_loader_state': {
                            'current_shard': train_loader.current_shard,
                            'current_position': train_loader.current_position,
                        }
                    }
                    torch.save(checkpoint, checkpoint_path)

        # HellaSwag评估
        if (step % 250 == 0 or last_step) and (not use_compile):
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
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
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
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")
        
        # 文本生成
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xcol = torch.gather(topk_indices, -1, ix)
                    xgen = torch.cat((xgen, xcol), dim=1)
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        # 训练步骤
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            if moe_aux_weight > 0:
                aux_loss = raw_model.get_moe_aux_loss() if hasattr(raw_model, "get_moe_aux_loss") else None
                if aux_loss is not None:
                    loss = loss + moe_aux_weight * aux_loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 学习率调度
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

