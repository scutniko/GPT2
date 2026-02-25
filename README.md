# GPT2 训练框架（模块化实验版）

本仓库是一个面向 GPT 类语言模型实验的训练框架，支持：

- 多种注意力机制：`base / mqa / gqa / mla`
- 多种位置编码：`learned / alibi / rope / sine`
- 多种 MLP：`mlp / relu / silu / swiglu / geglu / moe`
- 多种归一化：`layernorm / rmsnorm`
- YAML 继承式实验配置（`base + experiments`）
- 单卡训练、DDP 多卡训练、断点恢复、推理与评估

## 1. 目录结构

```text
.
├─ train.py                      # 统一入口（train/infer）
├─ train_nightly.sh              # 夜间多卡训练脚本（自动续训）
├─ core/                         # 运行时、配置、数据、训练循环、checkpoint
├─ models/gpt.py                 # GPT 主模型（可插拔组件）
├─ modules/                      # attention/mlp/norm/position 实现
├─ configs/
│  ├─ base/                      # 基础配置
│  └─ experiments/               # 实验配置
├─ scripts/
│  ├─ preprocess_data.py         # 语料预处理为 .npy shard
│  ├─ eval_lengths.py            # 多长度验证评估
│  ├─ kv_cache_eval.py           # KV cache 速度/一致性评估
│  └─ migrate_checkpoints.py     # 老 checkpoint 迁移到 v2
├─ hellaswag.py                  # HellaSwag 下载与评估工具
└─ plot_training_log.py          # 解析训练 stdout 日志并绘图
```

## 2. 环境安装

```bash
python -m pip install -r requirements.txt
```

依赖见 `requirements.txt`，核心包含：

- `torch>=2.0.0`
- `tiktoken`
- `datasets`
- `transformers`
- `PyYAML`

## 3. 快速开始

### 3.1 准备离线 token 数据（.npy shard）

训练数据加载器要求 `--data_root` 目录下存在文件名包含 `train` 和 `val` 的 shard 文件（例如 `dataset_train_000000.npy`、`dataset_val_000000.npy`）。

推荐先用预处理脚本生成：

```bash
python scripts/preprocess_data.py \
  --input /path/to/raw_corpus \
  --format auto \
  --output_dir /path/to/shards \
  --prefix dataset \
  --text_field text \
  --val_ratio 0.01 \
  --shard_tokens 50000000 \
  --add_eot
```

### 3.2 单进程训练

```bash
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log
```

### 3.3 从 checkpoint 恢复训练

```bash
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log \
  --resume log_train/baseline/log/model_15000.pt
```

### 3.4 仅加载权重初始化训练（不恢复 step/优化器）

```bash
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log \
  --init_from log_train/baseline/log/model_15000.pt
```

注意：`--resume` 与 `--init_from` 互斥。

### 3.5 推理

最简推理：

```bash
python train.py infer \
  --checkpoint log_train/baseline/log/model_15000.pt
```

带采样参数：

```bash
python train.py infer \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --prompt "你好，我是一个语言模型，" \
  --max_length 64 \
  --num_return_sequences 3 \
  --top_k 40 \
  --temperature 0.9 \
  --seed 42
```

当 checkpoint 中没有 `config_ref/experiment_name` 时，需手动指定：

```bash
python train.py infer \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --config baseline
```

## 4. 多卡与夜间训练

### 4.1 手动 torchrun

```bash
torchrun --standalone --nproc_per_node=8 train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log
```

### 4.2 `train_nightly.sh`（自动探测 GPU + 自动续训）

```bash
CONFIG_PATH=configs/experiments/mla.yaml \
LOG_SUBDIR=log \
DATASET_ROOT=/path/to/shards \
bash train_nightly.sh
```

脚本行为：

- 自动检测 GPU 数（`nvidia-smi -L`）
- 自动在 `log_train/<EXPERIMENT_NAME>/<LOG_SUBDIR>/` 下查找最新 `model_*.pt`
- 找到则自动追加 `--resume`，否则从头训练

环境变量说明：

- `CONFIG_PATH`：实验配置路径（默认 `configs/experiments/mla.yaml`）
- `LOG_SUBDIR`：日志子目录（必填）
- `DATASET_ROOT`：token shard 根目录（必填）
- `EXPERIMENT_NAME`：日志目录实验名（可选，默认取配置文件名）
- `INIT_FROM`：只加载权重（设置后不会自动 `--resume`）

## 5. 配置系统（YAML）

### 5.1 配置解析规则

- 支持短名：`baseline`
- 支持相对路径/绝对路径：`configs/experiments/baseline.yaml`
- 支持 `base` 继承（单个字符串或列表）
- 深度合并：子配置覆盖父配置同名字段

### 5.2 实验配置必填字段

- `experiment_name`
- `model`
- `components`
- `train`

`train` 至少包含：

- `max_lr`
- `min_lr`
- `warmup_steps`
- `max_steps`
- `weight_decay`
- `total_batch_size`
- `micro_batch_size`
- `sequence_length`

### 5.3 组件注册键

| 类型 | 可选值 |
|---|---|
| `components.attention` | `base`, `mqa`, `gqa`, `mla` |
| `components.position_encoding` | `learned`, `alibi`, `rope`, `sine`, `sinusoidal` |
| `components.mlp` | `default`, `mlp`, `relu`, `silu`, `swiglu`, `geglu`, `moe` |
| `components.norm` | `default`, `layernorm`, `rmsnorm` |

### 5.4 当前内置实验

`baseline`, `alibi`, `rope`, `sine`, `mqa`, `gqa`, `mla`, `relu`, `silu`, `swiglu`, `geglu`, `rmsnorm`, `moe`

## 6. 训练与日志行为说明

- 每个 step 打印训练信息：`loss/lr/grad_norm/dt/tok_sec`
- 每 250 步（及最后一步）做验证集 loss 评估
- 每 250 步（及最后一步）做 HellaSwag 评估与样本生成（`torch.compile=False` 时）
- checkpoint 默认保存到 `log_train/<experiment_name>/<log_subdir>/model_<step>.pt`
- 非恢复训练会清空当前 `log_file`
- 恢复训练会尝试恢复：
  - 模型参数
  - 优化器状态
  - 数据加载位置（`train_loader_state`）

`log_file`（默认 `log.txt`）写入格式为：

- `<step> train <loss>`
- `<step> val <loss>`
- `<step> hella <acc>`

## 7. Checkpoint 规范

当前主训练/推理代码只支持 `schema_version=2` 的 checkpoint。

主要字段：

- `schema_version`
- `model`
- `config_dict`
- `step`
- `optimizer`
- `train_loader_state`
- 可选：`experiment_name`, `config_ref`

旧文件可使用迁移脚本转换：

```bash
python scripts/migrate_checkpoints.py \
  --input /path/to/checkpoints \
  --pattern "model_*.pt" \
  --backup
```

## 8. 评估与辅助脚本

### 8.1 HellaSwag（独立脚本）

```bash
python hellaswag.py --model_type gpt2 --device cuda
```

### 8.2 多长度验证评估

```bash
python scripts/eval_lengths.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --lengths 512,1024 \
  --data_root /path/to/shards
```

### 8.3 KV cache 推理评估

```bash
python scripts/kv_cache_eval.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --max_length 64 \
  --num_return_sequences 1
```

### 8.4 训练日志绘图

`plot_training_log.py` 解析的是训练 stdout 日志（例如 `nohup` 输出），不是 `log.txt` 三列格式：

```bash
python plot_training_log.py nohup_log.txt training_curves.png 1000
```

## 9. 常见问题

### Q1: 报错找不到 train/val shards？

检查 `--data_root` 下是否有文件名包含 `train` / `val` 的 `.npy` 文件。

### Q2: DDP 启动失败？

DDP 仅支持 CUDA + NCCL，需使用 `torchrun` 启动并确保 GPU 可见。

### Q3: 推理报缺少配置引用？

在 `infer` 命令中增加 `--config`，例如 `--config baseline`。

### Q4: 夜间脚本没续上 checkpoint？

检查 `EXPERIMENT_NAME`、`LOG_SUBDIR` 与实际日志目录是否一致。

### Q5: checkpoint 版本不支持？

先用 `scripts/migrate_checkpoints.py` 迁移到 `schema_version=2`。

## 10. 开发建议

- 新实验建议放到 `configs/experiments/*.yaml`，并尽量复用 `configs/base/*.yaml`
- 大模型 checkpoint 请勿提交到仓库
- 修改核心训练逻辑后，建议至少做一次：
  - 短程 train（含保存/恢复）
  - infer
  - `scripts/kv_cache_eval.py` 一致性检查
