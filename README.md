# GPT-2 模块化训练框架（完整使用指南）

本仓库是一个面向 GPT 类语言模型实验的训练框架，支持：

- 可插拔组件：Attention（`base/mqa/gqa/mla`）、位置编码（`learned/alibi/rope/sine`）、MLP（`mlp/relu/silu/swiglu/geglu/moe`）、Norm（`layernorm/rmsnorm`）
- YAML 继承式实验配置（`configs/base` + `configs/experiments`）
- 单机单卡、`torchrun` 多卡 DDP、断点恢复、权重初始化训练
- 内置验证 loss、HellaSwag 评估、样本生成、KV cache 与多长度评估工具

---

## 1. 项目结构

```text
.
├─ train.py                      # 统一入口：train / infer
├─ train_nightly.sh              # 夜间多卡训练脚本（自动续训）
├─ core/                         # CLI/配置/运行时/数据加载/训练循环/checkpoint
├─ models/gpt.py                 # GPT 主模型（组件可插拔）
├─ modules/                      # attention/mlp/norm/position 实现
├─ configs/
│  ├─ base/                      # 公共配置
│  └─ experiments/               # 实验配置
├─ benchmarks/hellaswag/         # HellaSwag 数据与评估逻辑
├─ scripts/                      # 预处理、评估、迁移、绘图脚本
└─ hellaswag/                    # HellaSwag 缓存目录（jsonl）
```

---

## 2. 环境安装

```bash
python -m pip install -r requirements.txt
```

核心依赖：`torch`, `tiktoken`, `datasets`, `transformers`, `PyYAML`。

---

## 3. 快速开始（从原始语料到训练/推理）

### 3.1 预处理原始语料为 `.npy` shards

```bash
python scripts/preprocess_data.py \
  --input /path/to/raw_data \
  --format auto \
  --output_dir /path/to/shards \
  --prefix dataset \
  --text_field text \
  --val_ratio 0.01 \
  --shard_tokens 50000000 \
  --add_eot
```

要点：
- 训练加载器通过文件名包含 `train` / `val` 识别切分（例如 `dataset_train_000000.npy`）。
- 每个 shard 必须至少满足：`B * T * world_size + 1` 个 token（训练启动会 fail-fast 检查）。

### 3.2 单进程训练

```bash
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log
```

### 3.3 恢复训练 / 仅加载权重

```bash
# 恢复 step + optimizer + loader state
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log \
  --resume log_train/baseline/log/model_15000.pt

# 仅加载模型权重（step 从 0 开始）
python train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log \
  --init_from log_train/baseline/log/model_15000.pt
```

`--resume` 与 `--init_from` 互斥。

### 3.4 推理

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

若 checkpoint 中没有 `config_ref/experiment_name`，需显式加 `--config baseline`。

---

## 4. `train.py` 完整参数

### 4.1 `train` 子命令

- `--config`：实验配置短名或路径（必填）
- `--data_root`：离线 token shard 目录（必填）
- `--log_subdir`：日志子目录（必填）
- `--resume`：从 checkpoint 恢复训练
- `--init_from`：仅加载权重初始化

### 4.2 `infer` 子命令

- `--checkpoint`（必填）
- `--config`（可选，无法从 checkpoint 推断时必填）
- `--prompt`（默认 `"Hello, I'm a language model,"`）
- `--max_length`（默认 `32`）
- `--num_return_sequences`（默认 `5`）
- `--top_k`（默认 `50`）
- `--temperature`（默认 `1.0`，必须 > 0）
- `--seed`（默认 `42`）

---

## 5. 配置系统（YAML 继承）

### 5.1 解析规则

- 支持短名：`baseline`
- 支持路径：`configs/experiments/baseline.yaml`（相对/绝对均可）
- 支持 `base:` 单个或列表继承，采用深度合并

### 5.2 实验配置必填字段

- `experiment_name`
- `model`
- `components`
- `train`

`train` 必填键：
`max_lr`, `min_lr`, `warmup_steps`, `max_steps`, `weight_decay`, `total_batch_size`, `micro_batch_size`, `sequence_length`

### 5.3 组件注册键

- `components.attention`: `base`, `mqa`, `gqa`, `mla`
- `components.position_encoding`: `learned`, `alibi`, `rope`, `sine`, `sinusoidal`
- `components.mlp`: `default`, `mlp`, `relu`, `silu`, `swiglu`, `geglu`, `moe`
- `components.norm`: `default`, `layernorm`, `rmsnorm`

### 5.4 常见模型扩展字段（写在 `model:`）

- GQA: `n_kv_head`
- MLA: `kv_lora_rank`, `q_lora_rank`
- MoE: `n_experts`, `moe_top_k`, `moe_capacity_factor`, `moe_router_noise`, `moe_expert_type`
- 初始化：`init_method`（`default/xavier/kaiming`）、`init_distribution`（`normal/uniform`）

MoE 训练可额外设置 `train.moe_aux_weight`。

---

## 6. 训练行为说明

- 梯度累积步数：
  `grad_accum_steps = total_batch_size / (micro_batch_size * sequence_length * world_size)`
- 学习率：warmup + cosine decay（见 `core/training_utils.py`）
- 每 `250` step（及最后一步）执行：
  - 验证集 loss（20 个 batch）
  - HellaSwag 评估（训练时本地实现）
  - 保存 checkpoint（step>0）
- 每 `250` step（及最后一步，且 `torch.compile=False`）采样输出文本

输出目录：

```text
log_train/<experiment_name>/<log_subdir>/
├─ log.txt
└─ model_XXXXX.pt
```

`log.txt` 行格式：`<step> train|val|hella <value>`。

---

## 7. 多卡训练

### 7.1 手动 `torchrun`

```bash
torchrun --standalone --nproc_per_node=8 train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log
```

说明：DDP backend 固定为 `nccl`，需要 CUDA。

### 7.2 夜间脚本 `train_nightly.sh`

```bash
CONFIG_PATH=configs/experiments/mla.yaml \
LOG_SUBDIR=log \
DATASET_ROOT=/path/to/shards \
bash train_nightly.sh
```

行为：
- 自动检测 GPU 数
- 自动查找 `log_train/<experiment>/<log_subdir>/model_*.pt` 最新步数并续训
- 设置 `INIT_FROM` 时仅加载权重，不自动 `--resume`

---

## 8. 评估与工具脚本

### 8.1 多长度验证评估

```bash
python scripts/eval_lengths.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --lengths 512,1024,2048 \
  --data_root /path/to/shards \
  --val_steps 20
```

### 8.2 KV cache 速度/一致性评估

```bash
python scripts/kv_cache_eval.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --max_length 64 \
  --num_return_sequences 1 \
  --dtype bfloat16
```

`max_length > block_size` 时需 `--allow_long`，且仅 `RoPE/ALiBi/正弦`支持长序列外推，`Learned` 不支持。

### 8.3 HellaSwag（HF 模型基线）

```bash
python scripts/eval_hellaswag.py --model_type gpt2 --device cuda
```

### 8.4 迁移旧 checkpoint 到 schema v2

```bash
python scripts/migrate_checkpoints.py --input /path/to/ckpt_or_dir --backup
```

常用参数：`--dry_run`, `--output_root`, `--pattern`, `--force`。

### 8.5 从训练 stdout 绘图

```bash
python scripts/plot_training_log.py nohup_log.txt training_curves.png 1000
```

注意：此脚本解析的是训练 stdout（`step ... | loss ...`），不是 `log.txt` 三列文件。

---

## 9. 脚本参数速查（完整）

### 9.1 `scripts/preprocess_data.py`

- `--input`（必填）：输入文件或目录
- `--format`：`auto/jsonl/parquet`
- `--output_dir`（必填）
- `--prefix`：输出 shard 前缀（默认 `dataset`）
- `--text_field`：文本字段名（默认 `text`）
- `--tokenizer`：tiktoken 编码名（默认 `gpt2`）
- `--val_ratio`：随机切分验证集比例（仅无 `--split_field` 时生效）
- `--split_field`：已有划分字段名（如 `partition`）
- `--train_split/--val_split/--test_split`：字段值映射
- `--shard_tokens`：每 shard token 数（默认 `50000000`）
- `--min_chars/--max_chars`：样本长度过滤
- `--min_tail_tokens`：尾 shard 最小 token，低于阈值丢弃（默认 `1024`）
- `--add_eot`：每条样本末尾追加 EOT
- `--streaming`：使用流式读取
- `--seed`：随机种子

### 9.2 `scripts/eval_lengths.py`

- `--config`（必填）
- `--checkpoint`（必填）
- `--lengths`（必填，逗号分隔，如 `512,1024,2048`）
- `--data_root`（必填）
- `--val_steps`（默认 `20`）
- `--batch_size`（默认使用训练配置的 `micro_batch_size`）
- `--device`：`cuda/mps/cpu`（默认自动）

### 9.3 `scripts/kv_cache_eval.py`

- `--config`（必填）
- `--checkpoint`（可选；不填时需提供 `--log_subdir` 或 `--log_dir` 自动找最新）
- `--log_subdir` / `--log_dir`
- `--prompt`
- `--max_length`
- `--top_k`
- `--temperature`
- `--num_return_sequences`
- `--seed`
- `--dtype`：`float32/bfloat16`
- `--allow_long`：允许超过训练长度（仅 RoPE/ALiBi/正弦）

### 9.4 `scripts/migrate_checkpoints.py`

- `--input`（必填）：单文件或目录
- `--pattern`：目录扫描匹配模式（默认 `model_*.pt`）
- `--no_recursive`：关闭递归扫描
- `--output_root`：输出到新目录（不填则就地覆盖）
- `--dry_run`：仅预览不写文件
- `--force`：即使已是 v2 也重写
- `--backup`：就地覆盖前备份
- `--backup_suffix`：备份后缀（默认 `.v1.bak`）
- `--overwrite_backup`：允许覆盖已有备份

### 9.5 `scripts/eval_hellaswag.py`

- `-m/--model_type`：HF 模型名（默认 `gpt2`）
- `-d/--device`：评估设备（默认 `cuda`）

---

## 10. Checkpoint 规范（v2）

主流程仅支持 `schema_version=2`。核心字段：

- `schema_version`
- `model`
- `config_dict`
- `step`
- 可选：`optimizer`, `train_loader_state`, `experiment_name`, `config_ref`, `val_loss`

恢复训练时会尝试恢复优化器与数据加载进度；如果 shard 与 checkpoint 不匹配会报错并中止。

---

## 11. 常见问题

- **找不到 shard**：确认 `--data_root` 下存在文件名包含 `train` / `val` 的 `.npy`。
- **`total_batch_size` 断言失败**：调整使其可被 `micro_batch_size * sequence_length * world_size` 整除。
- **infer 报配置缺失**：补 `--config <experiment>`。
- **checkpoint 版本不支持**：先运行 `scripts/migrate_checkpoints.py` 迁移到 v2。
- **DDP 启动失败**：确认 CUDA/NCCL 可用，并使用 `torchrun` 启动。
