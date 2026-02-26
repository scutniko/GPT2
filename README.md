# GPT-2 模块化训练框架

本仓库是一个面向 GPT 类语言模型实验的训练框架，核心目标是用统一训练入口快速做结构消融与对比实验。

支持能力：
- 可插拔组件：Attention（`base/mqa/gqa/mla`）、位置编码（`learned/alibi/rope/sine`）、MLP（`mlp/relu/silu/swiglu/geglu/moe`）、Norm（`layernorm/rmsnorm`）
- YAML 继承式配置（`configs/base` + `configs/experiments`）
- 单卡训练、`torchrun` 多卡 DDP、`resume` 断点恢复、`init_from` 权重初始化
- 训练内置验证 loss + HellaSwag 评估 + 样本生成
- 多长度评估、KV cache 评估、旧 checkpoint 迁移、训练日志绘图

---

## 1. 项目结构

```text
.
├─ train.py                      # 统一入口：train / infer
├─ train_nightly.sh              # 夜间多卡脚本（自动续训）
├─ core/
│  ├─ cli.py                     # 命令行参数
│  ├─ experiment.py              # YAML 继承解析 + 组件解析
│  ├─ runtime.py                 # 设备/DDP 初始化
│  ├─ data_loader.py             # token shard 加载器
│  ├─ trainer.py                 # 训练循环
│  └─ checkpoint.py              # v2 checkpoint 读写/恢复
├─ models/gpt.py                 # GPT 主模型定义
├─ modules/                      # attention / position / mlp / norm / block
├─ configs/
│  ├─ base/                      # 基础模型与训练配置
│  └─ experiments/               # 实验配置
├─ scripts/                      # 预处理、评估、迁移、绘图
├─ benchmarks/hellaswag/         # HellaSwag 数据与评估逻辑
└─ hellaswag/                    # HellaSwag 缓存目录（jsonl）
```

---

## 2. 环境与安装

### 2.1 Python 解释器约定

本仓库约定统一使用：

`D:\Softwares\Anaconda\envs\torch\python.exe`

不要混用系统默认 `python`。

### 2.2 安装依赖

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"
& $PY -m pip install -r requirements.txt
```

`requirements.txt` 主要依赖：`torch`, `tiktoken`, `datasets`, `transformers`, `PyYAML`。

---

## 3. 快速开始

### 3.1 预处理原始语料为 `.npy` shards

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"
& $PY scripts/preprocess_data.py `
  --input D:\path\to\raw_data `
  --format auto `
  --output_dir D:\path\to\shards `
  --prefix dataset `
  --text_field text `
  --val_ratio 0.01 `
  --shard_tokens 50000000 `
  --add_eot
```

说明：
- 训练加载器按文件名是否包含 `train` / `val` 来识别切分。
- 预处理支持 `jsonl/json/parquet`，并支持 `split_field` 直接按字段划分 `train/val/test`。
- 小于 `--min_tail_tokens` 的尾 shard 会被丢弃（默认 1024）。

### 3.2 单进程训练

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"
& $PY train.py train `
  --config baseline `
  --data_root D:\path\to\shards `
  --log_subdir log
```

### 3.3 恢复训练 / 仅加载权重

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"

# 恢复 step + optimizer + data loader 状态
& $PY train.py train `
  --config baseline `
  --data_root D:\path\to\shards `
  --log_subdir log `
  --resume log_train\baseline\log\model_15000.pt

# 只加载模型权重（step 从 0 开始）
& $PY train.py train `
  --config baseline `
  --data_root D:\path\to\shards `
  --log_subdir log `
  --init_from log_train\baseline\log\model_15000.pt
```

`--resume` 与 `--init_from` 互斥（CLI 会直接报错）。

### 3.4 推理

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"
& $PY train.py infer `
  --checkpoint log_train\baseline\log\model_15000.pt `
  --prompt "你好，我是一个语言模型，" `
  --max_length 64 `
  --num_return_sequences 3 `
  --top_k 40 `
  --temperature 0.9 `
  --seed 42
```

若 checkpoint 不含 `config_ref/experiment_name`，需显式传 `--config`。

---

## 4. train.py 参数

### 4.1 `train` 子命令

- `--config`：实验配置名或路径（必填）
- `--data_root`：离线 token shard 目录（必填）
- `--log_subdir`：日志子目录（必填）
- `--resume`：恢复训练
- `--init_from`：只加载模型权重

### 4.2 `infer` 子命令

- `--checkpoint`（必填）
- `--config`（可选）
- `--prompt`（默认 `"Hello, I'm a language model,"`）
- `--max_length`（默认 `32`，必须 > 0）
- `--num_return_sequences`（默认 `5`，必须 > 0）
- `--top_k`（默认 `50`，必须 > 0）
- `--temperature`（默认 `1.0`，必须 > 0）
- `--seed`（默认 `42`）

---

## 5. 配置系统（YAML 继承）

### 5.1 解析规则

- `--config baseline`：会自动在 `configs/experiments/` 解析
- `--config path/to/xxx.yaml`：支持相对/绝对路径
- `base:` 支持字符串或列表，按顺序深度合并
- 检测循环继承并报错

### 5.2 实验配置必填字段

- `experiment_name`
- `model`
- `components`
- `train`

`train` 必填键：
`max_lr`, `min_lr`, `warmup_steps`, `max_steps`, `weight_decay`, `total_batch_size`, `micro_batch_size`, `sequence_length`

### 5.3 组件键（`core/registry.py`）

- `components.attention`: `base`, `mqa`, `gqa`, `mla`
- `components.position_encoding`: `learned`, `alibi`, `rope`, `sine`, `sinusoidal`
- `components.mlp`: `default`, `mlp`, `relu`, `silu`, `swiglu`, `geglu`, `moe`
- `components.norm`: `default`, `layernorm`, `rmsnorm`

### 5.4 常见扩展字段（写在 `model:`）

- GQA：`n_kv_head`
- MLA：`kv_lora_rank`, `q_lora_rank`, `mla_cache_mode`（`full`/`latent`）
- MoE：`n_experts`, `moe_top_k`, `moe_capacity_factor`, `moe_router_noise`, `moe_expert_type`
- 初始化：`init_method`（`default/xavier/kaiming`）、`init_distribution`（`normal/uniform`）

MoE 训练可额外配置 `train.moe_aux_weight`（控制辅助损失权重）。

### 5.5 当前实验配置清单

`alibi`, `baseline`, `geglu`, `gqa`, `mla`, `moe`, `mqa`, `relu`, `rmsnorm`, `rope`, `silu`, `sine`, `swiglu`

---

## 6. 训练与推理行为（代码对齐）

### 6.1 运行时与设备

- 单进程自动选设备：`cuda -> mps -> cpu`
- DDP 由 `RANK` 环境变量触发，backend 固定 `nccl`（需要 CUDA）
- autocast 策略：
  - `cuda`: bfloat16
  - `mps`: float16（不可用时回退）
  - `cpu`: bfloat16（不可用时回退）

### 6.2 训练循环关键逻辑

- 梯度累积步数：
  `grad_accum_steps = total_batch_size / (micro_batch_size * sequence_length * world_size)`
- 若 `total_batch_size` 不能整除上式分母会直接报错
- 训练前会 fail-fast 检查每个 shard 的 token 长度是否满足
  `B * T * world_size + 1`
- 学习率：warmup + cosine decay（`core/training_utils.py`）
- 每 `250` step（含 step=0）和最后一步做验证 loss（固定 20 个 batch）
- 每 `250` step（含 step=0）和最后一步做 HellaSwag 评估（`use_compile=False` 时）
- 每 `250` step（step>0）和最后一步保存 checkpoint
- 每 `250` step（step>0）和最后一步生成样本文本（`use_compile=False` 时）

### 6.3 日志与输出目录

输出目录：

```text
log_train/<experiment_name>/<log_subdir>/
├─ log.txt
└─ model_XXXXX.pt
```

`log.txt` 当前行格式：
- 训练：`<step> train <total_loss> ce=<ce_loss> aux=<aux_loss>`
- 验证：`<step> val <val_loss>`
- HellaSwag：`<step> hella <acc>`

---

## 7. Checkpoint 规范（v2）

主流程仅支持 `schema_version=2`。

常见字段：
- 必需（推理最小集合）：`schema_version`, `model`, `config_dict`
- 训练恢复相关：`step`, `optimizer`, `train_loader_state`
- 元数据：`experiment_name`, `config_ref`, `val_loss`

恢复训练时会尝试恢复优化器与数据加载进度；若 shard/位置不匹配会直接报错中止。

旧 checkpoint 可用迁移脚本转换到 v2。

---

## 8. 多卡训练

### 8.1 手动 `torchrun`

```bash
torchrun --standalone --nproc_per_node=8 train.py train \
  --config baseline \
  --data_root /path/to/shards \
  --log_subdir log
```

### 8.2 夜间脚本 `train_nightly.sh`

```bash
CONFIG_PATH=configs/experiments/mla.yaml \
LOG_SUBDIR=log \
DATASET_ROOT=/path/to/shards \
bash train_nightly.sh
```

行为：
- 自动检测 GPU 数量
- 自动查找 `log_train/<experiment>/<log_subdir>/model_*.pt` 最新 checkpoint 并 `--resume`
- 若设置 `INIT_FROM`，则优先走 `--init_from`（不会自动 `--resume`）

---

## 9. 评估与工具脚本

统一示例（Windows PowerShell）：

```powershell
$PY = "D:\Softwares\Anaconda\envs\torch\python.exe"
```

### 9.1 `scripts/eval_lengths.py`（多长度验证）

```powershell
& $PY scripts/eval_lengths.py `
  --config baseline `
  --checkpoint log_train\baseline\log\model_15000.pt `
  --lengths 512,1024,2048 `
  --data_root D:\path\to\shards `
  --val_steps 20
```

参数：`--config`, `--checkpoint`, `--lengths`, `--data_root`, `--val_steps`, `--batch_size`, `--device`。

### 9.2 `scripts/kv_cache_eval.py`（KV cache 速度/一致性）

```powershell
& $PY scripts/kv_cache_eval.py `
  --config baseline `
  --checkpoint log_train\baseline\log\model_15000.pt `
  --max_length 64 `
  --num_return_sequences 1 `
  --dtype bfloat16
```

参数：`--config`, `--checkpoint`, `--log_subdir`, `--log_dir`, `--prompt`, `--max_length`, `--top_k`, `--temperature`, `--num_return_sequences`, `--seed`, `--dtype`, `--allow_long`。

长序列说明：
- 当 `max_length > block_size` 时必须显式加 `--allow_long`
- 仅 `RoPE/ALiBi/Sinusoidal` 支持外推，`Learned` 不支持
- 即使允许长序列，`prompt` 长度仍需 `<= block_size`
- 超长场景只跑 KV cache 路径（不跑 no-cache 对照）

### 9.3 `scripts/eval_hellaswag.py`（HF 模型基线）

```powershell
& $PY scripts/eval_hellaswag.py --model_type gpt2 --device cuda
```

参数：`-m/--model_type`, `-d/--device`。

### 9.4 `scripts/migrate_checkpoints.py`（迁移历史 checkpoint 到 v2）

```powershell
& $PY scripts/migrate_checkpoints.py --input D:\path\to\ckpt_or_dir --backup
```

参数：`--input`, `--pattern`, `--no_recursive`, `--output_root`, `--dry_run`, `--force`, `--backup`, `--backup_suffix`, `--overwrite_backup`。

### 9.5 `scripts/plot_training_log.py`（从 stdout 绘图）

```powershell
& $PY scripts/plot_training_log.py nohup_log.txt training_curves.png 1000
```

注意：该脚本解析训练 stdout（`step ... | loss ...`）格式，不解析 `log.txt`。

### 9.6 `scripts/preprocess_data.py`（离线预处理）

核心参数：
- 输入与格式：`--input`, `--format`
- 输出：`--output_dir`, `--prefix`
- 字段与分词：`--text_field`, `--tokenizer`, `--add_eot`
- 切分：`--val_ratio` 或 `--split_field + --train_split/--val_split/--test_split`
- shard：`--shard_tokens`, `--min_tail_tokens`
- 过滤：`--min_chars`, `--max_chars`
- 其他：`--streaming`, `--seed`

---

## 10. 最小回归建议

仓库当前没有独立测试套件。改动核心逻辑后，建议至少执行：

1. 短程训练（`train.py train`）
2. 推理冒烟（`train.py infer`）
3. 可选：`scripts/kv_cache_eval.py` 做一致性/速度回归

---

## 11. 常见问题

- **找不到 shard**：确认 `--data_root` 下有文件名包含 `train` / `val` 的 `.npy`
- **`total_batch_size` 断言失败**：确保能被 `micro_batch_size * sequence_length * world_size` 整除
- **infer 报配置缺失**：补 `--config <experiment>`
- **checkpoint 版本不支持**：先运行 `scripts/migrate_checkpoints.py` 迁移到 v2
- **DDP 启动失败**：确认 CUDA/NCCL 可用，并使用 `torchrun`
