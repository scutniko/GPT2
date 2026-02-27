# GPT 类语言模型模块化训练框架

本项目是一个面向 GPT 系列语言模型实验的训练框架，核心目标是:

- 通过 YAML 配置快速切换实验变量（attention / position encoding / MLP / norm）。
- 统一训练、恢复训练、推理、评估和 checkpoint 管理流程。
- 支持单卡与 DDP 多卡训练，支持 KV cache 推理对比和长度外推评估。

---

## 1. 项目概览

### 1.1 功能特性

- 统一入口: `train.py train` / `train.py infer`
- 可插拔组件:
  - 注意力: `base`, `mqa`, `gqa`, `mla`
  - 位置编码: `learned`, `rope`, `alibi`, `sine`
  - MLP: `mlp`, `relu`, `silu`, `swiglu`, `geglu`, `moe`
  - 归一化: `layernorm`, `rmsnorm`
- 配置继承: 支持 `base` 字段递归继承与深度合并
- checkpoint: 统一使用 `schema_version=2`
- 内置评估:
  - 训练中周期性验证 loss
  - 训练中周期性 HellaSwag 验证
  - 多长度验证评估脚本
  - KV cache 速度/一致性评估脚本

### 1.2 目录结构

| 路径 | 说明 |
|---|---|
| `train.py` | 训练/推理统一入口 |
| `core/` | CLI、运行时、实验加载、checkpoint、数据加载、训练循环、评估逻辑 |
| `models/gpt.py` | GPT 主模型定义 |
| `modules/` | attention / position / mlp / norm / block 组件 |
| `configs/base/` | 基础模型与训练超参 |
| `configs/experiments/` | 具体实验配置 |
| `scripts/` | 数据预处理、迁移、评估、可视化工具 |
| `benchmarks/hellaswag/` | HellaSwag 下载、样本渲染、评估 |
| `hellaswag/` | HellaSwag 数据缓存目录 |
| `log_train/` | 训练日志与 checkpoint 输出目录 |

---

## 2. 环境与安装

### 2.1 Python 命令约定

本文档命令统一使用 `python`。

### 2.2 安装依赖

```bash
python -m pip install -r requirements.txt
```

---

## 3. 数据预处理

训练输入要求是离线 `.npy` token shard（1 维整型数组），文件名需包含 `train` / `val` 字样。

### 3.1 预处理脚本

```bash
python scripts/preprocess_data.py \
  --input <raw_data_or_dir> \
  --format auto \
  --output_dir <shards_dir> \
  --prefix dataset \
  --text_field text \
  --tokenizer gpt2 \
  --val_ratio 0.01 \
  --shard_tokens 50000000 \
  --min_tail_tokens 1024
```

常用参数:

- `--format`: `auto/jsonl/parquet`
- `--split_field`: 使用样本已有分割字段（否则按 `val_ratio` 随机划分）
- `--add_eot`: 每条样本末尾追加 EOT
- `--streaming`: 低内存流式读取

### 3.2 shard 长度约束

训练启动前会校验每个 shard 的 token 数至少满足:

`min_tokens_required = micro_batch_size * sequence_length * world_size + 1`

如果 shard 过短会直接报错并退出（fail-fast）。

---

## 4. 训练

### 4.1 最小训练命令

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir log
```

### 4.2 常见启动场景（可直接复制）

1. 从零开始训练（推荐首次跑通）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir run1
```

2. 从已有 checkpoint 续训（恢复 step/optimizer/loader）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir run1 \
  --resume log_train/baseline/run1/model_02500.pt
```

3. 用旧权重启动新实验（只加载模型参数）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir run2_finetune \
  --init_from log_train/baseline/run1/model_02500.pt
```

4. 快速烟雾测试（先把 `max_steps` 临时改小，比如 20）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir smoke
```

### 4.3 训练参数

`train.py train` 参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--config` | 是 | 无 | 实验配置名或配置路径 | 可写 `baseline` 或 `configs/experiments/baseline.yaml` |
| `--data_root` | 是 | 无 | token shard 根目录 | 目录内需有包含 `train` / `val` 的 `.npy` 文件 |
| `--log_subdir` | 是 | 无 | 日志子目录 | 最终输出路径: `log_train/<experiment>/<log_subdir>/` |
| `--resume` | 否 | `None` | 从 checkpoint 继续训练 | 恢复模型、优化器、dataloader 位置、step |
| `--init_from` | 否 | `None` | 从 checkpoint 加载初始权重 | 只加载模型参数，不恢复优化器和step |

参数约束:

- `--resume` 与 `--init_from` 不能同时传。
- `--resume` 的 checkpoint 必须是 `schema_version=2` 且包含优化器与 loader 状态。
- `--init_from` 只要求 checkpoint 中有 `model` 权重。

### 4.4 训练时默认行为

- 设备选择（单进程）: `cuda > mps > cpu`
- DDP: 通过 `torchrun` 环境变量自动识别，后端为 `nccl`
- 固定随机种子: `1337`
- 自动计算梯度累积:
  - `grad_accum_steps = total_batch_size / (B * T * world_size)`
- 学习率策略: warmup + cosine decay
- 周期性评估:
  - 每 `250` step（以及最后一步）评估 val loss
  - 每 `250` step（以及最后一步）评估 HellaSwag
- checkpoint 保存:
  - 在评估点保存 `model_{step:05d}.pt`
  - 第 `0` 步不会保存 checkpoint（代码里要求 `step > 0`）

### 4.5 启动训练前检查清单

1. `--data_root` 下是否有 `*train*.npy` 和 `*val*.npy`
2. 每个 shard token 数是否满足 `B*T*world_size+1`
3. `total_batch_size` 是否能被 `micro_batch_size*sequence_length*world_size` 整除
4. `--resume` 指向的 checkpoint 是否为 `schema_version=2`
5. 是否误同时传了 `--resume` 和 `--init_from`

### 4.6 训练输出解读

控制台会持续打印:

- `step ... | loss ... | ce ... | aux ... | lr ... | norm ... | tok/sec ...`
- `validation loss: ...`
- `HellaSwag accuracy: ...`

落盘文件:

- `log_train/<experiment>/<log_subdir>/log.txt`
- `log_train/<experiment>/<log_subdir>/model_XXXXX.pt`

如果是 `resume`，程序会自动:

- 恢复优化器状态
- 恢复 dataloader 的 shard/position
- 截断 `log.txt` 到恢复步之前（避免日志重复）

---

## 5. 恢复训练与热启动

### 5.1 恢复训练（resume）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir log \
  --resume log_train/baseline/log/model_15000.pt
```

恢复内容:

- 模型参数
- 优化器状态（并迁移到当前设备）
- 数据加载器状态（`current_shard` 与 `rank0_position`）
- 日志文件裁剪到恢复步之前

### 5.2 权重热启动（init_from）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir log_finetune \
  --init_from log_train/baseline/log/model_15000.pt
```

只加载模型，不恢复优化器与 step。

---

## 6. 推理

### 6.1 基本推理

```bash
python train.py infer \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --prompt "Hello, I'm a language model," \
  --max_length 64 \
  --num_return_sequences 4 \
  --top_k 50 \
  --temperature 1.0 \
  --seed 42
```

### 6.2 配置来源规则

- 如果不传 `--config`，推理会读取 checkpoint 内 `config_ref` 自动还原实验配置。
- 即使传了 `--config`，当前推理入口也要求 checkpoint 内存在 `config_ref` 字段（会先做 schema 校验）。

### 6.3 长上下文限制

- `learned` 位置编码不支持超过 `block_size`
- `rope` / `alibi` / `sine` 支持长上下文推理

### 6.4 推理参数详解

`train.py infer` 参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--checkpoint` | 是 | 无 | 推理用 checkpoint 路径 | 需为 `schema_version=2`，且包含 `config_ref` |
| `--config` | 否 | `None` | 显式指定配置 | 不传则自动读取 checkpoint 的 `config_ref` |
| `--prompt` | 否 | `"Hello, I'm a language model,"` | 提示词 | 会计入 `max_length` 总长度 |
| `--max_length` | 否 | `32` | 生成总长度（含 prompt） | 必须 `> 0`，且不能小于 prompt token 长度 |
| `--num_return_sequences` | 否 | `5` | 返回条数 | 必须 `> 0` |
| `--top_k` | 否 | `50` | Top-k 采样 | 必须 `> 0` |
| `--temperature` | 否 | `1.0` | 采样温度 | 必须 `> 0` |
| `--seed` | 否 | `42` | 随机种子 | 同样输入下可复现采样结果 |

---

## 7. 多卡训练

### 7.1 手工 torchrun

```bash
torchrun --standalone --nproc_per_node=8 train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir log
```

### 7.2 夜间脚本 `train_nightly.sh`

```bash
CONFIG_PATH=configs/experiments/mla.yaml \
LOG_SUBDIR=log \
DATASET_ROOT=<shards_dir> \
bash train_nightly.sh
```

运行环境说明:

- `train_nightly.sh` 是 Bash 脚本，建议在 Linux/WSL/Git Bash 中运行
- 若当前环境不便运行 `train_nightly.sh`，可直接使用上面的 `torchrun` 命令

脚本行为:

- 自动检测 GPU 数量
- 在 `log_train/<experiment>/<log_subdir>/` 自动寻找最新 checkpoint 恢复
- 可通过 `INIT_FROM` 指定仅加载权重的启动方式

### 7.3 夜间脚本参数详解

| 环境变量 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `CONFIG_PATH` | 否 | `configs/experiments/mla.yaml` | 配置路径 | 支持相对/绝对路径，最终透传到 `--config` |
| `LOG_SUBDIR` | 是 | 无 | 日志子目录 | 留空会报错退出 |
| `DATASET_ROOT` | 是 | 无 | 数据目录 | 对应 `train.py train --data_root` |
| `EXPERIMENT_NAME` | 否 | 配置文件名 | 实验名覆盖 | 影响 `log_train/<experiment>/...` 路径 |
| `INIT_FROM` | 否 | 未设置 | 初始化权重路径 | 设定后优先走 `--init_from`，并跳过自动 `--resume` |

### 7.4 夜间脚本常用启动模板

1. 全新训练（不指定 `INIT_FROM`，且目录无 checkpoint）

```bash
CONFIG_PATH=configs/experiments/baseline.yaml \
LOG_SUBDIR=nightly_run1 \
DATASET_ROOT=/data/my_shards \
bash train_nightly.sh
```

2. 自动续训（目录下有 `model_*.pt`）

```bash
CONFIG_PATH=configs/experiments/baseline.yaml \
LOG_SUBDIR=nightly_run1 \
DATASET_ROOT=/data/my_shards \
bash train_nightly.sh
```

3. 只加载初始化权重（迁移到新目录）

```bash
CONFIG_PATH=configs/experiments/rope.yaml \
LOG_SUBDIR=nightly_rope_init \
DATASET_ROOT=/data/my_shards \
INIT_FROM=/abs/path/model_05000.pt \
bash train_nightly.sh
```

### 7.5 夜间脚本常见失败与定位

1. 报 `必须设置 LOG_SUBDIR`
- 原因: 未传 `LOG_SUBDIR`
- 处理: 补上环境变量后重试

2. 报 `必须设置 DATASET_ROOT`
- 原因: 未传数据目录
- 处理: 指向预处理后的 shard 目录

3. 报 `No GPU detected`
- 原因: 脚本只用于 GPU 训练，且会检查 `CUDA_VISIBLE_DEVICES`/`nvidia-smi`
- 处理: 检查驱动、CUDA 可见设备或改用单进程 CPU/GPU 命令

4. 报 `配置文件不存在`
- 原因: `CONFIG_PATH` 写错
- 处理: 使用项目根目录相对路径或绝对路径

5. 明明有 checkpoint 但没自动续训
- 常见原因: `LOG_SUBDIR` 不一致，或目录下文件名不符合 `model_*.pt`
- 处理: 检查实际输出目录与文件命名

---

## 8. 配置系统（YAML）

### 8.1 配置解析规则

- `--config` 可传:
  - 短名（如 `baseline`）
  - 相对路径
  - 绝对路径
- `base` 字段支持字符串或列表，支持递归继承
- 合并策略为深度合并（同名字段递归覆盖）

### 8.2 实验配置必填字段

顶层必填:

- `experiment_name`
- `model`
- `components`
- `train`

`train` 内必填:

- `max_lr`, `min_lr`, `warmup_steps`, `max_steps`
- `weight_decay`, `total_batch_size`, `micro_batch_size`, `sequence_length`

### 8.3 组件键映射

- `components.attention`: `base/mqa/gqa/mla`
- `components.position_encoding`: `learned/alibi/rope/sine/sinusoidal`
- `components.mlp`: `default/mlp/relu/silu/swiglu/geglu/moe`
- `components.norm`: `default/layernorm/rmsnorm`

### 8.4 内置实验

| 实验 | attention | position | mlp | norm | 额外字段 |
|---|---|---|---|---|---|
| `baseline` | base | learned | default | layernorm | - |
| `rope` | base | rope | default | layernorm | 覆盖 `warmup_steps/max_steps` |
| `alibi` | base | alibi | default | layernorm | - |
| `sine` | base | sine | default | layernorm | - |
| `mqa` | mqa | learned | default | layernorm | - |
| `gqa` | gqa | learned | default | layernorm | `model.n_kv_head` |
| `mla` | mla | learned | default | layernorm | `kv_lora_rank/q_lora_rank/mla_cache_mode` |
| `relu` | base | learned | relu | layernorm | - |
| `silu` | base | learned | silu | layernorm | - |
| `swiglu` | base | learned | swiglu | layernorm | - |
| `geglu` | base | learned | geglu | layernorm | - |
| `rmsnorm` | base | learned | default | rmsnorm | - |
| `moe` | base | learned | moe | layernorm | `n_experts/moe_top_k/...` + `train.moe_aux_weight` |

### 8.5 新增实验模板

```yaml
base:
  - ../base/model_124m.yaml
  - ../base/train_default.yaml

experiment_name: my_exp

components:
  attention: base
  position_encoding: learned
  mlp: default
  norm: layernorm

model:
  # 可选: 在 GPTConfig 基础上扩展字段
  # n_kv_head: 4

train:
  # 可选: 覆盖 base 训练超参
  # max_steps: 5000
```

---

## 9. checkpoint 与日志

### 9.1 checkpoint schema（v2）

主训练/推理仅支持 `schema_version=2`。

典型字段:

- `schema_version`
- `model`
- `config_dict`
- `step`, `val_loss`
- `optimizer`
- `train_loader_state`:
  - `current_shard`
  - `rank0_position`
- `experiment_name`（可选）
- `config_ref`（推理入口要求该字段存在）

### 9.2 日志目录

- checkpoint: `log_train/<experiment>/<log_subdir>/model_XXXXX.pt`
- 训练日志: `log_train/<experiment>/<log_subdir>/log.txt`

`log.txt` 行格式示例:

```text
0 train 10.548814 ce=10.548814 aux=0.000000
0 val 10.3045
0 hella 0.2465
```

---

## 10. 工具脚本

### 10.1 脚本总览

| 脚本 | 作用 | 常见使用时机 |
|---|---|---|
| `scripts/eval_lengths.py` | 同一 checkpoint 在多个序列长度上评估 val loss/ppl | 看长度外推或长上下文退化 |
| `scripts/kv_cache_eval.py` | 对比 KV cache 与非 cache 的速度、输出一致性、显存峰值 | 推理性能分析 |
| `scripts/migrate_checkpoints.py` | 旧 checkpoint 批量迁移到 v2 schema | 加载历史模型失败时 |
| `scripts/eval_hellaswag.py` | 用 HF GPT 模型跑 HellaSwag 基线 | 和公开模型做参考对比 |
| `scripts/plot_training_log.py` | 从控制台日志绘制训练曲线 | 训练后分析趋势 |
| `scripts/preprocess_data.py` | 原始数据转 `.npy` shards | 训练前数据准备 |

### 10.2 `eval_lengths.py`（多长度验证评估）

```bash
python scripts/eval_lengths.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --lengths 512,1024,2048 \
  --data_root <shards_dir> \
  --val_steps 20
```

适用场景:

- 对比 `T=512/1024/2048` 的损失变化
- 判断位置编码是否具备长度外推能力

参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--config` | 是 | 无 | 实验配置名/路径 | 用于确定组件组合（attention/position/mlp/norm） |
| `--checkpoint` | 是 | 无 | 待评估 checkpoint | 需能和 `--config` 组合正确加载模型 |
| `--lengths` | 是 | 无 | 逗号分隔长度列表 | 例: `512,1024,2048`；不能为空 |
| `--data_root` | 是 | 无 | val 数据根目录 | 需要可读取到 val shard |
| `--val_steps` | 否 | `20` | 每个长度评估 batch 数 | 越大越稳定，耗时越长 |
| `--batch_size` | 否 | `None` | 评估 batch size | 不传时取训练配置的 `micro_batch_size` |
| `--device` | 否 | 自动 | `cuda/mps/cpu` | 不传时自动按 `cuda > mps > cpu` 选择 |

参数约束:

- `--lengths` 中每一项必须是正整数。

输出解释:

- 每个长度输出一行: `T=xxxx | val_loss=... | ppl=...`
- 建议关注趋势而不是单点
- `ppl = exp(val_loss)`，loss 的小变化会放大为 ppl 变化

- `learned` 位置编码不能外推到超过训练 `block_size`
- `rope/alibi/sine` 可评估更长长度

### 10.3 `kv_cache_eval.py`（KV cache 推理评估）

```bash
python scripts/kv_cache_eval.py \
  --config baseline \
  --checkpoint log_train/baseline/log/model_15000.pt \
  --max_length 64 \
  --num_return_sequences 1 \
  --dtype bfloat16
```

适用场景:

- 验证 KV cache 是否提升 tok/s
- 验证 KV cache 路径与非 cache 路径是否逐 token 一致
- 观察 CUDA 峰值显存变化

参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--config` | 是 | 无 | 实验配置名/路径 | 用于构建模型结构 |
| `--checkpoint` | 否 | `None` | 指定 checkpoint | 不传时需配合 `--log_subdir` 或 `--log_dir` |
| `--log_subdir` | 否 | `None` | 日志子目录 | 自动找 checkpoint 时路径为 `log_train/<exp>/<log_subdir>/` |
| `--log_dir` | 否 | `None` | 日志目录绝对/相对路径 | 自动找 checkpoint 时优先级高于 `--log_subdir` |
| `--prompt` | 否 | `"Hello, I'm a language model,"` | 提示词 | 会影响可生成 token 数 |
| `--max_length` | 否 | `64` | 生成总长度（含 prompt） | 超过训练长度需配合 `--allow_long` |
| `--top_k` | 否 | `50` | Top-k 采样 | 一般不超过词表大小 |
| `--temperature` | 否 | `1.0` | 采样温度 | 温度越低越保守 |
| `--num_return_sequences` | 否 | `1` | 并行生成条数 | 会线性增加显存占用 |
| `--seed` | 否 | `42` | 随机种子 | 保证可复现采样 |
| `--dtype` | 否 | `float32` | 推理精度 | `bfloat16` 在支持设备上更快、更省显存 |
| `--allow_long` | 否 | `False` | 允许超训练长度生成 | 仅适用于 `RoPE/ALiBi/Sinusoidal` |

输出解释:

- 速度对比: no-cache vs cache 的总耗时和 tok/s
- 一致性检查: 是否逐 token 完全一致
- KV cache 形状: 首层 cache tensor 形状
- 显存统计: CUDA 设备下显示峰值 MB

注意事项:

- 不传 `--checkpoint` 时，配合 `--log_subdir`/`--log_dir` 自动找最新 checkpoint
- `--allow_long` 允许超过训练长度（仅支持 RoPE/ALiBi/正弦）
- 当 `max_length > block_size` 时，脚本会跳过 no-cache 路径，只跑 KV cache 路径

### 10.4 `migrate_checkpoints.py`（旧版 checkpoint 迁移）

```bash
python scripts/migrate_checkpoints.py \
  --input log_train \
  --pattern "model_*.pt" \
  --dry_run
```

推荐两阶段执行:

1. 先预览（不写文件）

```bash
python scripts/migrate_checkpoints.py \
  --input log_train \
  --pattern "model_*.pt" \
  --dry_run
```

2. 再正式迁移（就地覆盖 + 备份）

```bash
python scripts/migrate_checkpoints.py \
  --input log_train \
  --pattern "model_*.pt" \
  --backup
```

参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--input` | 是 | 无 | checkpoint 文件或目录 | 目录模式会批量扫描 |
| `--pattern` | 否 | `model_*.pt` | 扫描匹配模式 | `--input` 为目录时生效 |
| `--no_recursive` | 否 | `False` | 关闭递归扫描 | 默认递归 |
| `--output_root` | 否 | `None` | 输出根目录 | 不传则原地覆盖 |
| `--dry_run` | 否 | `False` | 预览模式 | 不写文件，仅打印 |
| `--force` | 否 | `False` | 强制重写 v2 文件 | 即使已是 v2 也会重写 |
| `--backup` | 否 | `False` | 覆盖前备份 | 原地覆盖时建议开启 |
| `--backup_suffix` | 否 | `.v1.bak` | 备份后缀 | 仅 `--backup` 生效 |
| `--overwrite_backup` | 否 | `False` | 允许覆盖已有备份文件 | 仅 `--backup` 生效 |

输出解释:

- `[跳过]`: 已是 v2 且未 `--force`
- `[完成]`: 已成功迁移
- `[失败]`: 当前文件迁移失败（会给异常类型）

### 10.5 `eval_hellaswag.py`（HF 模型基线评估）

```bash
python scripts/eval_hellaswag.py --model_type gpt2 --device cuda
```

说明:

- 该脚本评估的是 HuggingFace 模型，不读取本项目 checkpoint
- 首次运行会自动下载 HellaSwag 到 `hellaswag/`

参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `-m`, `--model_type` | 否 | `gpt2` | HuggingFace 模型名 | 常见值: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| `-d`, `--device` | 否 | `cuda` | 评估设备 | 无 GPU 时改为 `cpu` 或 `mps` |

### 10.6 `plot_training_log.py`（训练日志绘图）

先保存训练控制台输出，再绘图。

1. 训练并保存控制台日志（Git Bash）

```bash
python train.py train \
  --config baseline \
  --data_root <shards_dir> \
  --log_subdir plot_demo 2>&1 | tee train_console.log
```

2. 绘制曲线

```bash
python scripts/plot_training_log.py train_console.log training_curves.png 0
```

参数说明:

| 位置参数 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|
| `log_file` | `nohup_log.txt` | 控制台日志路径 | 必须包含 `step ... | loss:` 文本 |
| `output_file` | `training_curves.png` | 输出图片路径 | 同名文件会被覆盖 |
| `start_step` | `1000` | 绘图起始步数 | 可跳过训练早期抖动 |

说明:

- 脚本解析的是控制台日志（含 `step ... | loss:`），不是 `log.txt` 三列格式
- 第三个参数是起始 step（例如只看后半程可设 `1000`）
- 输出图含训练 loss、验证 loss、HellaSwag accuracy 三个子图

### 10.7 `preprocess_data.py`（离线数据预处理）

该脚本在第 3 节已介绍，这里给出常见组合模板。

1. 自动识别格式（jsonl/parquet）

```bash
python scripts/preprocess_data.py \
  --input <raw_data_dir> \
  --format auto \
  --output_dir <shards_dir> \
  --text_field text \
  --prefix dataset
```

2. 用数据字段显式划分 train/val/test

```bash
python scripts/preprocess_data.py \
  --input <raw_data_dir> \
  --format parquet \
  --output_dir <shards_dir> \
  --text_field text \
  --split_field partition \
  --train_split train \
  --val_split val \
  --test_split test
```

3. 低内存流式预处理

```bash
python scripts/preprocess_data.py \
  --input <raw_data_dir> \
  --format jsonl \
  --output_dir <shards_dir> \
  --streaming \
  --shard_tokens 50000000 \
  --min_tail_tokens 1024
```

参数详解:

| 参数 | 必填 | 默认值 | 说明 | 注意事项 |
|---|---|---|---|---|
| `--input` | 是 | 无 | 输入文件或目录 | 支持单文件或目录递归扫描 |
| `--format` | 否 | `auto` | 输入格式 | 可选 `auto/jsonl/parquet`；`auto` 不允许混合格式 |
| `--output_dir` | 是 | 无 | 输出目录 | 会自动创建 |
| `--prefix` | 否 | `dataset` | shard 文件名前缀 | 输出文件形如 `prefix_train_000000.npy` |
| `--text_field` | 否 | `text` | 文本字段名 | 字段缺失样本会被跳过 |
| `--tokenizer` | 否 | `gpt2` | tiktoken 编码名 | 需是 tiktoken 支持的编码 |
| `--val_ratio` | 否 | `0.01` | 验证集比例 | 仅在未设置 `--split_field` 时生效 |
| `--split_field` | 否 | `None` | 样本内划分字段 | 设定后将按字段值分配 train/val/test |
| `--train_split` | 否 | `train` | 训练集字段值 | 配合 `--split_field` 使用 |
| `--val_split` | 否 | `val` | 验证集字段值 | 配合 `--split_field` 使用 |
| `--test_split` | 否 | `test` | 测试集字段值 | 配合 `--split_field` 使用 |
| `--shard_tokens` | 否 | `50000000` | 每个 shard 目标 token 数 | 必须 `> 0` |
| `--min_chars` | 否 | `1` | 样本最小字符数 | 小于阈值样本会被跳过 |
| `--max_chars` | 否 | `0` | 样本最大字符数 | `0` 表示不截断 |
| `--min_tail_tokens` | 否 | `1024` | 尾 shard 最小 token 阈值 | 小于阈值的尾 shard 丢弃；必须 `>= 0` |
| `--add_eot` | 否 | `False` | 每条样本追加 EOT | tokenizer 不支持 EOT 时会报错 |
| `--streaming` | 否 | `False` | 流式读取 | 内存占用更低，速度可能略慢 |
| `--seed` | 否 | `42` | 随机种子 | 影响随机切分可复现性 |

参数交互:

- 设定 `--split_field` 后，`--val_ratio` 不再生效。
- `--format auto` 下若检测到混合格式（jsonl + parquet）会报错，需显式指定 `--format`。

---

## 11. 常见问题

### 11.1 `total_batch_size` 无法整除

需满足:

`total_batch_size % (micro_batch_size * sequence_length * world_size) == 0`

### 11.2 shard 过短报错

- 调小 `micro_batch_size` 或 `sequence_length`
- 重新切 shard（增大 `--shard_tokens`）
- 确认每个 shard 至少有 `B*T*world_size + 1` 个 token

### 11.3 推理时配置不匹配

- 优先不传 `--config`，让程序从 checkpoint 的 `config_ref` 自动还原
- `--config` 仅用于覆盖/显式指定实验配置，不能绕过 `config_ref` 缺失导致的校验错误

### 11.4 无法加载旧 checkpoint

先执行迁移脚本转成 `schema_version=2`。

---

## 12. 最小自检流程

```bash
# 1) 训练若干步（建议先把配置里的 max_steps 临时调小）
python train.py train --config baseline --data_root <shards_dir> --log_subdir smoke

# 2) 使用最近 checkpoint 推理
python train.py infer --checkpoint log_train/baseline/smoke/model_00250.pt
```

如果以上流程可跑通，说明主链路（配置解析、训练、保存、推理加载）正常。


