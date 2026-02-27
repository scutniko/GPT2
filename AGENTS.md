# Repository Guidelines

## 项目结构与模块组织
- `train.py`：统一入口，包含 `train` / `infer` 子命令。  
- `core/`：运行时初始化、CLI 参数、实验配置加载、checkpoint 读写、数据加载与训练循环。  
- `models/gpt.py`：GPT 主模型实现。  
- `modules/`：可插拔组件（attention / position encoding / mlp / norm / block）。  
- `configs/base/`：公共基础配置；`configs/experiments/`：实验变体（如 `baseline.yaml`、`rope.yaml`、`mqa.yaml`）。  
- `scripts/`：预处理与评估工具（如 `preprocess_data.py`、`eval_lengths.py`、`kv_cache_eval.py`、`migrate_checkpoints.py`）。  
- `benchmarks/hellaswag/`：HellaSwag 下载、样本渲染与评估逻辑。  
- 训练输出统一写入 `log_train/<experiment>/<log_subdir>/`。  

## Python 环境约定
- 运行 Python 相关命令时，统一使用解释器：`D:\Softwares\Anaconda\envs\torch\python.exe`。
- 不要使用系统默认 `python` 或其他环境中的解释器。
- 命令示例：`D:\Softwares\Anaconda\envs\torch\python.exe train.py train --config baseline --data_root <shards_dir> --log_subdir log`。

## 构建、测试与开发命令
- 安装依赖：`python -m pip install -r requirements.txt`  
- 预处理数据：`python scripts/preprocess_data.py --input <raw_data> --output_dir <shards_dir> --format auto --text_field text`  
- 单进程训练：`python train.py train --config baseline --data_root <shards_dir> --log_subdir log`  
- 恢复训练：`python train.py train --config baseline --data_root <shards_dir> --log_subdir log --resume log_train/baseline/log/model_15000.pt`  
- 推理：`python train.py infer --checkpoint log_train/baseline/log/model_15000.pt`  
- 夜间多卡：`CONFIG_PATH=configs/experiments/mla.yaml LOG_SUBDIR=log DATASET_ROOT=<shards_dir> bash train_nightly.sh`  

## 编码风格与命名规范
- Python 统一 4 空格缩进。  
- 命名规范：函数/模块使用 `snake_case`，类使用 `CamelCase`（如 `GPT`, `GPTConfig`）。  
- 新实验优先通过 YAML 继承组织：公共项放 `configs/base/*.yaml`，差异项放 `configs/experiments/*.yaml`。  
- 注释聚焦复杂逻辑（数学推导、分布式行为），避免冗余注释。  

## 测试指南
- 当前无独立测试框架；核心改动后至少执行一次短程 `train` + `infer`。  
- 可选回归：  
  - `python scripts/kv_cache_eval.py --config baseline --checkpoint <ckpt> --max_length 64 --num_return_sequences 1`  
  - `python scripts/eval_lengths.py --config baseline --checkpoint <ckpt> --lengths 512,1024 --data_root <shards_dir>`  
- 如新增测试，请放在 `tests/`，命名为 `test_*.py`。  

## 提交与 PR 规范
- 提交信息使用简短祈使句，遵循现有风格：`add ...`、`fix ...`、`modify ...`、`refactor ...`。  
- 每次提交聚焦单一主题。  
- PR 建议包含：改动目的、关键配置、复现实验命令、核心指标（如 HellaSwag 准确率）与潜在风险。  

## 安全与配置提示
- 不要提交大体积 checkpoint 或模型权重。  
- 主训练/推理流程仅支持 `schema_version=2` checkpoint；旧文件请先执行 `python scripts/migrate_checkpoints.py` 迁移。  
