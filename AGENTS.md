# Repository Guidelines

## 项目结构与模块组织
- `train.py`：统一入口，使用子命令 `train` / `infer`。
- `core/`：运行时初始化、YAML 实验加载、checkpoint 读写、数据加载与训练循环。
- `models/gpt.py`：GPT 主模型定义；`modules/`：attention / position / mlp / norm 可插拔组件。
- `configs/base/`：可复用基础配置；`configs/experiments/`：实验配置（如 `baseline.yaml`、`rope.yaml`、`mqa.yaml`）。
- `scripts/`：预处理、多长度评估、KV cache 评估、checkpoint 迁移工具。
- `hellaswag/` 与 `hellaswag.py`：HellaSwag 数据缓存与评估脚本。

## 构建、测试与开发命令
- 安装依赖：`python -m pip install -r requirements.txt`
- 预处理数据：`python scripts/preprocess_data.py --input <raw_data> --output_dir <shards_dir> --format auto --text_field text`
- 单进程训练：`python train.py train --config baseline --data_root <shards_dir> --log_subdir log`
- 恢复训练：`python train.py train --config baseline --data_root <shards_dir> --log_subdir log --resume log_train/baseline/log/model_15000.pt`
- 推理：`python train.py infer --checkpoint log_train/baseline/log/model_15000.pt`
- 夜间多卡：`CONFIG_PATH=configs/experiments/mla.yaml LOG_SUBDIR=log DATASET_ROOT=<shards_dir> bash train_nightly.sh`

## 编码风格与命名规范
- Python 使用 4 空格缩进。
- 函数/模块使用 `snake_case`，类使用 `CamelCase`（如 `GPT`, `GPTConfig`）。
- 新实验优先通过 YAML 继承组织：公共项放 `configs/base/*.yaml`，变体放 `configs/experiments/*.yaml`。
- 对复杂数学或分布式逻辑仅添加简短必要注释，避免冗余注释。

## 测试指南
- 仓库当前无独立测试套件；核心改动后至少执行一次短程 `train` + `infer`。
- 可选回归检查：`python scripts/kv_cache_eval.py --config baseline --checkpoint <ckpt> --max_length 64 --num_return_sequences 1`
- 质量趋势可用：`python hellaswag.py --model_type gpt2 --device cuda`
- 若新增测试，请放在 `tests/` 下并使用 `test_*.py` 命名。

## 提交与 PR 规范
- 提交信息建议简短祈使句，遵循现有风格：`add ...`、`fix ...`、`modify ...`、`refactor ...`。
- 每次提交只聚焦一个改动主题（例如“checkpoint 迁移”或“新增 rope 实验”）。
- PR 请包含：改动目的、关键配置、复现实验命令、主要指标（如 HellaSwag 准确率）与潜在风险。

## 安全与配置提示
- checkpoint 默认写入 `log_train/<experiment>/log/`；不要提交大模型权重到仓库。
- 主训练/推理流程仅支持 `schema_version=2` checkpoint；旧文件先用 `scripts/migrate_checkpoints.py` 迁移。
