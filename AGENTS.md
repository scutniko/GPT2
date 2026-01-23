# Repository Guidelines

## Language Preference
- All user-facing text should be in Chinese, including logs, status messages, and explanations.
- Do not output chain-of-thought or hidden reasoning; provide concise Chinese answers and results only.

## Project Structure & Module Organization
- `train.py` is the main entry point for training, resume, and inference.
- `core/` contains configuration, data loading, and training utilities.
- `models/` defines the GPT model (`gpt.py`).
- `modules/` contains attention blocks, MLP variants, normalization, and position encodings.
- `experiments/` holds experiment configs (e.g., `baseline.py`, `rope.py`, `mqa.py`).
- `hellaswag/` stores cached HellaSwag `.jsonl` datasets; `hellaswag.py` handles download/eval.
- `train_nightly.sh` runs multi‑GPU training and auto‑resumes from the latest checkpoint.

## Build, Test, and Development Commands
- Install deps: `python -m pip install -r requirements.txt`
- Train (single process): `python train.py --experiment baseline`
- Resume: `python train.py --experiment baseline --resume log_train/baseline/log/model_15000.pt`
- Inference: `python train.py --experiment baseline --inference log_train/baseline/log/model_15000.pt`
- Nightly multi‑GPU: `EXPERIMENT=mla bash train_nightly.sh`
- HellaSwag eval: `python hellaswag.py --model_type gpt2 --device cuda`

## Coding Style & Naming Conventions
- Python with 4‑space indentation.
- Modules and functions use `snake_case`; classes use `CamelCase` (e.g., `GPT`, `GPTConfig`).
- Keep experiment configs in `experiments/` and name files after the variant (`rope.py`, `gqa.py`).
- Prefer lightweight comments for non‑obvious math or distributed logic.

## Testing Guidelines
- No dedicated test suite in this repo.
- Use `hellaswag.py` to sanity‑check model quality and regression trends.
- If you add tests, place them under `tests/` and use `test_*.py` naming.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative subjects like `add ...`, `fix ...`, `modify ...`.
- Keep commits focused on one change area.
- PRs should describe the experiment or model change, training settings, and any metrics (e.g., HellaSwag accuracy).

## Security & Configuration Tips
- Checkpoint paths are under `log_train/<experiment>/log/`; avoid committing large checkpoints.
- For DDP, ensure `torchrun` and CUDA are available before running `train_nightly.sh`.
