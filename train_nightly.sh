#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# 模块化训练框架 - 夜间训练脚本
# 支持所有实验：baseline, alibi, rope, sine, mqa, gqa, mla
# =====================================================

# ------------------------------------------------
# 配置区域 - 修改这里来切换实验
# ------------------------------------------------
EXPERIMENT="${EXPERIMENT:-mla}"  # 可通过环境变量覆盖，默认mla
LOG_SUBDIR="${LOG_SUBDIR:-}"
DATASET_ROOT="${DATASET_ROOT:-}"

# 动态获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# 基于项目根目录构建路径
TRAIN_FILE="${PROJECT_ROOT}/train.py"
LOG_DIR="${PROJECT_ROOT}/log_train/${EXPERIMENT}/${LOG_SUBDIR}"

echo "=============================================="
echo "[nightly] Modular Training Framework"
echo "[nightly] $(date)"
echo "=============================================="
echo "[nightly] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[nightly] EXPERIMENT=${EXPERIMENT}"
echo "[nightly] LOG_SUBDIR=${LOG_SUBDIR}"
echo "[nightly] DATASET_ROOT=${DATASET_ROOT}"
echo "[nightly] TRAIN_FILE=${TRAIN_FILE}"
echo "[nightly] LOG_DIR=${LOG_DIR}"

# ------------------------------------------------
# 1. 检查训练脚本是否存在
# ------------------------------------------------
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[nightly][ERROR] Training script not found: ${TRAIN_FILE}"
  echo "[nightly][ERROR] Please make sure you're in the correct directory"
  exit 1
fi

# ------------------------------------------------
# 2. 检查日志子目录
# ------------------------------------------------
if [[ -z "${LOG_SUBDIR}" ]]; then
  echo "[nightly][ERROR] 必须设置 LOG_SUBDIR（例如 log 或 log_code）"
  exit 1
fi

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "[nightly][ERROR] 必须设置 DATASET_ROOT（数据集目录）"
  exit 1
fi

# ------------------------------------------------
# 3. 自动探测 GPU 数量
# ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
else
  GPU_COUNT=0
fi

if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "[nightly][ERROR] No GPU detected!"
  exit 1
fi

echo "[nightly] Detected GPU_COUNT=${GPU_COUNT}"

# ------------------------------------------------
# 4. 找最新 checkpoint（如果存在）
# ------------------------------------------------
mkdir -p "${LOG_DIR}"
latest_ckpt="$(ls -1t "${LOG_DIR}"/*.pt 2>/dev/null | head -n 1 || true)"

if [[ -n "${latest_ckpt}" ]]; then
  echo "[nightly] Found checkpoint: ${latest_ckpt}"
else
  echo "[nightly] No checkpoint found, starting from scratch"
fi

# ------------------------------------------------
# 5. 启动 torchrun（自动多卡）
# ------------------------------------------------
TORCHRUN_ARGS=(
  --standalone
  --nproc_per_node="${GPU_COUNT}"
)

TRAIN_ARGS=(
  --experiment "${EXPERIMENT}"
  --log_subdir "${LOG_SUBDIR}"
  --data_root "${DATASET_ROOT}"
)

if [[ -n "${INIT_FROM:-}" ]]; then
  TRAIN_ARGS+=(--init_from "${INIT_FROM}")
fi

if [[ -z "${INIT_FROM:-}" && -n "${latest_ckpt}" ]]; then
  TRAIN_ARGS+=(--resume "${latest_ckpt}")
fi

echo "=============================================="
echo "[nightly] Starting training..."
echo "[nightly] Command:"
echo "torchrun ${TORCHRUN_ARGS[*]} ${TRAIN_FILE} ${TRAIN_ARGS[*]}"
echo "=============================================="

exec torchrun \
  "${TORCHRUN_ARGS[@]}" \
  "${TRAIN_FILE}" \
  "${TRAIN_ARGS[@]}"