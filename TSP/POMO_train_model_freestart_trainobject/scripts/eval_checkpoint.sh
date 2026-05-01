#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_checkpoint.sh <checkpoint_path> [output_json]" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_JSON="${2:-./result_lib/eval_checkpoint.json}"
CUDA_DEVICE_NUM="${CUDA_DEVICE_NUM:-0}"
SEED="${SEED:-42}"

python test.py \
  --data_path ../data/val \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --use_cuda true \
  --cuda_device_num "${CUDA_DEVICE_NUM}" \
  --seed "${SEED}" \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log false \
  --output_json "${OUTPUT_JSON}"
