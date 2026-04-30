#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXP_NAME="${1:-pomo_model_zshuffle_stage1}"
CUDA_DEVICE_NUM="${CUDA_DEVICE_NUM:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt}"

python train.py \
  --exp_name "${EXP_NAME}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --use_cuda true \
  --cuda_device_num "${CUDA_DEVICE_NUM}" \
  --epochs 200 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --use_polynet true \
  --z_dim 16 \
  --poly_embedding_dim 256 \
  --force_first_move true \
  --shuffle_z_train true \
  --curriculum "1:100" \
  --distribution_mix "uniform:1.0" \
  --dynamic_batch_sizes "100:64" \
  --load_optimizer false \
  --load_scheduler false \
  --reset_epoch true \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true \
  --detailed_log false
