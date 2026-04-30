#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXP_NAME="${1:-pomo_train_model_zshuffle_zgated_difflr}"
CUDA_DEVICE_NUM="${CUDA_DEVICE_NUM:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt}"

python train.py \
  --exp_name "${EXP_NAME}" \
  --init_checkpoint "${INIT_CHECKPOINT}" \
  --use_cuda true \
  --cuda_device_num "${CUDA_DEVICE_NUM}" \
  --epochs 500 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --use_differential_lr true \
  --backbone_lr 1e-6 \
  --poly_lr 1e-5 \
  --use_polynet true \
  --z_dim 16 \
  --poly_embedding_dim 256 \
  --force_first_move true \
  --use_z_gate true \
  --shuffle_z_train true \
  --curriculum "1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300" \
  --distribution_mix "uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05" \
  --dynamic_batch_sizes "100:64,125:48,150:40,200:24,250:16,300:12" \
  --load_optimizer false \
  --load_scheduler false \
  --reset_epoch true \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true \
  --detailed_log false
