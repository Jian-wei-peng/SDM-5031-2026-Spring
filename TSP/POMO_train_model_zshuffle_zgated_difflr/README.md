# POMO_train_model_zshuffle_zgated_difflr: Final Combined Candidate

This directory combines the full set of approved improvements:

```text
multi-scale / mixed-distribution training
+ PolyNet-style z-conditioned decoder residual
+ random z assignment during training
+ gated z residual
+ differential learning rate
```

It answers:

```text
Do z-shuffle, gated residual, and differential LR combine into a stronger and more stable final model?
```

Compare against:

```text
TSP/POMO_train_model
TSP/POMO_train_model_zshuffle
TSP/POMO_train_model_zgated
TSP/POMO_train_model_difflr
```

## Included Changes

### Training-Side

- Multi-scale curriculum over problem sizes.
- Mixed synthetic distributions for TSPLIB-style generalization.
- Dynamic batch sizes.
- Fine-tuning from a POMO or `POMO_train` checkpoint.

### Model-Side

- PolyNet-style latent code `z`.
- Gated z-conditioned residual:

```text
gate = sigmoid(MLP_gate([mh_atten_out, z]))
mh_atten_out = mh_atten_out + gate * residual([mh_atten_out, z])
```

- Conservative gate initialization:

```text
poly_gate_2.weight = 0
poly_gate_2.bias = -4
```

### z-Shuffle

During training only:

```text
rollout i starts from node i, but receives a randomly permuted z code.
```

During evaluation:

```text
rollout i starts from node i and receives deterministic z_i.
```

### Differential LR

The optimizer uses two parameter groups:

```text
backbone parameters: lr = 1e-6
poly_layer / poly_gate parameters: lr = 1e-5
```

The trainer logs parameter counts and learning rates so the grouping can be checked.

## Fairness Constraints

- `pomo_size = problem_size` during evaluation.
- `aug_factor = 8`.
- Total candidate tours are exactly `problem_size * 8`.
- No EAS, 2-opt, LKH3, local search, beam expansion, or test-time iterative adaptation.
- Public validation data is used only for evaluation and model selection.

## Train

```bash
cd TSP/POMO_train_model_zshuffle_zgated_difflr
bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_zgated_difflr
```

Recommended if a strong `POMO_train` checkpoint is available:

```bash
INIT_CHECKPOINT=../POMO_train/result/YOUR_BEST_TRAIN/checkpoint-500.pt \
bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_zgated_difflr_from_best_train
```

Optional GPU selection:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_zgated_difflr_gpu1
```

## Evaluate Checkpoints

Automatic validation only evaluates the final checkpoint. Manually evaluate intermediate checkpoints:

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-50.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-50.json

bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-200.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-200.json

bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-500.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-500.json
```

## Interpret Results

Good sign:

- Lowest or near-lowest `avg_aug_gap`.
- High number of instances beating baseline.
- Large instances such as `pr299`, `pr226`, and `kroA200` improve.
- Small instances such as `eil101`, `kroC100`, and `pr124` remain stable.

Do not automatically select this version just because it is the most complex. Select the checkpoint with the best validation tradeoff under the course constraints.
