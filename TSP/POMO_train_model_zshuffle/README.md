# POMO_train_model_zshuffle: Training + Model + z-Shuffle

This directory is the final combined z-shuffle candidate:

```text
multi-scale / mixed-distribution training
+ PolyNet-style z-conditioned decoder residual
+ random z assignment during training
```

It answers the second ablation question:

```text
Q2: After training-side improvements are added, does random z assignment further
improve the combined model?
```

Compare this directory against `TSP/POMO_train_model`.

## Training-Side Changes

- Multi-scale curriculum over problem sizes.
- Mixed synthetic distributions for better TSPLIB-style generalization.
- Dynamic batch sizes.
- Fine-tuning from a POMO or `POMO_train` checkpoint.

## Model-Side Changes

- Adds a PolyNet-style latent policy code `z`.
- Adds a z-conditioned residual branch in the decoder:

```text
mh_atten_out = mh_atten_out + MLP([mh_atten_out, z])
```

- Keeps POMO all-start rollouts in stage 1.
- Initializes the residual branch so initial behavior is close to the loaded POMO checkpoint.

## z-Shuffle Change

During training only, each instance randomly permutes the assignment between rollout index and z code:

```text
rollout i starts from node i, but receives a randomly permuted z code.
```

During evaluation, z assignment remains deterministic:

```text
rollout i starts from node i and receives deterministic z_i.
```

This keeps evaluation reproducible and prevents the test-time candidate budget from increasing.

## Fairness Constraints

- `pomo_size = problem_size` during evaluation.
- `aug_factor = 8`.
- Total candidate tours are exactly `problem_size * 8`.
- No EAS, 2-opt, LKH3, local search, beam expansion, or test-time iterative adaptation.
- Public validation data is used only for evaluation and model selection.

## Train

```bash
cd TSP/POMO_train_model_zshuffle
bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_stage1
```

Recommended if a strong `POMO_train` checkpoint is available:

```bash
INIT_CHECKPOINT=../POMO_train/result/YOUR_BEST_TRAIN/checkpoint-500.pt \
bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_from_best_train
```

Optional GPU selection:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pomo_train_model_zshuffle_gpu1
```

## Evaluate Checkpoints

The script saves checkpoints every 50 epochs. Automatic validation only evaluates the final checkpoint, so evaluate intermediate checkpoints manually:

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

## Compare

```bash
bash scripts/compare_with_baseline.sh \
  ./result/YOUR_RUN/eval_val_checkpoint-500.json \
  ./result/YOUR_RUN/compare_with_baseline.md
```

Main comparisons:

```text
POMO_train_model vs POMO_train_model_zshuffle
POMO_model_zshuffle vs POMO_train_model_zshuffle
```

Good sign:

- `avg_aug_gap` improves over `POMO_train_model`.
- More instances beat baseline.
- Large instances such as `pr299`, `pr226`, and `kroA200` improve.
- Small instances such as `eil101`, `kroC100`, and `pr124` remain stable.

If no improvement appears:

- evaluate intermediate checkpoints, not only the final one;
- start from the best `POMO_train` checkpoint;
- reduce learning rate;
- compare with `POMO_model_zshuffle` to see whether z-shuffle helps only with richer training distributions.
