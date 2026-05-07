# POMO_train_model: Training + Model Improvements

This directory combines both improvement lines:

```text
multi-scale / mixed-distribution training
+ PolyNet-style z-conditioned decoder residual
```

It is the candidate strongest variant.

## Training-Side Changes

- Multi-scale curriculum over problem sizes.
- Mixed synthetic distributions for better TSPLIB-style generalization.
- Dynamic batch sizes.
- Fine-tuning from a POMO checkpoint.

## Model-Side Changes

- Adds a PolyNet-style latent policy code `z`.
- Adds a z-conditioned residual branch in the decoder:

```text
mh_atten_out = mh_atten_out + MLP([mh_atten_out, z])
```

- Keeps POMO all-start rollouts in stage 1.
- Initializes the residual branch so the initial behavior is close to the loaded POMO checkpoint.

## Fairness Constraints

- `pomo_size = problem_size` during evaluation.
- `aug_factor = 8`.
- Total candidate tours are exactly `problem_size * 8`.
- No EAS, 2-opt, LKH3, local search, beam expansion, or test-time iterative adaptation.
- Public validation data is used only for evaluation and model selection.

## Train

```bash
cd TSP/POMO_train_model
bash scripts/run_stage1_train.sh pomo_train_model_stage1
```

Optional:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pomo_train_model_gpu1
INIT_CHECKPOINT=../POMO_train/result/YOUR_BEST_TRAIN/checkpoint-500.pt bash scripts/run_stage1_train.sh pomo_train_model_from_best_train
```

Starting from the best `POMO_train` checkpoint is recommended because this variant then only needs to learn the additional model-side residual on top of an already improved training distribution.

## Evaluate A Checkpoint

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-500.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-500.json
```

## Compare With Baseline

```bash
bash scripts/compare_with_baseline.sh \
  ./result/YOUR_RUN/eval_val.json \
  ./result/YOUR_RUN/compare_with_baseline.md
```

## How To Interpret Results

Good sign:

- `avg_aug_gap` decreases.
- At least `7/10` validation instances beat baseline.
- Large instances such as `pr299`, `pr226`, and `kroA200` improve.
- Small instances such as `eil101`, `kroC100`, and `pr124` remain stable.

If large instances improve but small instances regress:

- lower learning rate;
- start from the best `POMO_train` checkpoint;
- increase the uniform ratio in `distribution_mix`;
- evaluate intermediate checkpoints, not just the final one.

If no improvement appears:

- train longer;
- try `poly_embedding_dim=512`;
- use a stronger initialization checkpoint;
- later consider stage 2 where z-conditioned rollouts replace part of the forced-first-move diversity while staying under the same candidate budget.
