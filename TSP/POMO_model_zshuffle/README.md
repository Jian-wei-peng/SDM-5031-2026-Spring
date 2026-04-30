# POMO_model_zshuffle: Model-Side z-Shuffle Ablation

This directory is a strict model-side ablation:

```text
original POMO training setup
+ PolyNet-style z-conditioned decoder residual
+ random z assignment during training
```

It answers the first ablation question:

```text
Q1: Does random z assignment make the z-conditioned residual more useful by
reducing the fixed binding between POMO start index and z code?
```

Compare this directory against `TSP/POMO_model`.

## Changes From `POMO_model`

- Adds `--shuffle_z_train`.
- During training, each instance randomly permutes the assignment between rollout index and z code.
- During evaluation, z assignment remains deterministic and reproducible.

Current training behavior:

```text
rollout i starts from node i, but receives a randomly permuted z code.
```

Current evaluation behavior:

```text
rollout i starts from node i and receives deterministic z_i.
```

## What Is Not Changed

- Training data remains uniform random TSP.
- Training problem size remains fixed at `N=100` by default.
- No multi-scale curriculum.
- No mixed synthetic distributions.
- No extra test-time z samples.
- No 2-opt, LKH3, EAS, local search, beam expansion, or test-time iterative adaptation.

## Fairness Constraints

Evaluation keeps:

```text
pomo_size = problem_size
aug_factor = 8
candidate tours = problem_size * 8
```

## Train

```bash
cd TSP/POMO_model_zshuffle
bash scripts/run_stage1_train.sh pomo_model_zshuffle_200
```

Optional:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pomo_model_zshuffle_gpu1
INIT_CHECKPOINT=./result/YOUR_POMO_CHECKPOINT/checkpoint-200.pt bash scripts/run_stage1_train.sh pomo_model_zshuffle_from_custom_pomo
```

Default training uses 200 epochs, because this variant is mainly for mechanism verification. If it improves over `POMO_model`, run a longer 500-epoch follow-up.

## Evaluate Checkpoints

The script saves checkpoints every 50 epochs. Automatic validation only evaluates the final checkpoint, so evaluate intermediate checkpoints manually:

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-50.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-50.json

bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-100.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-100.json

bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-200.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-200.json
```

## Compare

```bash
bash scripts/compare_with_baseline.sh \
  ./result/YOUR_RUN/eval_val_checkpoint-200.json \
  ./result/YOUR_RUN/compare_with_baseline.md
```

Main comparison:

```text
POMO_model vs POMO_model_zshuffle
```

Good sign:

- `avg_aug_gap` improves over `POMO_model`.
- More instances beat baseline.
- Large-instance gains do not come with severe small-instance regression.
