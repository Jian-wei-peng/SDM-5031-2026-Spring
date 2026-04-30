# POMO_train_model_difflr: Differential-LR Ablation

This directory is based on `TSP/POMO_train_model` and changes only the fine-tuning optimizer.

It answers:

```text
Does differential learning rate protect the POMO backbone while allowing the new z residual module to learn?
```

Compare against:

```text
TSP/POMO_train_model
```

## Changes

The optimizer uses two parameter groups:

```text
backbone parameters: lr = 1e-6
poly_layer / poly_gate parameters: lr = 1e-5
```

This variant has no gate by default, so `poly_gate` parameters are absent and only `poly_layer` parameters use the larger LR.

Risk control:

```text
--load_optimizer false
--load_scheduler false
--reset_epoch true
```

Old optimizer state is not loaded because the parameter-group structure is different from the original POMO optimizer.

The trainer logs:

```text
base parameter count
poly parameter count
backbone_lr
poly_lr
```

so the parameter grouping can be checked from `log.txt`.

## What Is Not Changed

- Keeps multi-scale curriculum and mixed synthetic distributions from `POMO_train_model`.
- Keeps the original ungated z residual.
- Keeps deterministic z assignment during training and evaluation.
- Does not add extra test-time samples.
- Does not use 2-opt, LKH3, EAS, local search, beam expansion, or test-time iterative adaptation.

## Fairness Constraints

Evaluation keeps:

```text
pomo_size = problem_size
aug_factor = 8
candidate tours = problem_size * 8
```

## Train

```bash
cd TSP/POMO_train_model_difflr
bash scripts/run_stage1_train.sh pomo_train_model_difflr
```

Recommended if a strong `POMO_train` checkpoint is available:

```bash
INIT_CHECKPOINT=../POMO_train/result/YOUR_BEST_TRAIN/checkpoint-500.pt \
bash scripts/run_stage1_train.sh pomo_train_model_difflr_from_best_train
```

## Evaluate Checkpoints

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-500.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-500.json
```

Evaluate intermediate checkpoints as well:

```text
50, 100, 150, 200, 300, 400, 500
```

## Interpret Results

Good sign:

- Lower `avg_aug_gap` than `POMO_train_model`.
- Small near-100 instances remain stable.
- Large-instance gains from the combined model are not lost.
