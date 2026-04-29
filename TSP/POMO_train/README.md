# POMO_train: Training-Side Improvement Only

This directory is the POMO training-side ablation. It keeps the POMO model architecture unchanged and improves only the training pipeline.

## Changes From Original POMO

- Multi-scale curriculum over problem sizes.
- Mixed synthetic training distributions.
- Dynamic batch sizes for larger problem sizes.
- Checkpoint-based fine-tuning from the official POMO baseline.
- Automatic validation and JSON export after training.
- Baseline comparison utility.

## What Is Not Changed

- No PolyNet decoder residual.
- No `z` policy code.
- No model-side architecture change.
- No 2-opt, LKH3, EAS, local search, beam expansion, or test-time iterative adaptation.
- Public validation data is used only for evaluation, not training.

## Train

```bash
cd TSP/POMO_train
bash scripts/run_train.sh pomo_train_multiscale_mixed
```

Optional:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_train.sh pomo_train_gpu1
INIT_CHECKPOINT=./result/YOUR_PREVIOUS_RUN/checkpoint-500.pt bash scripts/run_train.sh pomo_train_continue
```

## Evaluate A Checkpoint

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-500.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-500.json
```

The evaluation interface keeps:

```text
aug_factor = 8
pomo_size = problem_size
candidate tours = problem_size * 8
```

## Compare With Baseline

```bash
bash scripts/compare_with_baseline.sh \
  ./result/YOUR_RUN/eval_val.json \
  ./result/YOUR_RUN/compare_with_baseline.md
```

Main checks:

- `avg_aug_gap` should decrease from baseline.
- At least `7/10` validation instances should beat baseline.
- Large instances such as `pr299`, `pr226`, and `kroA200` should improve.
- Small instances should not regress heavily.
