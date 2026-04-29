# POMO_model: Model-Side Improvement Only

This directory is the model-side ablation. It keeps the original uniform TSP training setup and adds only a PolyNet-style decoder residual.

## Changes From Original POMO

- Adds `use_polynet`, `z_dim`, and `poly_embedding_dim` model parameters.
- Adds a z-conditioned decoder residual:

```text
mh_atten_out = mh_atten_out + MLP([mh_atten_out, z])
```

- Initializes the final residual layer to zero, so training starts close to the loaded POMO checkpoint.
- Loads old POMO checkpoints with `strict=False` when PolyNet mode is enabled.
- Keeps the original POMO all-start mechanism.

## What Is Not Changed

- Training data remains uniform random TSP.
- Training problem size remains fixed at `N=100` by default.
- No multi-scale curriculum.
- No mixed synthetic distributions.
- No 2-opt, LKH3, EAS, local search, beam expansion, or test-time iterative adaptation.

## Train

```bash
cd TSP/POMO_model
bash scripts/run_stage1_train.sh pomo_model_poly_stage1
```

Optional:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pomo_model_poly_stage1_gpu1
INIT_CHECKPOINT=./result/YOUR_POMO_CHECKPOINT/checkpoint-500.pt bash scripts/run_stage1_train.sh pomo_model_from_custom_pomo
```

## Evaluate A Checkpoint

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-200.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-200.json
```

The evaluation budget remains:

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

This variant isolates whether the z-conditioned residual helps without training-distribution changes.
