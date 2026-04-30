# POMO_train_model_zgated: Gated z-Residual Ablation

This directory is based on `TSP/POMO_train_model` and changes only the z-conditioned decoder residual into a gated residual.

It answers:

```text
Does a conservative gate make the z-conditioned residual more stable during multi-scale fine-tuning?
```

Compare against:

```text
TSP/POMO_train_model
```

## Changes

The original z residual is:

```text
mh_atten_out = mh_atten_out + residual([mh_atten_out, z])
```

This variant uses:

```text
gate = sigmoid(MLP_gate([mh_atten_out, z]))
mh_atten_out = mh_atten_out + gate * residual([mh_atten_out, z])
```

Risk control:

```text
poly_layer_2 is zero-initialized.
poly_gate_2.weight is zero-initialized.
poly_gate_2.bias is initialized to -4.0, so sigmoid(-4) is about 0.018.
```

This means training starts close to the loaded POMO behavior and learns to enable the z residual gradually.

## What Is Not Changed

- Keeps multi-scale curriculum and mixed synthetic distributions from `POMO_train_model`.
- Keeps deterministic z assignment during training and evaluation.
- Does not use differential learning rate.
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
cd TSP/POMO_train_model_zgated
bash scripts/run_stage1_train.sh pomo_train_model_zgated
```

Recommended if a strong `POMO_train` checkpoint is available:

```bash
INIT_CHECKPOINT=../POMO_train/result/YOUR_BEST_TRAIN/checkpoint-500.pt \
bash scripts/run_stage1_train.sh pomo_train_model_zgated_from_best_train
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
- Similar or better large-instance performance.
- Less small-instance regression than the ungated residual.
