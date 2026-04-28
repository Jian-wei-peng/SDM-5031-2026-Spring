# PJW-model: Stage-1 Poly-POMO

This folder contains an isolated model-side experiment. It does not modify the original `TSP/POMO` files outside this directory.

The goal of this stage is to test a conservative PolyNet-style improvement on top of POMO:

```text
POMO all-start rollouts + z-conditioned decoder residual
```

The evaluation budget remains unchanged:

```text
candidate tours = problem_size * 8
```

No 2-opt, LKH3, EAS, local search, beam expansion, or post-processing is used.

## What Changed

Files copied and modified in this directory:

```text
PJW-model/
├── README.md
├── train.py
├── test.py
├── TSPModel.py
├── TSPTrainer.py
├── TSPTester_LIB.py
├── TSPEnv.py
├── TSProblemDef.py
├── tsplib_utils.py
├── compare_eval.py
└── scripts/
    ├── run_stage1_train.sh
    ├── eval_checkpoint.sh
    └── compare_with_baseline.sh
```

Main implementation points:

- `TSPModel.py` adds `use_polynet`, `z_dim`, and `poly_embedding_dim`.
- The decoder adds a PolyNet-style residual:

```text
mh_atten_out = mh_atten_out + MLP([mh_atten_out, z])
```

- `poly_layer_2` is zero-initialized, so the model initially behaves like the loaded POMO checkpoint.
- `TSPTrainer.py` loads old POMO checkpoints with `strict=False` when PolyNet is enabled.
- `TSPTrainer.py` and `TSPTester_LIB.py` generate deterministic binary `z` codes for rollouts.
- `TSPTester_LIB.py` keeps `pomo_size = problem_size`, so with `aug_factor=8` the total candidate count is exactly `problem_size * 8`.

## Why This May Help

POMO already generates multiple candidates by forcing different first nodes. However, all candidates still share the same decoder policy. PolyNet adds a learned policy code `z`, allowing rollouts to follow slightly different construction strategies.

This helps the official metric because the final solution is selected as the best candidate among rollouts and augmentations. Better diversity can reduce the chance that all candidates make the same structural mistake, especially on larger TSPLIB instances such as `pr299`, `pr226`, and `kroA200`.

## Server Setup

From the project root on the server:

```bash
cd /path/to/SDM-5031-2026-Spring/TSP/POMO/PJW-model
conda activate pytorch
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run Stage-1 Training

Recommended one-command run:

```bash
bash scripts/run_stage1_train.sh pjw_model_stage1_poly_pomo
```

Optional environment variables:

```bash
CUDA_DEVICE_NUM=1 bash scripts/run_stage1_train.sh pjw_model_stage1_poly_pomo_gpu1
INIT_CHECKPOINT=../result/YOUR_BEST_POMO/checkpoint-500.pt bash scripts/run_stage1_train.sh pjw_model_stage1_from_best_pomo
```

The default checkpoint is:

```text
../result/saved_tsp100_model2_longTrain/checkpoint-3000.pt
```

If you already have a better multi-scale/mixed-distribution POMO checkpoint, use it as `INIT_CHECKPOINT`. That is usually better than starting from the original baseline.

## Manual Training Command

Equivalent explicit command:

```bash
python train.py \
  --exp_name pjw_model_stage1_poly_pomo \
  --init_checkpoint ../result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --epochs 500 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --use_polynet true \
  --z_dim 16 \
  --poly_embedding_dim 256 \
  --force_first_move true \
  --curriculum "1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300" \
  --distribution_mix "uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05" \
  --dynamic_batch_sizes "100:64,125:48,150:40,200:24,250:16,300:12" \
  --load_optimizer false \
  --load_scheduler false \
  --reset_epoch true \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true
```

## Outputs

Each run writes to `PJW-model/result/`, for example:

```text
result/20260428_203000_pjw_model_stage1_poly_pomo/
├── config.json
├── log.txt
├── checkpoint-50.pt
├── checkpoint-100.pt
├── ...
├── checkpoint-500.pt
├── eval_val.json
└── src/
```

Important files to send back:

```text
config.json
log.txt
checkpoint-*.pt
eval_val.json
```

## Evaluate A Checkpoint

Training with `--eval_after_train true` automatically evaluates the final checkpoint. To evaluate an intermediate checkpoint:

```bash
bash scripts/eval_checkpoint.sh \
  ./result/YOUR_RUN/checkpoint-300.pt \
  ./result/YOUR_RUN/eval_val_checkpoint-300.json
```

This command uses:

```text
aug_factor = 8
pomo_size = problem_size
candidate tours = problem_size * 8
```

## Compare With Baseline

After evaluation:

```bash
bash scripts/compare_with_baseline.sh \
  ./result/YOUR_RUN/eval_val.json \
  ./result/YOUR_RUN/compare_with_baseline.md
```

Look at:

- `New avg_aug_gap` vs baseline `2.329568%`.
- `Better-than-baseline count`, target is at least `7/10`.
- Large instances: `pr299`, `pr226`, `kroA200`.
- Small/easy instances: `kroC100`, `eil101`, `pr124`; these should not regress heavily.

## How To Interpret Results

Good sign:

```text
avg_aug_gap decreases
pr299/pr226/kroA200 improve
>= 7/10 instances beat baseline
small 100-ish instances remain stable
```

Mixed sign:

```text
large instances improve but many small instances regress
```

Possible response:

- lower learning rate to `3e-6`;
- start from the best existing POMO fine-tuned checkpoint;
- train only the PolyNet layers for a short warm-up, then unfreeze all layers;
- increase the uniform proportion in `distribution_mix`.

Bad sign:

```text
avg_aug_gap increases and most instances regress
```

Possible response:

- reduce `lr`;
- reduce epochs;
- use `--distribution_mix "uniform:1.0"` for a first warm-up;
- verify the initialization checkpoint is correct;
- evaluate intermediate checkpoints instead of only the final one.

No obvious change:

```text
avg_aug_gap is close to POMO and candidate diversity does not help
```

Possible response:

- train longer;
- try a better initial checkpoint;
- increase `poly_embedding_dim` to `512`;
- later consider stage 2: `force_first_move=false` with `problem_size` z-conditioned rollouts, still under the same candidate budget.

## What Not To Use

Do not use these in this course project:

- EAS / Efficient Active Search;
- 2-opt, LKH3, or any local search;
- iterative test-time adaptation;
- more than `problem_size * 8` model-generated candidate tours;
- training directly on `TSP/data/val`.

The public validation set is only for model selection and reporting.
