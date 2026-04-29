# POMO Training Notes

`train.py` is the main training entrypoint. It includes the training-side improvements that were validated in the former PJW experiment folder:

- checkpoint-based fine-tuning from a baseline model;
- multi-scale curriculum over problem sizes;
- synthetic distribution mix for TSPLIB-style distribution shift;
- dynamic batch sizes for larger problem sizes;
- automatic config saving and optional validation after training.

The model architecture remains unchanged. The standard evaluation entrypoint is still `test.py`.

## Recommended Fine-Tuning

```bash
cd TSP/POMO

python train.py \
  --exp_name pomo_ft_multiscale_mixed_lr1e-5 \
  --init_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --epochs 500 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --curriculum "1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300" \
  --distribution_mix "uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05" \
  --dynamic_batch_sizes "100:64,125:48,150:40,200:24,250:16,300:12" \
  --load_optimizer false \
  --load_scheduler false \
  --reset_epoch true \
  --eval_after_train true
```

Training output is written under `result/`. Each run saves:

- `config.json`
- `log.txt`
- `checkpoint-*.pt`
- `eval_val.json` when `--eval_after_train true`
- `src/` source snapshot

## From-Scratch Training

To reproduce the original fixed-size POMO-style training path, disable checkpoint loading and use only uniform `N=100` training:

```bash
python train.py \
  --exp_name pomo_n100_uniform_from_scratch \
  --init_checkpoint "" \
  --epochs 3100 \
  --train_episodes 100000 \
  --lr 1e-4 \
  --curriculum "1:100" \
  --distribution_mix "uniform:1.0" \
  --dynamic_batch_sizes "100:64" \
  --eval_after_train false
```

## Validation And Comparison

Manual validation:

```bash
python test.py \
  --data_path ../data/val \
  --checkpoint_path ./result/YOUR_EXP/checkpoint-500.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log false \
  --output_json ./result/YOUR_EXP/eval_val_checkpoint-500.json
```

Compare against a baseline JSON:

```bash
python compare_eval.py \
  --baseline ./result_lib/baseline_eval_cpu.json \
  --new ./result/YOUR_EXP/eval_val.json \
  --output ./result/YOUR_EXP/compare_with_baseline.md
```

Use public validation only for evaluation, not for training. The training path does not use 2-opt, LKH, beam search expansion, local search, or any post-processing of generated tours.
