# PJW Training Experiments

这个目录用于训练侧改进实验。原始课程代码不动，所有 PJW 版本的训练改动都放在这里。

核心约束：

- 评测仍然使用 `../test.py` 的课程标准接口。
- 测试时只使用 POMO 原生 `problem_size` 个起点和 `aug_factor=8`。
- 不使用 2-opt、LKH3、beam search 扩候选、局部搜索或任何模型生成解后的二次优化。
- `../../data/val` 只用于评估，不用于训练。

## 文件说明

```text
PJW/
├── README.md
├── train.py
├── TSPTrainer.py
├── TSPEnv.py
├── TSPModel.py
├── TSProblemDef.py
└── compare_eval.py
```

- `train.py`: PJW 训练入口，支持命令行配置、多规模 curriculum、mixed synthetic distributions、从 baseline checkpoint fine-tune，并可在训练结束后自动调用课程 `test.py` 跑公开验证集。
- `TSPTrainer.py`: PJW trainer，支持每个 batch 随机采样 problem size 和 synthetic distribution。
- `TSPEnv.py`: 从原始 `../TSPEnv.py` 复制而来，训练逻辑保持一致，只补了 CUDA 设备安全处理。
- `TSPModel.py`: 从原始 `../TSPModel.py` 复制而来，模型结构和 checkpoint key 不变，只补了 CUDA 设备安全处理。
- `TSProblemDef.py`: PJW synthetic TSP 数据生成器，保留原始 `uniform`，新增 `clustered`、`anisotropic`、`grid_jitter`、`ring`、`line_biased`、`mixed_density`、`integer`。
- `compare_eval.py`: 对比 baseline eval JSON 和新实验 eval JSON，输出逐实例是否优于 baseline。

## 服务器环境

进入服务器项目目录后：

```bash
cd /path/to/SDM-5031-2026-Spring/TSP/POMO/PJW
conda activate pytorch
```

确认 CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 推荐实验 1：只做 N=100 fine-tune

目的：确认从 baseline checkpoint 继续训练不会把模型训崩。

```bash
python train.py \
  --exp_name pjw_ft_n100_uniform_lr1e-5 \
  --init_checkpoint ../result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --epochs 100 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --curriculum "1:100" \
  --distribution_mix "uniform:1.0" \
  --dynamic_batch_sizes "100:64" \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true
```

## 推荐实验 2：多规模 uniform fine-tune

目的：优先改善 `pr299`、`pr226`、`kroA200` 这类大规模实例。

```bash
python train.py \
  --exp_name pjw_ft_multiscale_uniform_lr1e-5 \
  --init_checkpoint ../result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --epochs 500 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --curriculum "1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300" \
  --distribution_mix "uniform:1.0" \
  --dynamic_batch_sizes "100:64,125:48,150:40,200:24,250:16,300:12" \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true
```

## 推荐实验 3：多规模 + mixed distribution fine-tune

目的：改善 TSPLIB 风格分布偏移，同时保留较高比例的 uniform，避免破坏 baseline 已学到的 TSP100 能力。

```bash
python train.py \
  --exp_name pjw_ft_multiscale_mixed_lr1e-5 \
  --init_checkpoint ../result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --epochs 500 \
  --train_episodes 100000 \
  --lr 1e-5 \
  --curriculum "1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300" \
  --distribution_mix "uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05" \
  --dynamic_batch_sizes "100:64,125:48,150:40,200:24,250:16,300:12" \
  --model_save_interval 50 \
  --img_save_interval 50 \
  --eval_after_train true
```

## 输出目录

每次运行会在 `PJW/result/` 下生成实验目录，例如：

```text
PJW/result/20260425_153000_pjw_ft_multiscale_mixed_lr1e-5/
├── config.json
├── log.txt
├── checkpoint-50.pt
├── checkpoint-100.pt
├── ...
├── checkpoint-500.pt
├── eval_val.json
└── src/
```

其中：

- `config.json`: 本次实验参数。
- `log.txt`: 训练日志。
- `checkpoint-*.pt`: 保存的模型权重。
- `eval_val.json`: 训练结束后自动跑公开验证集得到的结果。
- `src/`: 本次运行源码快照。

## 手动评测某个 checkpoint

如果想评测中间 checkpoint，不要改 `test.py`，直接调用课程标准评测入口：

```bash
cd /path/to/SDM-5031-2026-Spring/TSP/POMO

python test.py \
  --data_path ../data/val \
  --checkpoint_path ./PJW/result/YOUR_EXP/checkpoint-300.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log false \
  --output_json ./PJW/result/YOUR_EXP/eval_val_checkpoint-300.json
```

## 和 baseline 对比

在 `PJW/` 目录下运行：

```bash
python compare_eval.py \
  --baseline ../result_lib/baseline_eval_cpu.json \
  --new ./result/YOUR_EXP/eval_val.json \
  --output ./result/YOUR_EXP/compare_with_baseline.md
```

重点看：

- `New avg_aug_gap` 是否低于 baseline 的 `2.329568%`。
- `Better-than-baseline count` 是否至少 `7/10`。
- `pr299`、`pr226`、`kroA200` 是否明显改善。
- `kroC100`、`eil101`、`pr124` 是否没有明显退化。

## 建议回传给负责分析的文件

每个服务器实验至少回传：

```text
config.json
log.txt
checkpoint-*.pt
eval_val.json
compare_with_baseline.md
```

最终提交 checkpoint 时，仍然用课程原始命令评测：

```bash
cd TSP/POMO
python test.py \
  --data_path /path/to/hidden_test_set \
  --checkpoint_path /path/to/your/checkpoint.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log false \
  --output_json /path/to/eval_result.json
```
