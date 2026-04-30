# SDM-5031-2026-Spring Project: DRL for TSP

本仓库用于 `SDM-5031-2026-Spring` 课程 Project，目标是使用深度强化学习解决 Traveling Salesman Problem (TSP)。
## Important Notes

- `TSP/data/val` 现在提供给同学的是公开验证集，用于开发、调参和本地对比。
- `TSP/data/val` 不是最终验收用的测试集。课程最终验收会使用课程组保留的隐藏测试集。
- 官方性能指标看的是 `avg_aug_gap`，也就是开启 test-time augmentation 后的平均 gap。

## Project Requirements

- `20%` Performance: 测试集上测试，并且在至少 `70%` 的实例上优于 baseline。
- `10%` Performance Rank: 按实验结果在全体同学中的平均排名计分。
- `10%` Method Novelty: 需要提交论文风格的 project report 和 presentation，说明方法设计、实验设置与创新点。

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── TSP/
│   ├── TSProblemDef.py           # original POMO problem generator used by TSP/POMO
│   ├── data/
│   │   └── val/                  # public validation set, NOT the final hidden test set
│   ├── POMO/                     # original POMO baseline code
│   ├── POMO_train/               # training-side improvement only
│   ├── POMO_model/               # model-side improvement only
│   ├── POMO_model_zshuffle/      # model-side + random z assignment ablation
│   ├── POMO_train_model/         # combined training + model improvements
│   ├── POMO_train_model_zshuffle/ # combined version + random z assignment
│   ├── POMO_train_model_zgated/   # combined version + gated z residual
│   ├── POMO_train_model_difflr/   # combined version + differential LR
│   └── POMO_train_model_zshuffle_zgated_difflr/ # final combined candidate
└── utils/
    ├── utils.py
    └── log_image_style/
```

## Project Variants

This repository keeps several TSP solver variants for ablation and final model selection. The full matrix is:

| Directory | Training-side improvement | z residual | z shuffle | z gate | diff LR | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `TSP/POMO` | No | No | No | No | No | Original POMO baseline. |
| `TSP/POMO_train` | Yes | No | No | No | No | Tests multi-scale curriculum, mixed synthetic distributions, dynamic batch sizes, and checkpoint fine-tuning. |
| `TSP/POMO_model` | No | Yes | No | No | No | Tests PolyNet-style z-conditioned decoder residual with original fixed-size uniform training. |
| `TSP/POMO_model_zshuffle` | No | Yes | Yes | No | No | Tests whether random z assignment improves the z residual itself. |
| `TSP/POMO_train_model` | Yes | Yes | No | No | No | Tests the combination of training-side improvements and z residual. |
| `TSP/POMO_train_model_zshuffle` | Yes | Yes | Yes | No | No | Tests whether z-shuffle further improves the combined model. |
| `TSP/POMO_train_model_zgated` | Yes | Yes | No | Yes | No | Tests whether a conservative gate stabilizes the z residual. |
| `TSP/POMO_train_model_difflr` | Yes | Yes | No | No | Yes | Tests whether differential LR protects the backbone while training new z modules. |
| `TSP/POMO_train_model_zshuffle_zgated_difflr` | Yes | Yes | Yes | Yes | Yes | Final combined candidate. |

The two z-shuffle ablations answer two separate questions:

```text
Q1: Does random z assignment make the z-conditioned residual more useful?
    Compare POMO_model vs POMO_model_zshuffle.

Q2: After training-side improvements are added, does random z assignment further improve the combined model?
    Compare POMO_train_model vs POMO_train_model_zshuffle.
```

In z-shuffle variants, training randomly permutes the assignment between rollout index and z code. Evaluation keeps deterministic z assignment, so the official candidate budget does not increase.

The stability-oriented ablations are evaluated only on the combined training setting:

```text
Q3: Does a gated z residual reduce fine-tuning instability or small-instance regression?
    Compare POMO_train_model vs POMO_train_model_zgated.

Q4: Does differential LR protect the original POMO backbone while training the new z module?
    Compare POMO_train_model vs POMO_train_model_difflr.

Q5: Do z-shuffle, z-gate, and differential LR combine into the strongest final variant?
    Compare POMO_train_model_zshuffle_zgated_difflr against all combined variants.
```

The `POMO_model` line is intentionally kept focused on model-only z residual and z-shuffle mechanism checks. The gated residual and differential-LR variants are placed on the `POMO_train_model` line because they target final fine-tuning stability under multi-scale mixed-distribution training.

All variants keep the standardized evaluation interface:

```bash
python test.py \
  --data_path ../data/val \
  --checkpoint_path /path/to/checkpoint.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --output_json /path/to/eval.json
```

For the course constraint, all provided evaluation scripts keep:

```text
pomo_size = problem_size
aug_factor = 8
candidate tours = problem_size * 8
```

No variant uses 2-opt, LKH3, EAS, local search, beam expansion, or test-time iterative adaptation.

## Environment and Requirements

推荐使用 `Python 3.10` 或 `Python 3.11`。项目依赖见 [requirements.txt](requirements.txt)。

建议安装步骤如下：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`torch` 建议根据你的 CUDA 版本单独安装对应的官方 wheel：

```bash
https://pytorch.org/get-started/locally/
```

## Data and Baseline

- 公开验证集目录：`TSP/data/val`
- 最终测试集：课程组保留，不公开
- baseline 模型：`TSP/POMO/result/saved_tsp100_model2_longTrain/checkpoint-3000.pt`
- 训练脚本：`TSP/POMO/train.py`
- 统一评测脚本：`TSP/POMO/test.py`

当前代码默认读取 TSPLIB 格式实例，并输出：

- `no_aug_score`
- `aug_score`
- `no_aug_gap`
- `aug_gap`

请注意：

- 本地调参和公开结果对比时，应以 `avg_aug_gap` 为主。
- 最终隐藏测试集验收时，课程组也会以 augmentation 后的结果作为主指标。

## Standardized Submission Interface

为了方便同学提交代码后由课程组直接切换到隐藏测试集统一验收，请保留 `TSP/POMO/test.py` 的命令行接口。课程组会使用类似下面的命令进行评测：

```bash
cd TSP/POMO
python test.py \
  --data_path /path/to/hidden_test_set \
  --checkpoint_path /path/to/checkpoint.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log false \
  --output_json /path/to/eval_result.json
```

其中：

- `--data_path` 指向课程组的隐藏测试集目录。
- `--checkpoint_path` 指向待评测模型的 checkpoint 文件。
- `--output_json` 会导出机器可读的评测结果，方便课程组统一收集。

## Quick Start

### 1. 复现 baseline 在公开验证集上的结果

```bash
cd TSP/POMO
python test.py \
  --data_path ../data/val \
  --checkpoint_path ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --augmentation_enable true \
  --aug_factor 8
```

请使用输出中的 `avg_aug_gap` 作为 baseline 比较指标，而不是 `avg_no_aug_gap`。

### 2. 训练你自己的模型

```bash
cd TSP/POMO
python train.py
```

当前训练脚本的默认设置为：

- `problem_size = 100`
- `epochs = 3100`
- `train_episodes = 100000`
- `train_batch_size = 64`

训练过程会在 `TSP/POMO/result/` 下自动创建实验目录，并保存：

- checkpoint
- 日志
- 训练曲线图
- 源码快照

### 3. 用你自己的 checkpoint 在公开验证集上验证

```bash
cd TSP/POMO
python test.py \
  --data_path ../data/val \
  --checkpoint_path /path/to/your/checkpoint.pt \
  --augmentation_enable true \
  --aug_factor 8 \
  --output_json ./result_lib/your_eval.json
```

## Suggested Project Workflow

1. 先运行 baseline，记录公开验证集上的 `avg_aug_gap` 和逐实例 `aug_gap`。
2. 修改模型或训练策略，并保存新的 checkpoint。
3. 用同一份公开验证集重新评测，比较 `avg_aug_gap`，并统计在多少个实例上优于 baseline。

## Deliverables

建议最终提交内容包括：

- 代码仓库 （在README.md中包含运行脚本命令）
- 最优 checkpoint
- project report
- presentation slides
- 基于公开验证集的 baseline 对比结果

## Acknowledgement

本项目实现基于 POMO 思路展开，并针对 `SDM-5031-2026-Spring` 课程项目做了训练、验证和统一评测接口上的适配。
