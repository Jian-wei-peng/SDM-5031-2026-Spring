> 任务：
>
> - 在保持 TSP/POMO/test.py 标准评测接口不变的前提下，提交一个比现有 POMO baseline 更强的 DRL-based TSP solver
> - 官方主指标是 avg_aug_gap
> - 隐藏测试集完成最终结果测试，公开的 TSP/data/val 只用于开发
>
> 
>
> 旅行商问题 (Traveling Salesperson Problem, 简称 TSP) 是运筹学和理论计算机科学中最经典的组合优化问题之一
>
> - 定义：给定一系列城市（或节点）以及每对城市之间的距离，TSP 要求找出一条最短的路线，使得旅行商能够访问每个城市恰好一次，并最终返回出发点
> - 复杂性：TSP 是一个著名的 NP-Hard 问题。随着城市数量的增加，可能的路线排列组合呈阶乘级增长。例如，10个城市有约 36万条可能路线，而 50 个城市的路线总数则是一个天文数字，导致传统的精确搜索算法在节点增加时计算量爆炸
>
> 
>
> 传统方法（如精确求解器 Concorde）在规模变大时极其耗时，而启发式算法需要大量人工设计的规则
>
> DRL 是让神经网络通过试错学习出一种构建路径的策略，求解过程通常被建模为一个马尔可夫决策过程：
>
> - 状态 (State)：当前所在的城市、所有城市的几何坐标图，以及目前为止已经访问过的城市序列
> - 动作 (Action)：在当前城市，选择下一个要访问的未去过的城市
> - 奖励 (Reward)：由于目标是寻找最短路径，奖励通常设置为最终生成的闭环路径总长度的负值（RL 会最大化奖励，从而最小化距离）
>
> 
>
> POMO (**P**olicy **O**ptimization with **M**ultiple **O**ptima) 的核心突破在于巧妙利用了 TSP 问题的“对称性”：
>
> **一个最优的 TSP 闭环路径，无论从哪个城市作为起点出发，其最终的总距离是不变的**
>
> - **传统训练的局限**：在标准 RL 训练中，智能体面对一张图通常只从一个随机起点开始构建一条路径，这导致梯度的方差很大，模型容易陷入局部最优
> - **POMO 的创新**：
>   - 训练阶段，POMO 强制策略网络对于同一个 TSP 图实例，同时从 $N$ 个不同的起点（例如图中的所有城市）开始生成 $N$ 条完整的轨迹
>     - 优势：通过比较这 $N$ 条轨迹的长度，模型可以利用它们相互作为基线来更精准地计算 Advantage（优势函数）。这种共享基线的方法极大地稳定了训练过程，让模型更容易探索到全局更优解
>   - 推断阶段，POMO 通常也会结合数据增强技术（Augmented Evaluation，例如将整张图旋转、翻转生成 $M$ 个等价变体，对应的就是文档中的 `aug_factor 8` 参数），生成多个解并取其中的最好结果，以此进一步提升最终表现 
> - 沿用了主流的基于 Transformer 的架构来处理 TSP 的几何特征：
>   - Encoder：使用多层自注意力机制（Self-Attention）。将每个城市的坐标 $(x, y)$ 映射为高维特征向量。编码器会捕获节点之间的相对位置关系和图的全局结构信息。
>   - Decoder：是一个自回归（Autoregressive）过程。在每一步：
>     1. 利用交叉注意力机制（Cross-Attention）结合编码器的全局信息和当前所在城市的信息
>     2. 使用 Mask（掩码）遮盖掉已经访问过的城市 
>     3. 输出剩余未访问城市的概率分布
>     4. 按照概率（训练时）或贪心/采样（测试时）选择下一个城市，直到所有城市访问完毕



## 1. Overview

**公开验证集：**

- 规模是 100 到 299，全部是 TSPLIB EUC_2D 实例

```txt
TSP/data/val/
  kroA100, kroC100, kroE100
  eil101
  pr124
  ch150, kroB150
  kroA200
  pr226
  pr299
```

**Baseline checkpoint:** 

```txt
TSP/POMO/result/saved_tsp100_model2_longTrain/checkpoint-3000.pt
```

**官方评测入口:**
```txt
cd TSP/POMO
python test.py \
  --data_path ../data/val \
  --checkpoint_path ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --augmentation_enable true \
  --aug_factor 8
```

**代码结构 (标准的 POMO):**

```txt
├── TSP/
│   ├── TSProblemDef.py  			训练时用 get_random_problems() 生成随机 TSP 实例
│		│									   			测试时用 augment_xy_data_by_8_fold() 做 8 倍坐标增强
│   └── POMO/
│       ├── train.py					训练入口。默认训练 TSP100
│       ├── test.py						官方标准评测入口，需保持 CLI 接口不变
│       ├── TSPTrainer.py			训练主逻辑：
│			  │											创建环境、模型；加载checkpoint；执行rollout；计算 POMO policy gradient loss
│		    │											保存 checkpoint 和训练曲线 
│       ├── TSPTester_LIB.py	TSPLIB 评测主逻辑
│		    │											读取 TSPLIB，归一化坐标，8-fold augmentation
│		    │											取所有 POMO start 和 augmentation 里的最短 tour
│       ├── TSPModel.py				POMO 神经网络模型
│       ├── TSPEnv.py					TSP 环境
│       ├── tsplib_utils.py		TSPLIB 工具
```



## 2. Baseline 权重跑公开验证集

**公开验证集：`TSP/data/val `**

**步骤：**

```bash
# conda 环境
conda activate pytorch

# 进入评测脚本所在目录
cd TSP/POMO

# 运行 baseline
# CPU
python test.py \
  --data_path ../data/val \
  --checkpoint_path ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda false \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log true \
  --output_json ./result_lib/baseline_eval_cpu.json

# GPU
python test.py \
  --data_path ../data/val \
  --checkpoint_path ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
  --use_cuda true \
  --cuda_device_num 0 \
  --augmentation_enable true \
  --aug_factor 8 \
  --detailed_log true \
  --output_json ./result_lib/baseline_eval_gpu.json
```

会生成一个 JSON 文件：

- 这个 JSON 文件就是之后和自己模型做对比的 baseline 结果

```txt
TSP/POMO/result_lib/baseline_eval_cpu.json
```

还会自动生成一个日志目录，例如

```txt
TSP/POMO/result_lib/20260425_161200_aug8_test_TSPLIB_POMO/
├── run_log.txt		完整日志
└── src/					本次运行的源码快照，用来保证实验可复现
```



### 2.1 结果分析

**Baseline 公开验证集结果：**

总体结果：

```bash
avg_no_aug_gap = 3.754% 
avg_aug_gap    = 2.330% 
```

逐实例 aug_gap：

| instance | size | aug_gap | 诊断             |
| :------- | :--- | :------ | :--------------- |
| pr299    | 299  | 13.984% | 最大短板         |
| pr226    | 226  | 4.116%  | 明显偏弱         |
| kroA200  | 200  | 2.125%  | 中大规模泛化不足 |
| kroB150  | 150  | 1.052%  | 中等退化         |
| kroA100  | 100  | 0.648%  | 尚可             |
| kroE100  | 100  | 0.494%  | 尚可             |
| ch150    | 150  | 0.398%  | 很好             |
| kroC100  | 100  | 0.241%  | 很好             |
| eil101   | 101  | 0.159%  | 很好             |
| pr124    | 124  | 0.078%  | 很好             |

- 最关键的一点：
  - pr299 一个实例贡献了总 aug_gap 的约 **60%**
  - 如果只看 pr299 + pr226 + kroA200，这三个实例贡献了总误差的约 **87%**
  - 所以 baseline 的主要问题不是整体都差，而是**大规模和特定 TSPLIB 分布上崩得比较明显**



#### （1） 缺陷 1：固定 TSP100 训练，规模泛化不足

当前 train.py 里写死：

```
problem_size = 100 
pomo_size = 100 
```

但公开验证集包含：

```
100, 101, 124, 150, 200, 226, 299 
```

模型结构虽然可以处理不同节点数，但策略是在 TSP100 上学出来的。到了 200、226、299 点，局部密度、tour 长度、attention 范围、mask rollout 长度都变了，所以性能明显下降

对应现象：

```
pr299   13.984% 
pr226   4.116% 
kroA200 2.125% 
```

解决方案：

**多规模 curriculum fine-tuning **

- 推荐训练规模：

  ```txt
  Stage 1: N = 100 
  Stage 2: N ∈ {100, 125, 150} 
  Stage 3: N ∈ {100, 125, 150, 200} 
  Stage 4: N ∈ {100, 125, 150, 200, 250, 300} 
  ```

- 目标是让模型逐渐适应更长的 tour construction，而不是突然从 100 跳到 300



#### （2） 缺陷 2：训练分布过单一，只见 uniform random

当前 TSProblemDef.py 只生成：

```
torch.rand(batch_size, problem_size, 2) 
```

这意味着训练集全部是 [0,1]^2 上均匀随机点

但 TSPLIB 实例不是纯 uniform random。尤其是 pr 系列，坐标分布更结构化；eil、kro、ch、pr 的空间形态也不同。baseline 在 pr124 很好，但在 pr226/pr299 很差，说明问题不只是 pr 家族，而是**大规模 + 结构化分布叠加**后泛化变差。

解决方案：

**混合 synthetic distributions **

- 可以训练时随机生成：

  ```txt
  Uniform random
  Clustered Gaussian
  Anisotropic rectangle
  Grid jitter
  Ring / circle-like
  Line-biased / diagonal-biased
  Mixed-density regions
  Integer-coordinate TSPLIB-like
  ```

- 不要直接用公开 val 训练，公开 val 只能评估。应该用 synthetic 数据扩大覆盖面。



#### （3）缺陷 3：baseline checkpoint 已经偏向 TSP100 optimum behavior

baseline 在 100 附近表现不错：

```
kroC100 0.241% 
eil101  0.159% 
pr124   0.078% 
```

所以不能粗暴大学习率继续训练，否则可能出现：**大规模变好，小规模退化 **

解决方案：

**从 baseline 小学习率 fine-tune **

- 推荐：

  ```txt
  init_checkpoint = checkpoint-3000.pt
  lr = 1e-5 或 3e-5
  保存多个 checkpoint
  每个 checkpoint 都跑 val
  ```

- 不要从零训练作为第一方案。从零训练成本高，而且容易丢掉 baseline 已学到的 TSP100 能力





最稳的方向是：

1. 从 baseline checkpoint 继续 fine-tune。
2. 把训练规模从固定 N=100 改成多规模采样，例如 100/125/150/200/250/300。
3. 改训练数据分布，让模型见到更像 TSPLIB 的点集，而不是只见 uniform random。
4. 保持 test.py 接口不变，仍然只用 aug_factor=8 和 POMO 原生 N 个起点。
5. 每次新 checkpoint 都和这份 baseline JSON 做逐实例对比，目标是降低 avg_aug_gap，并在至少 70% 实例上优于 baseline。





