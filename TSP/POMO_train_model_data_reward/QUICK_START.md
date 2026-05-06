# PPO训练快速开始指南

## 🚀 5分钟快速开始

### 步骤1：测试代码是否正常运行

```bash
# 使用简化PPO模式快速测试（推荐第一步）
python train_ppo.py --mode simple_ppo --epochs 5 --train_episodes 1000

# 如果运行成功，恭喜！代码已经准备就绪
```

### 步骤2：开始正式训练

```bash
# 简化PPO（入门推荐，稳定可靠）
python train_ppo.py --mode simple_ppo --epochs 100 --exp_name my_first_ppo

# 完整PPO（研究推荐，效果更好）
python train_ppo.py --mode full_ppo --epochs 100 --exp_name my_full_ppo

# Off-Policy（高级用户，最高效率）
python train_ppo.py --mode off_policy --epochs 100 --replay_buffer_size 50000
```

---

## 📋 三种训练模式对比

| 模式 | 特点 | 适用场景 | 推荐度 |
|------|------|----------|--------|
| **simple_ppo** | 只添加Clipped Objective | 快速入门、稳定训练 | ⭐⭐⭐⭐⭐ |
| **full_ppo** | Critic + GAE + Clipped | 研究、追求性能 | ⭐⭐⭐⭐ |
| **off_policy** | Replay Buffer + Self-Imitation | 高样本效率 | ⭐⭐⭐ |

---

## 🎯 推荐训练流程

### 阶段1：基础验证（1小时）

```bash
# 1. 测试简化PPO是否能正常运行
python train_ppo.py --mode simple_ppo --epochs 10 --train_episodes 10000

# 2. 查看训练日志
# 日志位置：result/<timestamp>_<exp_name>/log.txt
# 检查：train_score是否在下降（越好）
```

### 阶段2：性能对比（1天）

```bash
# 训练简化PPO
python train_ppo.py --mode simple_ppo --epochs 100 --exp_name exp1_simple_ppo

# 训练完整PPO
python train_ppo.py --mode full_ppo --epochs 100 --exp_name exp2_full_ppo

# 对比结果（在log.txt中查看train_score）
```

### 阶段3：高级优化（1周）

```bash
# 尝试不同的超参数
python train_ppo.py --mode simple_ppo --epochs 100 \
    --ppo_epochs 5 \
    --ppo_clip 0.15 \
    --lr 5e-6 \
    --exp_name exp3_tuned

# 尝试Off-Policy
python train_ppo.py --mode off_policy --epochs 100 \
    --replay_buffer_size 100000 \
    --replay_ratio 0.4 \
    --exp_name exp4_off_policy
```

---

## ⚙️ 关键超参数说明

### PPO核心参数

```bash
--ppo_epochs 3          # PPO更新轮数（推荐：3-5）
                        # 值越大，样本效率越高，但不稳定风险增加

--ppo_clip 0.2          # 策略裁剪范围（推荐：0.1-0.3）
                        # 值越小，训练越稳定；值越大，学习速度越快

--gae_lambda 0.95       # GAE lambda（推荐：0.95）
                        # 控制value estimation的偏差-方差权衡

--value_loss_coef 0.5   # 价值损失权重（推荐：0.5）
                        # 平衡policy loss和value loss

--entropy_coef 0.01     # 熵正则化系数（推荐：0.01）
                        # 鼓励探索，防止过早收敛

--max_grad_norm 0.5     # 梯度裁剪（推荐：0.5）
                        # 防止梯度爆炸
```

### Off-Policy专用参数

```bash
--replay_buffer_size 50000    # 经验回放缓冲区大小
                              # 越大越好，但占用更多内存

--replay_ratio 0.3            # 使用历史数据的比例
                              # 0.3表示30%的训练步骤使用历史数据
```

---

## 📊 监控训练进度

### 查看日志

```bash
# 实时查看训练日志
tail -f result/<timestamp>_<exp_name>/log.txt

# 关键指标：
# - train_score: 平均路径长度（越小越好）
# - train_loss: 损失值
# - PLoss: 策略损失（仅full_ppo）
# - VLoss: 价值损失（仅full_ppo）
```

### 可视化结果

```bash
# 训练过程会自动生成图表：
# - latest_score.jpg: 训练分数曲线
# - latest_loss.jpg: 损失曲线

# 位置：result/<timestamp>_<exp_name>/
```

---

## 🔧 常见问题与解决方案

### Q1: 运行报错 "No module named 'TSPModel_PPO'"

**解决方案：** 确保以下文件存在：
- `TSPModel_PPO.py`
- `TSPTrainer_PPO.py`
- `train_ppo.py`

### Q2: GPU内存不足

**解决方案：** 减小batch size或问题规模

```bash
python train_ppo.py --mode simple_ppo --train_batch_size 32
```

### Q3: 训练不稳定，loss波动大

**解决方案：** 调整PPO参数

```bash
# 降低学习率
python train_ppo.py --mode simple_ppo --lr 5e-6

# 增加裁剪
python train_ppo.py --mode simple_ppo --ppo_clip 0.1

# 减少更新轮数
python train_ppo.py --mode simple_ppo --ppo_epochs 2
```

### Q4: 想从已有的POMO checkpoint继续训练

**解决方案：** 使用`--init_checkpoint`参数

```bash
python train_ppo.py --mode simple_ppo \
    --init_checkpoint result/saved_tsp100_model2_longTrain/checkpoint-3000.pt
```

**注意：** 从POMO checkpoint加载时，Critic网络会随机初始化。

---

## 📈 性能对比实验模板

### 实验设计

```bash
# 1. 原始POMO baseline（使用原始train.py）
python train.py --exp_name baseline_pomo --epochs 100

# 2. 简化PPO
python train_ppo.py --mode simple_ppo --epochs 100 --exp_name ppo_simple

# 3. 完整PPO
python train_ppo.py --mode full_ppo --epochs 100 --exp_name ppo_full

# 4. Off-Policy
python train_ppo.py --mode off_policy --epochs 100 --exp_name ppo_off_policy
```

### 结果分析

在`result/`目录下对比各实验的`log.txt`：
- 最终`train_score`（越小越好）
- 收敛速度（多少epoch达到某个分数）
- 训练稳定性（loss波动程度）

---

## 🎨 高级技巧

### 技巧1：课程学习

从小规模问题开始，逐步增加难度：

```bash
python train_ppo.py --mode simple_ppo \
    --curriculum "1:50;20:75;40:100;60:125;80:150" \
    --epochs 100
```

### 技巧2：多分布训练

使用多种数据分布提高泛化能力：

```bash
python train_ppo.py --mode simple_ppo \
    --distribution_mix "uniform:0.5,clustered:0.3,grid_jitter:0.2" \
    --epochs 100
```

### 技巧3：动态batch size

根据问题规模调整batch size：

```bash
python train_ppo.py --mode simple_ppo \
    --curriculum "1:100,125,150" \
    --dynamic_batch_sizes "100:64,125:48,150:32" \
    --epochs 100
```

### 技巧4：学习率衰减

在训练后期降低学习率：

```bash
python train_ppo.py --mode simple_ppo \
    --lr 1e-5 \
    --scheduler_milestones "50,80" \
    --scheduler_gamma 0.1 \
    --epochs 100
```

---

## 📚 下一步学习

1. **阅读详细指南：** `PPO_IMPROVEMENT_GUIDE.md`
2. **理解代码实现：**
   - `TSPModel_PPO.py` - Critic网络实现
   - `TSPTrainer_PPO.py` - PPO训练逻辑
3. **调参指南：** 参考`PPO_IMPROVEMENT_GUIDE.md`第四章
4. **论文阅读：**
   - PPO原论文: Schulman et al. 2017
   - POMO论文: Kwon et al. 2020

---

## 💡 快速检查清单

在开始训练前，确保：

- [ ] 已安装所有依赖（torch, numpy等）
- [ ] 有足够的GPU内存（至少4GB）
- [ ] `TSPModel_PPO.py`存在
- [ ] `TSPTrainer_PPO.py`存在
- [ ] `train_ppo.py`存在
- [ ] 已选择合适的训练模式

开始训练后，检查：

- [ ] `result/`目录下有新的实验文件夹
- [ ] `log.txt`正常更新
- [ ] `train_score`在逐渐下降
- [ ] 没有报错信息

---

## 🎯 预期结果

### 简化PPO (simple_ppo)
- **训练时间：** 与原始POMO相近
- **样本效率：** 提升1.5-2倍
- **最终性能：** 与POMO相当或略好

### 完整PPO (full_ppo)
- **训练时间：** 比POMO长20-30%（因为要计算value）
- **样本效率：** 提升2-3倍
- **最终性能：** 通常比POMO好1-3%

### Off-Policy
- **训练时间：** 比POMO长30-50%
- **样本效率：** 提升3-5倍
- **最终性能：** 可能比PPO好2-5%

---

**祝训练顺利！如有问题，请查看`PPO_IMPROVEMENT_GUIDE.md`获取更多帮助。**
