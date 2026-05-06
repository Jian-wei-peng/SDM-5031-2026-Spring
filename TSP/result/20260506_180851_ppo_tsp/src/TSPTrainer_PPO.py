"""
TSP Trainer with PPO Support - 支持PPO和Off-Policy训练

提供三种训练模式：
1. simple_ppo: 简化PPO（只添加Clipped Objective，推荐入门）
2. full_ppo: 完整PPO（Critic + GAE + Clipped Objective）
3. off_policy: Off-Policy训练（带Replay Buffer）

使用方法：
    from TSPTrainer_PPO import TSPTrainerPPO as Trainer
"""

import random
from logging import getLogger
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from TSPEnv import TSPEnv as Env
from TSPModel_PPO import TSPModel_PPO as Model
from TSProblemDef import get_random_problems
from utils.utils import *


# =============================================================================
# Replay Buffer for Off-Policy Training
# =============================================================================

class TSPReplayBuffer:
    """
    TSP专用的经验回放缓冲区
    
    与标准RL的replay buffer不同，TSP是构造性问题：
    - 存储"问题实例 + 完整解轨迹 + 奖励"
    - 而非单个(state, action, reward, next_state)转移
    
    用途：
    1. Behavior Cloning: 让当前策略学习历史最优解
    2. Importance Sampling: 用旧策略的数据更新当前策略
    3. Self-Imitation Learning: 学习自己过去的好解
    """

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, problem, trajectory, reward, problem_size):
        """
        存储一个完整的TSP实例和解
        
        Args:
            problem: (problem_size, 2) - TSP实例坐标
            trajectory: (problem_size,) - 节点访问顺序
            reward: float - 解的质量（负的路径长度）
            problem_size: int - 问题规模
        """
        self.buffer.append({
            'problem': problem.cpu(),
            'trajectory': trajectory.cpu(),
            'reward': reward,
            'problem_size': problem_size,
        })

    def sample(self, batch_size, problem_size=None):
        """
        采样一批历史数据
        
        Args:
            batch_size: 采样数量
            problem_size: 可选，只采样特定规模的问题
        
        Returns:
            problems, trajectories, rewards
        """
        if problem_size is not None:
            # 只从特定规模的问题中采样
            candidates = [item for item in self.buffer 
                         if item['problem_size'] == problem_size]
        else:
            candidates = list(self.buffer)
        
        if not candidates:
            return None, None, None
        
        batch = random.sample(candidates, min(batch_size, len(candidates)))
        
        problems = torch.stack([item['problem'] for item in batch])
        trajectories = torch.stack([item['trajectory'] for item in batch])
        rewards = torch.tensor([item['reward'] for item in batch])
        
        return problems, trajectories, rewards

    def sample_best(self, batch_size, problem_size=None):
        """采样奖励最高的历史数据（Self-Imitation Learning）"""
        if problem_size is not None:
            candidates = [item for item in self.buffer 
                         if item['problem_size'] == problem_size]
        else:
            candidates = list(self.buffer)
        
        if not candidates:
            return None, None, None
        
        # 按奖励排序，取最好的
        candidates.sort(key=lambda x: x['reward'], reverse=True)
        batch = candidates[:min(batch_size, len(candidates))]
        
        problems = torch.stack([item['problem'] for item in batch])
        trajectories = torch.stack([item['trajectory'] for item in batch])
        rewards = torch.tensor([item['reward'] for item in batch])
        
        return problems, trajectories, rewards

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# PPO Trainer
# =============================================================================

class TSPTrainerPPO:
    """
    支持PPO训练的TSP训练器
    
    训练模式：
    - 'simple_ppo': 简化PPO（Clipped Objective + 多轮更新）
    - 'full_ppo': 完整PPO（Critic + GAE + Clipped Objective）
    - 'off_policy': Off-Policy训练（带Replay Buffer）
    """

    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # PPO超参数
        ppo_config = trainer_params.get('ppo_config', {})
        self.ppo_mode = ppo_config.get('mode', 'simple_ppo')  # 'simple_ppo', 'full_ppo', 'off_policy'
        self.ppo_epochs = ppo_config.get('ppo_epochs', 3)     # PPO更新轮数
        self.ppo_clip = ppo_config.get('ppo_clip', 0.2)       # PPO裁剪参数
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)  # GAE lambda
        self.value_loss_coef = ppo_config.get('value_loss_coef', 0.5)  # 价值损失系数
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)       # 熵正则化系数
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)      # 梯度裁剪

        # Off-Policy参数
        self.replay_buffer_size = ppo_config.get('replay_buffer_size', 50000)
        self.replay_ratio = ppo_config.get('replay_ratio', 0.3)  # 使用replay数据的比例
        self.replay_buffer = TSPReplayBuffer(self.replay_buffer_size) if self.ppo_mode == 'off_policy' else None

        # 设备设置
        use_cuda = self.trainer_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # 使用PPO版本的模型（带Critic）
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        self._load_checkpoint_if_needed(trainer_params.get('model_load', {'enable': False}))
        self.time_estimator = TimeEstimator()

        self.logger.info(f'PPO Mode: {self.ppo_mode}')
        self.logger.info(f'PPO Config: epochs={self.ppo_epochs}, clip={self.ppo_clip}, '
                        f'gae_lambda={self.gae_lambda}, value_coef={self.value_loss_coef}, '
                        f'entropy_coef={self.entropy_coef}, max_grad_norm={self.max_grad_norm}')

    def _load_checkpoint_if_needed(self, model_load):
        """加载checkpoint（兼容原始POMO和PPO模型）"""
        if not model_load.get('enable', False):
            return

        has_direct_path = 'checkpoint_path' in model_load
        if has_direct_path:
            checkpoint_fullname = model_load['checkpoint_path']
            checkpoint_epoch = None
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint_epoch = int(model_load['epoch'])

        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # 尝试加载，对于缺少critic_head的旧checkpoint使用partial loading
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if 'critic_head' in str(e):
                self.logger.info('Loading from non-PPO checkpoint, initializing critic_head randomly...')
                # 过滤掉critic_head相关的key
                model_state = self.model.state_dict()
                for key in state_dict:
                    if key in model_state and key.startswith('critic_head') is False:
                        model_state[key] = state_dict[key]
                self.model.load_state_dict(model_state)
            else:
                raise

        load_optimizer = model_load.get('load_optimizer', not has_direct_path)
        load_scheduler = model_load.get('load_scheduler', not has_direct_path)
        reset_epoch = model_load.get('reset_epoch', has_direct_path)

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = int(checkpoint.get('epoch', checkpoint_epoch or 0))
        if not reset_epoch and saved_epoch > 0:
            self.start_epoch = saved_epoch + 1
            if not load_scheduler:
                self.scheduler.last_epoch = saved_epoch - 1
        if 'result_log' in checkpoint:
            self.result_log.set_raw_data(checkpoint['result_log'])

        self.logger.info('Saved Model Loaded: {}'.format(checkpoint_fullname))

    # =========================================================================
    # 主训练循环
    # =========================================================================

    def run(self):
        """主训练循环"""
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            self.scheduler.step()

            # 根据PPO模式选择训练方法
            if self.ppo_mode == 'simple_ppo':
                train_score, train_loss = self._train_one_epoch_simple_ppo(epoch)
            elif self.ppo_mode == 'full_ppo':
                train_score, train_loss = self._train_one_epoch_full_ppo(epoch)
            elif self.ppo_mode == 'off_policy':
                train_score, train_loss = self._train_one_epoch_off_policy(epoch)
            else:
                raise ValueError(f"Unknown PPO mode: {self.ppo_mode}")

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params['epochs']
            )
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self._save_latest_images()

            if all_done or (epoch % model_save_interval) == 0:
                self._save_checkpoint(epoch)

            if all_done or (epoch % img_save_interval) == 0:
                self._save_checkpoint_images(epoch)

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    # =========================================================================
    # 模式1：简化PPO（推荐入门）
    # =========================================================================

    def _train_one_epoch_simple_ppo(self, epoch):
        """简化PPO：只添加Clipped Objective + 多轮更新"""
        score_am = AverageMeter()
        loss_am = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        active_sizes = self._active_problem_sizes(epoch)
        self.logger.info("Epoch {:3d}: active problem sizes: {} [Simple PPO]".format(epoch, active_sizes))

        while episode < train_num_episode:
            problem_size = self._sample_problem_size(epoch)
            distribution = self._sample_distribution()

            remaining = train_num_episode - episode
            batch_size = min(self._batch_size_for_problem(problem_size), remaining)

            avg_score, avg_loss = self._train_one_batch_simple_ppo(
                batch_size=batch_size,
                problem_size=problem_size,
                distribution=distribution,
            )
            score_am.update(avg_score, batch_size)
            loss_am.update(avg_loss, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) '
                        'N: {:3d}, dist: {:>13s}, Score: {:.4f}, Loss: {:.4f}'.format(
                            epoch, episode, train_num_episode,
                            100. * episode / train_num_episode,
                            problem_size, distribution, score_am.avg, loss_am.avg
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Loss: {:.4f}'.format(
                epoch, 100. * episode / train_num_episode, score_am.avg, loss_am.avg
            )
        )

        return score_am.avg, loss_am.avg

    def _train_one_batch_simple_ppo(self, batch_size, problem_size, distribution):
        """
        简化PPO训练一个batch
        
        核心改进：
        1. 先用no_grad收集数据，保存old_log_prob
        2. 多轮更新，使用Clipped Objective限制策略更新幅度
        """
        self.model.train()

        # ========== 阶段1：数据收集（no_grad）==========
        with torch.no_grad():
            self._load_training_problems(batch_size, problem_size, distribution)
            reset_state, _, _ = self.env.reset()
            z = self._make_z(batch_size, self.env.pomo_size)
            self.model.pre_forward(reset_state, z)

            prob_list = []
            state, reward, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob[:, :, None])

            prob_list = torch.cat(prob_list, dim=2)
            old_log_prob = prob_list.log().sum(dim=2).detach()  # (batch, pomo)

        # ========== 阶段2：PPO多轮更新 ==========
        total_loss = 0
        for ppo_step in range(self.ppo_epochs):
            self.model.pre_forward(reset_state, z)

            prob_list = []
            state, _, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob[:, :, None])

            prob_list = torch.cat(prob_list, dim=2)
            new_log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)

            # 计算重要性采样比率
            ratio = (new_log_prob - old_log_prob).exp()  # (batch, pomo)

            # 计算advantage（使用batch均值作为基线，与原始POMO相同）
            advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)

            # PPO Clipped Loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantage
            loss = -torch.min(surr1, surr2).mean()

            self.model.zero_grad()
            loss.backward()
            # 梯度裁剪（PPO常用技巧）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / self.ppo_epochs
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        return score_mean.item(), avg_loss

    # =========================================================================
    # 模式2：完整PPO（Critic + GAE + Clipped Objective）
    # =========================================================================

    def _train_one_epoch_full_ppo(self, epoch):
        """完整PPO训练"""
        score_am = AverageMeter()
        loss_am = AverageMeter()
        policy_loss_am = AverageMeter()
        value_loss_am = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        active_sizes = self._active_problem_sizes(epoch)
        self.logger.info("Epoch {:3d}: active problem sizes: {} [Full PPO]".format(epoch, active_sizes))

        while episode < train_num_episode:
            problem_size = self._sample_problem_size(epoch)
            distribution = self._sample_distribution()

            remaining = train_num_episode - episode
            batch_size = min(self._batch_size_for_problem(problem_size), remaining)

            avg_score, avg_loss, p_loss, v_loss = self._train_one_batch_full_ppo(
                batch_size=batch_size,
                problem_size=problem_size,
                distribution=distribution,
            )
            score_am.update(avg_score, batch_size)
            loss_am.update(avg_loss, batch_size)
            policy_loss_am.update(p_loss, batch_size)
            value_loss_am.update(v_loss, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) '
                        'N: {:3d}, dist: {:>13s}, Score: {:.4f}, Loss: {:.4f}, '
                        'PLoss: {:.4f}, VLoss: {:.4f}'.format(
                            epoch, episode, train_num_episode,
                            100. * episode / train_num_episode,
                            problem_size, distribution, score_am.avg, loss_am.avg,
                            policy_loss_am.avg, value_loss_am.avg
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Loss: {:.4f}, '
            'PLoss: {:.4f}, VLoss: {:.4f}'.format(
                epoch, 100. * episode / train_num_episode,
                score_am.avg, loss_am.avg,
                policy_loss_am.avg, value_loss_am.avg
            )
        )

        return score_am.avg, loss_am.avg

    def _train_one_batch_full_ppo(self, batch_size, problem_size, distribution):
        """
        完整PPO训练一个batch
        
        包含：
        1. Critic网络估计价值
        2. GAE计算优势
        3. Clipped Objective
        4. 价值函数损失
        5. 熵正则化
        """
        self.model.train()

        # ========== 阶段1：数据收集 ==========
        with torch.no_grad():
            self._load_training_problems(batch_size, problem_size, distribution)
            reset_state, _, _ = self.env.reset()
            z = self._make_z(batch_size, self.env.pomo_size)
            self.model.pre_forward(reset_state, z)

            log_probs = []
            values = []
            entropies = []

            state, reward, done = self.env.pre_step()
            step_count = 0
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                step_count += 1

                log_prob = prob.log()
                log_probs.append(log_prob)

                # 熵计算
                probs = prob.exp()
                entropy = -(probs * log_prob).sum(dim=-1)  # (batch, pomo)
                entropies.append(entropy)

                # 价值估计
                value = self.model.get_value(state)  # (batch,)
                values.append(value)

            # 堆叠数据
            log_probs_tensor = torch.stack(log_probs, dim=2)  # (batch, pomo, steps)
            values_tensor = torch.stack(values, dim=2)  # (batch, steps)
            entropies_tensor = torch.stack(entropies, dim=2)  # (batch, pomo, steps)

            # 计算GAE和returns
            returns, advantages = self._compute_gae(reward, values_tensor)

            # 保存旧策略的log_prob
            old_log_probs = log_probs_tensor.sum(dim=2).detach()  # (batch, pomo)
            old_entropies = entropies_tensor.mean(dim=2).detach()  # (batch, pomo)

        # ========== 阶段2：PPO多轮更新 ==========
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0

        for ppo_step in range(self.ppo_epochs):
            self.model.pre_forward(reset_state, z)

            new_log_probs = []
            new_entropies = []

            state, _, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)

                new_log_probs.append(prob.log())
                probs = prob.exp()
                entropy = -(probs * prob.log()).sum(dim=-1)
                new_entropies.append(entropy)

            new_log_probs = torch.stack(new_log_probs, dim=2).sum(dim=2)  # (batch, pomo)
            new_entropies = torch.stack(new_entropies, dim=2).mean(dim=2)  # (batch, pomo)

            # 重要性采样比率
            ratio = (new_log_probs - old_log_probs).exp()

            # PPO Clipped Policy Loss
            surr1 = ratio * advantages  # (batch, pomo)
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            new_value = self.model.get_value(state)  # (batch,)
            # 扩展returns以匹配pomo维度
            returns_expanded = returns.unsqueeze(1).expand_as(reward)  # (batch, pomo)
            # 对value也扩展
            new_value_expanded = new_value.unsqueeze(1).expand_as(reward)  # (batch, pomo)
            value_loss = ((new_value_expanded - returns_expanded) ** 2).mean()

            # Entropy Bonus
            entropy_loss = -new_entropies.mean()

            # 总损失
            loss = (policy_loss +
                   self.value_loss_coef * value_loss +
                   self.entropy_coef * entropy_loss)

            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()

        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_value_loss = total_value_loss / self.ppo_epochs
        avg_loss = total_loss / self.ppo_epochs

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        return score_mean.item(), avg_loss, avg_policy_loss, avg_value_loss

    def _compute_gae(self, rewards, values):
        """
        计算GAE (Generalized Advantage Estimation)
        
        对于TSP构造性问题，奖励只在最后一步给出。
        因此GAE的计算需要特殊处理。
        
        Args:
            rewards: (batch, pomo) - 最终奖励
            values: (batch, steps) - 每步的价值估计
        
        Returns:
            returns: (batch,) - 回报
            advantages: (batch, pomo) - 优势函数
        """
        # TSP的特殊性：奖励只在最后一步给出
        # 因此我们使用最后一步的value作为基线
        
        values_final = values[:, -1]  # (batch,) 最后一步的价值
        
        # Returns就是最终奖励（因为中间没有奖励）
        returns = rewards.max(dim=1).values.float()  # (batch,) 取最好的pomo的奖励
        
        # Advantage = reward - baseline(value)
        # 对每个pomo计算advantage
        advantages = rewards.float() - values_final.unsqueeze(1)  # (batch, pomo)
        
        # 标准化优势（提高训练稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    # =========================================================================
    # 模式3：Off-Policy训练（带Replay Buffer）
    # =========================================================================

    def _train_one_epoch_off_policy(self, epoch):
        """Off-Policy训练"""
        score_am = AverageMeter()
        loss_am = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        active_sizes = self._active_problem_sizes(epoch)
        self.logger.info("Epoch {:3d}: active problem sizes: {} [Off-Policy]".format(epoch, active_sizes))

        while episode < train_num_episode:
            problem_size = self._sample_problem_size(epoch)
            distribution = self._sample_distribution()

            remaining = train_num_episode - episode
            batch_size = min(self._batch_size_for_problem(problem_size), remaining)

            avg_score, avg_loss = self._train_one_batch_off_policy(
                batch_size=batch_size,
                problem_size=problem_size,
                distribution=distribution,
            )
            score_am.update(avg_score, batch_size)
            loss_am.update(avg_loss, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) '
                        'N: {:3d}, dist: {:>13s}, Score: {:.4f}, Loss: {:.4f}, '
                        'ReplayBuf: {}'.format(
                            epoch, episode, train_num_episode,
                            100. * episode / train_num_episode,
                            problem_size, distribution, score_am.avg, loss_am.avg,
                            len(self.replay_buffer)
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Loss: {:.4f}, '
            'ReplayBuf: {}'.format(
                epoch, 100. * episode / train_num_episode,
                score_am.avg, loss_am.avg, len(self.replay_buffer)
            )
        )

        return score_am.avg, loss_am.avg

    def _train_one_batch_off_policy(self, batch_size, problem_size, distribution):
        """
        Off-Policy训练一个batch
        
        混合策略：
        1. On-policy数据收集 + PPO更新
        2. 将好的解存入Replay Buffer
        3. 以一定概率从Replay Buffer采样进行Self-Imitation Learning
        """
        self.model.train()

        # ========== 阶段1：On-policy数据收集 ==========
        with torch.no_grad():
            self._load_training_problems(batch_size, problem_size, distribution)
            reset_state, _, _ = self.env.reset()
            z = self._make_z(batch_size, self.env.pomo_size)
            self.model.pre_forward(reset_state, z)

            prob_list = []
            selected_list = []

            state, reward, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob[:, :, None])
                selected_list.append(selected)

            prob_list = torch.cat(prob_list, dim=2)
            old_log_prob = prob_list.log().sum(dim=2).detach()

            # 将好的解存入Replay Buffer
            for i in range(batch_size):
                best_pomo_idx = reward[i].argmax()
                trajectory = torch.stack([s[i, best_pomo_idx] for s in selected_list])
                self.replay_buffer.push(
                    self.env.problems[i],
                    trajectory,
                    reward[i, best_pomo_idx].item(),
                    problem_size
                )

        # ========== 阶段2：PPO更新（同simple_ppo）==========
        total_loss = 0
        for ppo_step in range(self.ppo_epochs):
            self.model.pre_forward(reset_state, z)

            prob_list = []
            state, _, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list.append(prob[:, :, None])

            prob_list = torch.cat(prob_list, dim=2)
            new_log_prob = prob_list.log().sum(dim=2)

            ratio = (new_log_prob - old_log_prob).exp()
            advantage = reward - reward.float().mean(dim=1, keepdims=True)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantage
            ppo_loss = -torch.min(surr1, surr2).mean()

            self.model.zero_grad()
            ppo_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += ppo_loss.item()

        # ========== 阶段3：Self-Imitation Learning（可选）==========
        sil_loss_val = 0
        if len(self.replay_buffer) > batch_size and random.random() < self.replay_ratio:
            sil_loss = self._self_imitation_learning(batch_size // 2, problem_size)
            if sil_loss is not None:
                sil_loss_val = sil_loss

        avg_loss = total_loss / self.ppo_epochs + sil_loss_val

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        return score_mean.item(), avg_loss

    def _self_imitation_learning(self, batch_size, problem_size):
        """
        Self-Imitation Learning: 从Replay Buffer中学习历史最优解
        
        核心思想：让当前策略学习自己过去产生的好解，
        这是一种off-policy的方法，可以提高样本效率。
        
        参考: Oh et al. "Self-Imitation Learning" (ICML 2018)
        """
        # 从replay buffer采样最好的解
        problems, trajectories, rewards = self.replay_buffer.sample_best(
            batch_size, problem_size
        )
        
        if problems is None:
            return None
        
        problems = problems.to(self.device)
        trajectories = trajectories.to(self.device)
        rewards = rewards.to(self.device)
        
        # 用当前策略在历史问题上生成解
        self.env.problem_size = problem_size
        self.env.pomo_size = problem_size
        self.env.env_params['problem_size'] = problem_size
        self.env.env_params['pomo_size'] = problem_size
        self.env.batch_size = problems.size(0)
        self.env.problems = problems
        
        device = problems.device
        self.env.BATCH_IDX = torch.arange(problems.size(0), device=device)[:, None].expand(problems.size(0), problem_size)
        self.env.POMO_IDX = torch.arange(problem_size, device=device)[None, :].expand(problems.size(0), problem_size)
        
        reset_state, _, _ = self.env.reset()
        z = self._make_z(problems.size(0), problem_size)
        self.model.pre_forward(reset_state, z)
        
        # 收集当前策略的log_prob
        prob_list = []
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list.append(prob[:, :, None])
        
        prob_list = torch.cat(prob_list, dim=2)
        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
        
        # 只学习比历史最优更好的解（或接近的解）
        max_pomo_reward, _ = reward.max(dim=1)  # (batch,)
        
        # SIL Loss: 最大化好解的log_prob
        # 使用加权的方式：奖励越高的解权重越大
        advantage = max_pomo_reward.float() - rewards.float().to(self.device)
        # 只在当前策略产生更好解时才更新（避免退化）
        weight = torch.clamp(advantage, min=0)
        
        if weight.sum() > 0:
            sil_loss = -(weight * log_prob.max(dim=1).values).mean()
            
            self.model.zero_grad()
            sil_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            return sil_loss.item()
        
        return 0

    # =========================================================================
    # 辅助方法（与原始TSPTrainer相同）
    # =========================================================================

    def _load_training_problems(self, batch_size, problem_size, distribution):
        self.env.problem_size = problem_size
        self.env.pomo_size = problem_size
        self.env.env_params['problem_size'] = problem_size
        self.env.env_params['pomo_size'] = problem_size
        self.env.batch_size = batch_size
        self.env.problems = get_random_problems(
            batch_size=batch_size,
            problem_size=problem_size,
            distribution=distribution,
        )
        device = self.env.problems.device
        self.env.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, problem_size)
        self.env.POMO_IDX = torch.arange(problem_size, device=device)[None, :].expand(batch_size, problem_size)

    def _active_problem_sizes(self, epoch):
        schedule = self.trainer_params.get('problem_size_schedule')
        if not schedule:
            return [self.env_params['problem_size']]

        active = schedule[0]['sizes']
        for stage in schedule:
            if epoch >= stage['start_epoch']:
                active = stage['sizes']
            else:
                break
        return active

    def _sample_problem_size(self, epoch):
        return random.choice(self._active_problem_sizes(epoch))

    def _sample_distribution(self):
        mix = self.trainer_params.get('distribution_mix', {'uniform': 1.0})
        names = list(mix.keys())
        weights = list(mix.values())
        return random.choices(names, weights=weights, k=1)[0]

    def _batch_size_for_problem(self, problem_size):
        dynamic = self.trainer_params.get('dynamic_batch_sizes', {})
        if problem_size in dynamic:
            return dynamic[problem_size]
        return self.trainer_params['train_batch_size']

    def _make_z(self, batch_size, rollout_size):
        if not self.model_params.get('use_polynet', False):
            return None

        z_dim = self.model_params['z_dim']
        device = self.env.problems.device
        rollout_idx = torch.arange(rollout_size, device=device, dtype=torch.long)
        bit_idx = torch.arange(z_dim, device=device, dtype=torch.long)
        z = ((rollout_idx[:, None] >> bit_idx[None, :]) & 1).float()
        return z[None, :, :].expand(batch_size, rollout_size, z_dim)

    def _save_checkpoint(self, epoch):
        self.logger.info("Saving trained_model")
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data(),
            'trainer_params': self.trainer_params,
            'optimizer_params': self.optimizer_params,
            'model_params': self.model_params,
        }
        torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

    def _save_latest_images(self):
        try:
            self.logger.info("Saving log_image")
            image_prefix = '{}/latest'.format(self.result_folder)
            util_save_log_image_with_label(
                image_prefix,
                self.trainer_params['logging']['log_image_params_1'],
                self.result_log,
                labels=['train_score'],
            )
            util_save_log_image_with_label(
                image_prefix,
                self.trainer_params['logging']['log_image_params_2'],
                self.result_log,
                labels=['train_loss'],
            )
        except Exception as exc:
            self.logger.info("Skip latest log image because plotting failed: {}".format(exc))

    def _save_checkpoint_images(self, epoch):
        try:
            image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
            util_save_log_image_with_label(
                image_prefix,
                self.trainer_params['logging']['log_image_params_1'],
                self.result_log,
                labels=['train_score'],
            )
            util_save_log_image_with_label(
                image_prefix,
                self.trainer_params['logging']['log_image_params_2'],
                self.result_log,
                labels=['train_loss'],
            )
        except Exception as exc:
            self.logger.info("Skip checkpoint log image because plotting failed: {}".format(exc))
