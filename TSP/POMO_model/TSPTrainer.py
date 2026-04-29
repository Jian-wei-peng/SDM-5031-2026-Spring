from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from utils.utils import *


class TSPTrainer:
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

        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        self._load_checkpoint_if_needed(trainer_params.get('model_load', {'enable': False}))
        self.time_estimator = TimeEstimator()

    def _load_checkpoint_if_needed(self, model_load):
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
        is_polynet_ckpt = any(key.startswith('decoder.poly_layer') for key in state_dict)
        use_polynet = self.model_params.get('use_polynet', False)
        self.model.load_state_dict(state_dict, strict=(not use_polynet or is_polynet_ckpt))

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
        self.logger.info(
            'load_optimizer: {}, load_scheduler: {}, reset_epoch: {}, polynet_checkpoint: {}'.format(
                load_optimizer, load_scheduler, reset_epoch, is_polynet_ckpt
            )
        )

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            self.scheduler.step()

            train_score, train_loss = self._train_one_epoch(epoch)
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

    def _train_one_epoch(self, epoch):
        score_am = AverageMeter()
        loss_am = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        problem_size = self.env_params['problem_size']
        self.logger.info("Epoch {:3d}: model-only ablation uses fixed uniform N={}".format(epoch, problem_size))

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
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
                            problem_size, 'uniform', score_am.avg, loss_am.avg
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Loss: {:.4f}'.format(
                epoch, 100. * episode / train_num_episode, score_am.avg, loss_am.avg
            )
        )

        return score_am.avg, loss_am.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        z = self._make_z(batch_size, self.env.pomo_size)
        self.model.pre_forward(reset_state, z)

        prob_list = torch.zeros(
            size=(batch_size, self.env.pomo_size, 0),
            device=self.env.problems.device,
        )

        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

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
