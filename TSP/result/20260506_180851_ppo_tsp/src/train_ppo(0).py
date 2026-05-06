"""
PPO训练入口脚本 - 快速开始指南

使用方法：
    # 1. 简化PPO（推荐入门）
    python train_ppo.py --mode simple_ppo --epochs 100

    # 2. 完整PPO（研究用）
    python train_ppo.py --mode full_ppo --epochs 100

    # 3. Off-Policy（高级）
    python train_ppo.py --mode off_policy --epochs 100 --replay_buffer_size 50000

训练模式说明：
    - simple_ppo: 只添加Clipped Objective，实现简单，效果稳定
    - full_ppo: 添加Critic网络和GAE，效果更好但需要调参
    - off_policy: 带Replay Buffer，样本效率最高但实现复杂

推荐流程：
    1. 先用simple_ppo测试代码是否正常运行
    2. 对比simple_ppo和原始POMO的性能
    3. 如果效果好，尝试full_ppo
    4. 如果追求更高的样本效率，尝试off_policy
"""

import argparse
import json
import logging
import os
import sys

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(TSP_DIR, ".."))

os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, TSP_DIR)
sys.path.insert(2, REPO_DIR)

from utils.utils import create_logger, copy_all_src, get_result_folder
from TSPTrainer_PPO import TSPTrainerPPO as Trainer

# =============================================================================
# 模型参数配置
# =============================================================================

BASE_MODEL_PARAMS = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

# =============================================================================
# 辅助函数
# =============================================================================

def str2bool(value):
    """将字符串转换为布尔值"""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(value):
    """解析整数列表"""
    if value is None or value == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_curriculum(value):
    """解析课程学习参数"""
    stages = []
    for raw_stage in value.split(";"):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        start_raw, sizes_raw = raw_stage.split(":", maxsplit=1)
        sizes = parse_int_list(sizes_raw)
        if not sizes:
            raise ValueError(f"Empty size list in curriculum stage: {raw_stage}")
        stages.append({"start_epoch": int(start_raw), "sizes": sizes})
    if not stages:
        raise ValueError("Curriculum must contain at least one stage.")
    return sorted(stages, key=lambda stage: stage["start_epoch"])


def parse_distribution_mix(value):
    """解析数据分布混合比例"""
    mix = {}
    for raw_item in value.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        name, weight = raw_item.split(":", maxsplit=1)
        mix[name.strip()] = float(weight)
    if not mix:
        raise ValueError("Distribution mix must not be empty.")
    return mix


def parse_dynamic_batch_sizes(value):
    """解析动态批大小"""
    sizes = {}
    if value is None or value == "":
        return sizes
    for raw_item in value.split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        problem_size, batch_size = raw_item.split(":", maxsplit=1)
        sizes[int(problem_size)] = int(batch_size)
    return sizes


# =============================================================================
# 参数解析
# =============================================================================

def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="PPO Training for TSP - 支持多种训练模式"
    )

    # ========== 训练模式 ==========
    parser.add_argument(
        "--mode",
        type=str,
        default="simple_ppo",
        choices=["simple_ppo", "full_ppo", "off_policy"],
        help="训练模式：simple_ppo（简化版）、full_ppo（完整版）、off_policy（离线策略）"
    )

    # ========== 基本配置 ==========
    parser.add_argument("--exp_name", default="ppo_tsp")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_device_num", type=int, default=0)

    # ========== 训练超参数 ==========
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_episodes", type=int, default=100000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--scheduler_milestones", default="")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    # ========== PPO专用参数 ==========
    parser.add_argument("--ppo_epochs", type=int, default=3,
                       help="PPO每批数据的更新轮数（推荐3-5）")
    parser.add_argument("--ppo_clip", type=float, default=0.2,
                       help="PPO裁剪参数（推荐0.1-0.3）")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda参数（推荐0.95）")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                       help="价值损失系数（推荐0.5）")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="熵正则化系数（推荐0.01）")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="梯度裁剪阈值（推荐0.5）")

    # ========== Off-Policy专用参数 ==========
    parser.add_argument("--replay_buffer_size", type=int, default=50000,
                       help="Replay Buffer大小（仅off_policy模式）")
    parser.add_argument("--replay_ratio", type=float, default=0.3,
                       help="使用replay数据的比例（仅off_policy模式）")

    # ========== 模型加载 ==========
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="初始化checkpoint路径（空字符串表示从头训练）"
    )
    parser.add_argument("--load_optimizer", type=str2bool, default=False)
    parser.add_argument("--load_scheduler", type=str2bool, default=False)
    parser.add_argument("--reset_epoch", type=str2bool, default=True)

    # ========== 课程学习 ==========
    parser.add_argument(
        "--curriculum",
        default="1:100",
        help="课程学习配置，格式：start_epoch:size,size;..."
    )
    parser.add_argument(
        "--distribution_mix",
        default="uniform:1.0",
        help="数据分布混合，格式：distribution:weight,..."
    )
    parser.add_argument(
        "--dynamic_batch_sizes",
        default="",
        help="动态批大小，格式：problem_size:batch_size,..."
    )

    # ========== 日志和保存 ==========
    parser.add_argument("--model_save_interval", type=int, default=10)
    parser.add_argument("--img_save_interval", type=int, default=10)

    # ========== PolyNet配置 ==========
    parser.add_argument("--use_polynet", type=str2bool, default=False)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--poly_embedding_dim", type=int, default=256)
    parser.add_argument("--force_first_move", type=str2bool, default=True)

    return parser


def build_model_params(args):
    """构建模型参数"""
    model_params = dict(BASE_MODEL_PARAMS)
    model_params.update({
        'use_polynet': args.use_polynet,
        'z_dim': args.z_dim,
        'poly_embedding_dim': args.poly_embedding_dim,
        'force_first_move': args.force_first_move,
    })
    return model_params


def build_params(args):
    """构建所有训练参数"""
    curriculum = parse_curriculum(args.curriculum)
    first_size = curriculum[0]["sizes"][0]
    init_checkpoint = args.init_checkpoint.strip()

    # 环境参数
    env_params = {
        'problem_size': first_size,
        'pomo_size': first_size,
    }

    # 优化器参数
    optimizer_params = {
        'optimizer': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        },
        'scheduler': {
            'milestones': parse_int_list(args.scheduler_milestones),
            'gamma': args.scheduler_gamma,
        },
    }

    # PPO配置
    ppo_config = {
        'mode': args.mode,
        'ppo_epochs': args.ppo_epochs,
        'ppo_clip': args.ppo_clip,
        'gae_lambda': args.gae_lambda,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'replay_buffer_size': args.replay_buffer_size,
        'replay_ratio': args.replay_ratio,
    }

    # 训练器参数
    trainer_params = {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'problem_size_schedule': curriculum,
        'distribution_mix': parse_distribution_mix(args.distribution_mix),
        'dynamic_batch_sizes': parse_dynamic_batch_sizes(args.dynamic_batch_sizes),
        'ppo_config': ppo_config,
        'logging': {
            'model_save_interval': args.model_save_interval,
            'img_save_interval': args.img_save_interval,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_tsp_20.json',
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json',
            },
        },
        'model_load': {
            'enable': bool(init_checkpoint),
            'checkpoint_path': os.path.abspath(init_checkpoint) if init_checkpoint else "",
            'load_optimizer': args.load_optimizer,
            'load_scheduler': args.load_scheduler,
            'reset_epoch': args.reset_epoch,
        },
    }

    # 日志参数
    logger_params = {
        'log_file': {
            'desc': args.exp_name,
            'filename': 'log.txt',
        }
    }

    return env_params, build_model_params(args), optimizer_params, trainer_params, logger_params


def dump_config(args, env_params, model_params, optimizer_params, trainer_params):
    """保存配置到JSON文件"""
    config = {
        "args": vars(args),
        "env_params": env_params,
        "model_params": model_params,
        "optimizer_params": optimizer_params,
        "trainer_params": trainer_params,
    }
    config_path = os.path.join(get_result_folder(), "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    # 解析参数
    args = build_parser().parse_args()
    env_params, model_params, optimizer_params, trainer_params, logger_params = build_params(args)

    # 初始化日志
    create_logger(**logger_params)
    logger = logging.getLogger("root")
    
    # 打印配置
    logger.info("=" * 70)
    logger.info("PPO Training for TSP")
    logger.info("=" * 70)
    logger.info(f"Training Mode: {args.mode}")
    logger.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
    logger.info(f"TSP_DIR: {TSP_DIR}")
    logger.info("=" * 70)
    logger.info(f"args: {args}")
    logger.info(f"env_params: {env_params}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"optimizer_params: {optimizer_params}")
    logger.info(f"trainer_params: {trainer_params}")

    # 保存配置
    dump_config(args, env_params, model_params, optimizer_params, trainer_params)

    # 创建训练器
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )

    # 备份源代码
    copy_all_src(trainer.result_folder)
    
    # 开始训练
    logger.info("=" * 70)
    logger.info("Starting Training...")
    logger.info("=" * 70)
    trainer.run()

    logger.info("=" * 70)
    logger.info("Training Completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
