# import argparse
# import json
# import logging
# import os
# import subprocess
# import sys
#
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# TSP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# REPO_DIR = os.path.abspath(os.path.join(TSP_DIR, ".."))
#
# os.chdir(SCRIPT_DIR)
# sys.path.insert(0, SCRIPT_DIR)
# sys.path.insert(1, TSP_DIR)
# sys.path.insert(2, REPO_DIR)
#
#
# from utils.utils import create_logger, copy_all_src, get_result_folder
# from TSPTrainer import TSPTrainer as Trainer
#
#
# BASE_MODEL_PARAMS = {
#     'embedding_dim': 128,
#     'sqrt_embedding_dim': 128 ** (1 / 2),
#     'encoder_layer_num': 6,
#     'qkv_dim': 16,
#     'head_num': 8,
#     'logit_clipping': 10,
#     'ff_hidden_dim': 512,
#     'eval_type': 'argmax',
# }
#
#
# def str2bool(value):
#     if isinstance(value, bool):
#         return value
#     lowered = value.lower()
#     if lowered in {"true", "1", "yes", "y"}:
#         return True
#     if lowered in {"false", "0", "no", "n"}:
#         return False
#     raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
#
#
# def parse_int_list(value):
#     if value is None or value == "":
#         return []
#     return [int(item.strip()) for item in value.split(",") if item.strip()]
#
#
# def parse_curriculum(value):
#     stages = []
#     for raw_stage in value.split(";"):
#         raw_stage = raw_stage.strip()
#         if not raw_stage:
#             continue
#         start_raw, sizes_raw = raw_stage.split(":", maxsplit=1)
#         sizes = parse_int_list(sizes_raw)
#         if not sizes:
#             raise ValueError(f"Empty size list in curriculum stage: {raw_stage}")
#         stages.append({"start_epoch": int(start_raw), "sizes": sizes})
#     if not stages:
#         raise ValueError("Curriculum must contain at least one stage.")
#     return sorted(stages, key=lambda stage: stage["start_epoch"])
#
#
# def parse_distribution_mix(value):
#     mix = {}
#     for raw_item in value.split(","):
#         raw_item = raw_item.strip()
#         if not raw_item:
#             continue
#         name, weight = raw_item.split(":", maxsplit=1)
#         mix[name.strip()] = float(weight)
#     if not mix:
#         raise ValueError("Distribution mix must not be empty.")
#     return mix
#
#
# def parse_dynamic_batch_sizes(value):
#     sizes = {}
#     if value is None or value == "":
#         return sizes
#     for raw_item in value.split(","):
#         raw_item = raw_item.strip()
#         if not raw_item:
#             continue
#         problem_size, batch_size = raw_item.split(":", maxsplit=1)
#         sizes[int(problem_size)] = int(batch_size)
#     return sizes
#
#
# def build_parser():
#     parser = argparse.ArgumentParser(
#         description="POMO training entrypoint with multi-scale distribution-aware fine-tuning."
#     )
#     parser.add_argument("--exp_name", default="pomo_ft_curriculum_mixed")
#     parser.add_argument("--use_cuda", type=str2bool, default=True)
#     parser.add_argument("--cuda_device_num", type=int, default=0)
#
#     parser.add_argument("--epochs", type=int, default=500)
#     parser.add_argument("--train_episodes", type=int, default=100000)
#     parser.add_argument("--train_batch_size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=1e-5)
#     parser.add_argument("--weight_decay", type=float, default=1e-6)
#     parser.add_argument("--scheduler_milestones", default="")
#     parser.add_argument("--scheduler_gamma", type=float, default=0.1)
#
#     parser.add_argument(
#         "--init_checkpoint",
#         default=os.path.abspath(
#             os.path.join(SCRIPT_DIR, "result", "saved_tsp100_model2_longTrain", "checkpoint-3000.pt")
#         ),
#         help='Checkpoint used for fine-tuning. Pass "" to train from scratch.',
#     )
#     parser.add_argument("--load_optimizer", type=str2bool, default=False)
#     parser.add_argument("--load_scheduler", type=str2bool, default=False)
#     parser.add_argument("--reset_epoch", type=str2bool, default=True)
#
#     parser.add_argument(
#         "--curriculum",
#         default="1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300",
#         help="Format: start_epoch:size,size;start_epoch:size,size",
#     )
#     parser.add_argument(
#         "--distribution_mix",
#         default="uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05",
#         help="Format: distribution:weight,distribution:weight",
#     )
#     parser.add_argument(
#         "--dynamic_batch_sizes",
#         default="100:64,125:48,150:40,200:24,250:16,300:12",
#         help="Format: problem_size:batch_size,problem_size:batch_size",
#     )
#
#     parser.add_argument("--model_save_interval", type=int, default=50)
#     parser.add_argument("--img_save_interval", type=int, default=50)
#     parser.add_argument("--eval_after_train", type=str2bool, default=True)
#     parser.add_argument("--eval_data_path", default=os.path.abspath(os.path.join(TSP_DIR, "data", "val")))
#     parser.add_argument("--eval_aug_factor", type=int, default=8)
#     parser.add_argument("--detailed_log", type=str2bool, default=False)
#
#     parser.add_argument("--use_polynet", type=str2bool, default=True)
#     parser.add_argument("--z_dim", type=int, default=16)
#     parser.add_argument("--poly_embedding_dim", type=int, default=256)
#     parser.add_argument("--force_first_move", type=str2bool, default=True)
#     return parser
#
#
# def build_model_params(args):
#     if not args.force_first_move:
#         raise ValueError("PJW-model stage 1 supports only --force_first_move true.")
#
#     model_params = dict(BASE_MODEL_PARAMS)
#     model_params.update({
#         'use_polynet': args.use_polynet,
#         'z_dim': args.z_dim,
#         'poly_embedding_dim': args.poly_embedding_dim,
#         # Stage 1 keeps the original POMO first move: one rollout starts from each node.
#         'force_first_move': args.force_first_move,
#     })
#     return model_params
#
#
# def build_params(args):
#     curriculum = parse_curriculum(args.curriculum)
#     first_size = curriculum[0]["sizes"][0]
#     init_checkpoint = args.init_checkpoint.strip()
#
#     env_params = {
#         'problem_size': first_size,
#         'pomo_size': first_size,
#     }
#     optimizer_params = {
#         'optimizer': {
#             'lr': args.lr,
#             'weight_decay': args.weight_decay,
#         },
#         'scheduler': {
#             'milestones': parse_int_list(args.scheduler_milestones),
#             'gamma': args.scheduler_gamma,
#         },
#     }
#     trainer_params = {
#         'use_cuda': args.use_cuda,
#         'cuda_device_num': args.cuda_device_num,
#         'epochs': args.epochs,
#         'train_episodes': args.train_episodes,
#         'train_batch_size': args.train_batch_size,
#         'problem_size_schedule': curriculum,
#         'distribution_mix': parse_distribution_mix(args.distribution_mix),
#         'dynamic_batch_sizes': parse_dynamic_batch_sizes(args.dynamic_batch_sizes),
#         'logging': {
#             'model_save_interval': args.model_save_interval,
#             'img_save_interval': args.img_save_interval,
#             'log_image_params_1': {
#                 'json_foldername': 'log_image_style',
#                 'filename': 'style_tsp_20.json',
#             },
#             'log_image_params_2': {
#                 'json_foldername': 'log_image_style',
#                 'filename': 'style_loss_1.json',
#             },
#         },
#         'model_load': {
#             'enable': bool(init_checkpoint),
#             'checkpoint_path': os.path.abspath(init_checkpoint) if init_checkpoint else "",
#             'load_optimizer': args.load_optimizer,
#             'load_scheduler': args.load_scheduler,
#             'reset_epoch': args.reset_epoch,
#         },
#     }
#     logger_params = {
#         'log_file': {
#             'desc': args.exp_name,
#             'filename': 'log.txt',
#         }
#     }
#     return env_params, build_model_params(args), optimizer_params, trainer_params, logger_params
#
#
# def dump_config(args, env_params, model_params, optimizer_params, trainer_params):
#     config = {
#         "args": vars(args),
#         "env_params": env_params,
#         "model_params": model_params,
#         "optimizer_params": optimizer_params,
#         "trainer_params": trainer_params,
#     }
#     config_path = os.path.join(get_result_folder(), "config.json")
#     with open(config_path, "w", encoding="utf-8") as f:
#         json.dump(config, f, ensure_ascii=False, indent=2)
#
#
# def run_validation(args, checkpoint_path):
#     eval_json = os.path.join(get_result_folder(), "eval_val.json")
#     command = [
#         sys.executable,
#         "test.py",
#         "--data_path", os.path.abspath(args.eval_data_path),
#         "--checkpoint_path", os.path.abspath(checkpoint_path),
#         "--use_cuda", str(args.use_cuda).lower(),
#         "--cuda_device_num", str(args.cuda_device_num),
#         "--augmentation_enable", "true",
#         "--aug_factor", str(args.eval_aug_factor),
#         "--detailed_log", str(args.detailed_log).lower(),
#         "--output_json", os.path.abspath(eval_json),
#     ]
#     logging.getLogger("root").info("Running validation: {}".format(" ".join(command)))
#     subprocess.run(command, cwd=SCRIPT_DIR, check=True)
#
#
# def main():
#     args = build_parser().parse_args()
#     env_params, model_params, optimizer_params, trainer_params, logger_params = build_params(args)
#
#     create_logger(**logger_params)
#     logger = logging.getLogger("root")
#     logger.info("POMO multi-scale distribution-aware training")
#     logger.info("SCRIPT_DIR: {}".format(SCRIPT_DIR))
#     logger.info("TSP_DIR: {}".format(TSP_DIR))
#     logger.info("args: {}".format(args))
#     logger.info("env_params{}".format(env_params))
#     logger.info("model_params{}".format(model_params))
#     logger.info("optimizer_params{}".format(optimizer_params))
#     logger.info("trainer_params{}".format(trainer_params))
#
#     dump_config(args, env_params, model_params, optimizer_params, trainer_params)
#
#     trainer = Trainer(
#         env_params=env_params,
#         model_params=model_params,
#         optimizer_params=optimizer_params,
#         trainer_params=trainer_params,
#     )
#
#     copy_all_src(trainer.result_folder)
#     trainer.run()
#
#     final_checkpoint = os.path.abspath(
#         os.path.join(get_result_folder(), "checkpoint-{}.pt".format(args.epochs))
#     )
#     if args.eval_after_train:
#         run_validation(args, final_checkpoint)
#
#
# if __name__ == "__main__":
#     main()

import argparse
import json
import logging
import os
import subprocess
import sys

# ---------------------------- 路径设置 ----------------------------
# 获取当前脚本所在的目录（train/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取TSP项目的根目录（train/ 的上一级目录，即 TSP/）
TSP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# 获取仓库根目录（TSP/ 的上一级目录，可能是整个项目的根目录）
REPO_DIR = os.path.abspath(os.path.join(TSP_DIR, ".."))

# 切换工作目录到脚本目录，保证相对路径正确
os.chdir(SCRIPT_DIR)
# 将几个关键目录加入 Python 模块搜索路径，方便导入自定义模块
sys.path.insert(0, SCRIPT_DIR)   # 当前目录
sys.path.insert(1, TSP_DIR)      # TSP 项目目录
sys.path.insert(2, REPO_DIR)     # 仓库根目录

# 导入自定义工具和训练器
from utils.utils import create_logger, copy_all_src, get_result_folder
from TSPTrainer import TSPTrainer as Trainer

# ---------------------------- 基础模型参数 ----------------------------
# 定义 POMO 模型的基础超参数（与论文/原始代码保持一致）
BASE_MODEL_PARAMS = {
    'embedding_dim': 128,               # 嵌入维度
    'sqrt_embedding_dim': 128 ** (1 / 2), # 嵌入维度的平方根，用于缩放注意力
    'encoder_layer_num': 6,             # 编码器层数
    'qkv_dim': 16,                      # 注意力中 Q、K、V 的维度
    'head_num': 8,                      # 多头注意力头数
    'logit_clipping': 10,               # logits 裁剪范围
    'ff_hidden_dim': 512,               # 前馈网络隐藏层维度
    'eval_type': 'argmax',              # 评估时选择动作的方式（argmax 或 sampling）
}

# ---------------------------- 辅助解析函数 ----------------------------
def str2bool(value):
    """将字符串或布尔值转换为布尔类型，用于 argparse 的 type 参数"""
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def parse_int_list(value):
    """解析逗号分隔的整数列表，例如 '100,125,150' -> [100,125,150]"""
    if value is None or value == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def parse_curriculum(value):
    """
    解析课程学习（curriculum）参数。
    格式： "start_epoch:size,size;start_epoch:size,size"
    例如："1:100;51:100,125,150"
    表示：第1个epoch开始只用size=100的训练；第51个epoch开始使用size=100,125,150混合。
    返回按 start_epoch 排序的列表，每个元素是一个字典 {'start_epoch': int, 'sizes': list}
    """
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
    """
    解析训练数据的分布混合比例。
    格式："分布名:权重,分布名:权重" 例如 "uniform:0.70,clustered:0.10"
    返回字典 {'分布名': 权重, ...}
    """
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
    """
    解析动态批大小（针对不同问题规模使用不同 batch size）。
    格式："问题规模:批大小,问题规模:批大小" 例如 "100:64,125:48"
    返回字典 {问题规模: 批大小, ...}
    """
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

# ---------------------------- 命令行参数构建 ----------------------------
def build_parser():
    """构建 ArgumentParser，定义所有命令行参数"""
    parser = argparse.ArgumentParser(
        description="POMO training entrypoint with multi-scale distribution-aware fine-tuning."
    )
    # 基本配置
    parser.add_argument("--exp_name", default="pomo_ft_curriculum_mixed")  # 实验名称
    parser.add_argument("--use_cuda", type=str2bool, default=True)         # 是否使用 GPU
    parser.add_argument("--cuda_device_num", type=int, default=0)          # GPU 设备编号

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=500)                 # 总训练轮数（epoch）
    parser.add_argument("--train_episodes", type=int, default=100000)      # 每个 epoch 的 episode 数（训练采样次数）
    parser.add_argument("--train_batch_size", type=int, default=64)        # 默认批大小（可能被动态批大小覆盖）
    parser.add_argument("--lr", type=float, default=1e-5)                  # 学习率
    parser.add_argument("--weight_decay", type=float, default=1e-6)        # 权重衰减
    parser.add_argument("--scheduler_milestones", default="")              # 学习率衰减的里程碑（逗号分隔），如 "100,200"
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)      # 学习率衰减因子

    # 模型加载/初始化
    parser.add_argument(
        "--init_checkpoint",
        default=os.path.abspath(
            os.path.join(SCRIPT_DIR, "result", "20260506_162527_pomo_ft_curriculum_mixed", "checkpoint-10.pt")
        ),
        help='Checkpoint used for fine-tuning. Pass "" to train from scratch.',
    )
    parser.add_argument("--load_optimizer", type=str2bool, default=False)   # 是否加载优化器状态
    parser.add_argument("--load_scheduler", type=str2bool, default=False)   # 是否加载学习率调度器状态
    parser.add_argument("--reset_epoch", type=str2bool, default=False)       # 是否重置训练轮数计数器

    # 课程学习与数据分布
    parser.add_argument(
        "--curriculum",
        default="1:100;51:100,125,150;151:100,125,150,200;301:100,125,150,200,250,300",
        help="Format: start_epoch:size,size;start_epoch:size,size",
    )
    parser.add_argument(
        "--distribution_mix",
        default="uniform:0.70,clustered:0.10,anisotropic:0.10,grid_jitter:0.05,mixed_density:0.05",
        help="Format: distribution:weight,distribution:weight",
    )
    parser.add_argument(
        "--dynamic_batch_sizes",
        default="100:64,125:48,150:40,200:24,250:16,300:12",
        help="Format: problem_size:batch_size,problem_size:batch_size",
    )

    # 日志与评估
    parser.add_argument("--model_save_interval", type=int, default=10)      # 模型保存间隔（epoch）
    parser.add_argument("--img_save_interval", type=int, default=10)        # 图像保存间隔（可能是损失图等）
    parser.add_argument("--eval_after_train", type=str2bool, default=True)  # 训练完成后是否运行验证
    parser.add_argument("--eval_data_path", default=os.path.abspath(os.path.join(TSP_DIR, "data", "val")))  # 验证数据路径
    parser.add_argument("--eval_aug_factor", type=int, default=8)           # 验证时数据增强倍数（如8倍对称性）
    parser.add_argument("--detailed_log", type=str2bool, default=False)     # 是否输出详细日志

    # 模型特定选项（PolyNet 相关）
    parser.add_argument("--use_polynet", type=str2bool, default=True)       # 是否使用 PolyNet 层
    parser.add_argument("--z_dim", type=int, default=16)                    # 隐变量 z 的维度
    parser.add_argument("--poly_embedding_dim", type=int, default=256)      # PolyNet 的嵌入维度
    parser.add_argument("--force_first_move", type=str2bool, default=True)  # 是否强制第一步选择所有节点（POMO风格）

    return parser

def build_model_params(args):
    """根据命令行参数构建模型参数字典"""
    # 根据 POMO 论文，Stage 1 必须强制第一步从每个节点开始，否则报错
    if not args.force_first_move:
        raise ValueError("PJW-model stage 1 supports only --force_first_move true.")

    model_params = dict(BASE_MODEL_PARAMS)
    model_params.update({
        'use_polynet': args.use_polynet,
        'z_dim': args.z_dim,
        'poly_embedding_dim': args.poly_embedding_dim,
        'force_first_move': args.force_first_move,
    })
    return model_params

def build_params(args):
    """将所有参数组装成训练需要的字典（环境、模型、优化器、训练器、日志）"""
    curriculum = parse_curriculum(args.curriculum)
    # 课程学习中第一个阶段的问题规模（用于初始化环境）
    first_size = curriculum[0]["sizes"][0]
    init_checkpoint = args.init_checkpoint.strip()

    # 环境参数：问题规模、POMO 中并行探索的个数（等于问题规模）
    env_params = {
        'problem_size': first_size,
        'pomo_size': first_size,
    }

    # 优化器与调度器参数
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

    # 训练器参数（核心配置）
    trainer_params = {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'problem_size_schedule': curriculum,                # 课程学习安排
        'distribution_mix': parse_distribution_mix(args.distribution_mix),  # 数据分布混合
        'dynamic_batch_sizes': parse_dynamic_batch_sizes(args.dynamic_batch_sizes),  # 动态批大小
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
            'enable': bool(init_checkpoint),                # 是否加载已有的模型
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
    """将全部配置保存为 JSON 文件，便于重现实验"""
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

def run_validation(args, checkpoint_path):
    """训练完成后调用 test.py 脚本对验证集进行评估"""
    eval_json = os.path.join(get_result_folder(), "eval_val.json")
    command = [
        sys.executable,
        "test.py",
        "--data_path", os.path.abspath(args.eval_data_path),
        "--checkpoint_path", os.path.abspath(checkpoint_path),
        "--use_cuda", str(args.use_cuda).lower(),
        "--cuda_device_num", str(args.cuda_device_num),
        "--augmentation_enable", "true",
        "--aug_factor", str(args.eval_aug_factor),
        "--detailed_log", str(args.detailed_log).lower(),
        "--output_json", os.path.abspath(eval_json),
    ]
    logging.getLogger("root").info("Running validation: {}".format(" ".join(command)))
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)

# ---------------------------- 主函数 ----------------------------
def main():
    # 1. 解析命令行参数
    args = build_parser().parse_args()
    # 2. 构建各个参数字典
    env_params, model_params, optimizer_params, trainer_params, logger_params = build_params(args)

    # 3. 初始化日志记录器
    create_logger(**logger_params)
    logger = logging.getLogger("root")
    logger.info("POMO multi-scale distribution-aware training")
    logger.info("SCRIPT_DIR: {}".format(SCRIPT_DIR))
    logger.info("TSP_DIR: {}".format(TSP_DIR))
    logger.info("args: {}".format(args))
    logger.info("env_params{}".format(env_params))
    logger.info("model_params{}".format(model_params))
    logger.info("optimizer_params{}".format(optimizer_params))
    logger.info("trainer_params{}".format(trainer_params))

    # 4. 保存配置到文件
    dump_config(args, env_params, model_params, optimizer_params, trainer_params)

    # 5. 创建训练器实例并开始训练
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )

    # 将当前所有源代码复制到结果文件夹，方便回溯
    copy_all_src(trainer.result_folder)
    trainer.run()   # 执行训练循环

    # 6. 训练完成后可选地运行验证
    final_checkpoint = os.path.abspath(
        os.path.join(get_result_folder(), "checkpoint-{}.pt".format(args.epochs))
    )
    if args.eval_after_train:
        run_validation(args, final_checkpoint)

if __name__ == "__main__":
    main()