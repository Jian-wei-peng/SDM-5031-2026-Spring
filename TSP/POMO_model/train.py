import argparse
import json
import logging
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_DIR = os.path.abspath(os.path.join(TSP_DIR, ".."))

os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, TSP_DIR)
sys.path.insert(2, REPO_DIR)


from utils.utils import create_logger, copy_all_src, get_result_folder
from TSPTrainer import TSPTrainer as Trainer


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


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_int_list(value):
    if value is None or value == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_curriculum(value):
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="POMO model-side ablation: uniform fixed-size training with PolyNet-style decoder residual."
    )
    parser.add_argument("--exp_name", default="pomo_model_poly_stage1")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_device_num", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_episodes", type=int, default=100000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--scheduler_milestones", default="")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    parser.add_argument(
        "--init_checkpoint",
        default=os.path.abspath(
            os.path.join(SCRIPT_DIR, "result", "saved_tsp100_model2_longTrain", "checkpoint-3000.pt")
        ),
        help='Checkpoint used for fine-tuning. Pass "" to train from scratch.',
    )
    parser.add_argument("--load_optimizer", type=str2bool, default=False)
    parser.add_argument("--load_scheduler", type=str2bool, default=False)
    parser.add_argument("--reset_epoch", type=str2bool, default=True)

    parser.add_argument(
        "--curriculum",
        default="1:100",
        help="Format: start_epoch:size,size;start_epoch:size,size",
    )
    parser.add_argument(
        "--distribution_mix",
        default="uniform:1.0",
        help="Format: distribution:weight,distribution:weight",
    )
    parser.add_argument(
        "--dynamic_batch_sizes",
        default="100:64",
        help="Format: problem_size:batch_size,problem_size:batch_size",
    )

    parser.add_argument("--model_save_interval", type=int, default=50)
    parser.add_argument("--img_save_interval", type=int, default=50)
    parser.add_argument("--eval_after_train", type=str2bool, default=True)
    parser.add_argument("--eval_data_path", default=os.path.abspath(os.path.join(TSP_DIR, "data", "val")))
    parser.add_argument("--eval_aug_factor", type=int, default=8)
    parser.add_argument("--detailed_log", type=str2bool, default=False)

    parser.add_argument("--use_polynet", type=str2bool, default=True)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--poly_embedding_dim", type=int, default=256)
    parser.add_argument("--force_first_move", type=str2bool, default=True)
    return parser


def build_model_params(args):
    if not args.force_first_move:
        raise ValueError("POMO_model stage 1 supports only --force_first_move true.")

    model_params = dict(BASE_MODEL_PARAMS)
    model_params.update({
        'use_polynet': args.use_polynet,
        'z_dim': args.z_dim,
        'poly_embedding_dim': args.poly_embedding_dim,
        # Stage 1 keeps the original POMO first move: one rollout starts from each node.
        'force_first_move': args.force_first_move,
    })
    return model_params


def build_params(args):
    curriculum = parse_curriculum(args.curriculum)
    first_size = curriculum[0]["sizes"][0]
    init_checkpoint = args.init_checkpoint.strip()

    env_params = {
        'problem_size': first_size,
        'pomo_size': first_size,
    }
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
    trainer_params = {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'problem_size_schedule': curriculum,
        'distribution_mix': parse_distribution_mix(args.distribution_mix),
        'dynamic_batch_sizes': parse_dynamic_batch_sizes(args.dynamic_batch_sizes),
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
    logger_params = {
        'log_file': {
            'desc': args.exp_name,
            'filename': 'log.txt',
        }
    }
    return env_params, build_model_params(args), optimizer_params, trainer_params, logger_params


def dump_config(args, env_params, model_params, optimizer_params, trainer_params):
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


def main():
    args = build_parser().parse_args()
    env_params, model_params, optimizer_params, trainer_params, logger_params = build_params(args)

    create_logger(**logger_params)
    logger = logging.getLogger("root")
    logger.info("POMO model-side ablation: fixed-size uniform training with PolyNet residual")
    logger.info("SCRIPT_DIR: {}".format(SCRIPT_DIR))
    logger.info("TSP_DIR: {}".format(TSP_DIR))
    logger.info("args: {}".format(args))
    logger.info("env_params{}".format(env_params))
    logger.info("model_params{}".format(model_params))
    logger.info("optimizer_params{}".format(optimizer_params))
    logger.info("trainer_params{}".format(trainer_params))

    dump_config(args, env_params, model_params, optimizer_params, trainer_params)

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )

    copy_all_src(trainer.result_folder)
    trainer.run()

    final_checkpoint = os.path.abspath(
        os.path.join(get_result_folder(), "checkpoint-{}.pt".format(args.epochs))
    )
    if args.eval_after_train:
        run_validation(args, final_checkpoint)


if __name__ == "__main__":
    main()
