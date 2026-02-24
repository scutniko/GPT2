"""
命令行参数解析
"""

import argparse


def parse_args():
    """解析训练入口参数。"""
    parser = argparse.ArgumentParser(description="GPT-2 Training with Ablation Studies")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置路径或名称（如 configs/experiments/baseline.yaml 或 baseline）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练 (e.g., log/model_15000.pt)",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="仅加载模型权重，不恢复优化器和step (e.g., log/model_15000.pt)",
    )
    parser.add_argument(
        "--inference",
        type=str,
        default=None,
        help="推理模式：加载检查点生成文本 (e.g., log/model_15000.pt)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="离线token shard目录（包含train/val切分的.npy文件）",
    )
    parser.add_argument(
        "--log_subdir",
        type=str,
        required=True,
        help="日志与checkpoint子目录（位于 log_train/<experiment>/ 下）",
    )
    args = parser.parse_args()
    if args.resume and args.init_from:
        parser.error("--resume 与 --init_from 不能同时使用")
    return args
