"""
命令行参数解析
"""

import argparse


def parse_args():
    """解析训练与推理入口参数。"""
    parser = argparse.ArgumentParser(description="GPT-2 训练与推理入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练模式")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="实验配置路径或名称（如 configs/experiments/baseline.yaml 或 baseline）",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练 (e.g., log/model_15000.pt)",
    )
    train_parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="仅加载模型权重，不恢复优化器和step (e.g., log/model_15000.pt)",
    )
    train_parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="离线token shard目录（包含train/val切分的.npy文件）",
    )
    train_parser.add_argument(
        "--log_subdir",
        type=str,
        required=True,
        help="日志与checkpoint子目录（位于 log_train/<experiment>/ 下）",
    )

    infer_parser = subparsers.add_parser("infer", help="推理模式")
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="用于推理的 checkpoint 路径",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选：显式指定实验配置路径或名称；不提供时将尝试从 checkpoint 元数据推断",
    )
    infer_parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, I'm a language model,",
        help="推理提示词",
    )
    infer_parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="生成总长度（包含提示词）",
    )
    infer_parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=5,
        help="返回生成条数",
    )
    infer_parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="top-k 采样参数",
    )
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度（>0）",
    )
    infer_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="推理随机种子",
    )

    args = parser.parse_args()
    if args.command == "train" and args.resume and args.init_from:
        parser.error("--resume 与 --init_from 不能同时使用")
    if args.command == "infer":
        if args.max_length <= 0:
            parser.error("--max_length 必须 > 0")
        if args.num_return_sequences <= 0:
            parser.error("--num_return_sequences 必须 > 0")
        if args.top_k <= 0:
            parser.error("--top_k 必须 > 0")
        if args.temperature <= 0:
            parser.error("--temperature 必须 > 0")
    return args
