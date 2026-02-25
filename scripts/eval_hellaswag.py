"""
HellaSwag 评估入口脚本。
"""

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from benchmarks.hellaswag import evaluate_hf_model


def main():
    parser = argparse.ArgumentParser(description="HellaSwag 评估")
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="要评估的模型类型")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="评估设备")
    args = parser.parse_args()
    evaluate_hf_model(args.model_type, args.device)


if __name__ == "__main__":
    main()

