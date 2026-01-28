"""
离线预处理：原始语料 -> 清洗/分词 -> .npy token shard
支持 jsonl / parquet
"""

import argparse
import os
import random

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def _collect_files(input_path, data_format):
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    exts = []
    if data_format == "jsonl":
        exts = [".jsonl", ".json"]
    elif data_format == "parquet":
        exts = [".parquet"]
    elif data_format == "auto":
        exts = [".jsonl", ".json", ".parquet"]
    else:
        raise ValueError(f"不支持的数据格式: {data_format}")

    files = []
    for root, _, filenames in os.walk(input_path):
        for name in filenames:
            if any(name.endswith(ext) for ext in exts):
                files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {input_path}")
    return sorted(files)


def _infer_format(files):
    exts = {os.path.splitext(f)[1].lower() for f in files}
    if exts.issubset({".jsonl", ".json"}):
        return "jsonl"
    if exts.issubset({".parquet"}):
        return "parquet"
    raise ValueError("auto 模式下检测到混合格式，请显式指定 --format")


def _load_dataset(files, data_format, streaming=False):
    if data_format == "jsonl":
        return load_dataset("json", data_files=files, split="train", streaming=streaming)
    if data_format == "parquet":
        return load_dataset("parquet", data_files=files, split="train", streaming=streaming)
    raise ValueError(f"不支持的数据格式: {data_format}")


def _save_shard(output_dir, prefix, split, shard_idx, tokens):
    filename = f"{prefix}_{split}_{shard_idx:06d}.npy"
    path = os.path.join(output_dir, filename)
    arr = np.array(tokens, dtype=np.int32)
    np.save(path, arr)


def main():
    parser = argparse.ArgumentParser(description="离线预处理：原始语料 -> .npy token shard")
    parser.add_argument("--input", type=str, required=True, help="输入文件或目录")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "jsonl", "parquet"], help="输入格式")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--prefix", type=str, default="dataset", help="输出文件前缀")
    parser.add_argument("--text_field", type=str, default="text", help="文本字段名")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="tiktoken 编码名")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="验证集比例（未指定 --split_field 时生效）")
    parser.add_argument("--split_field", type=str, default=None, help="已存在的划分字段名（例如 partition）")
    parser.add_argument("--train_split", type=str, default="train", help="训练集字段值（用于 --split_field）")
    parser.add_argument("--val_split", type=str, default="val", help="验证集字段值（用于 --split_field）")
    parser.add_argument("--test_split", type=str, default="test", help="测试集字段值（用于 --split_field）")
    parser.add_argument("--shard_tokens", type=int, default=50_000_000, help="每个 shard 的 token 数")
    parser.add_argument("--min_chars", type=int, default=1, help="最小字符长度")
    parser.add_argument("--max_chars", type=int, default=0, help="最大字符长度，0 表示不截断")
    parser.add_argument("--add_eot", action="store_true", help="在每条样本末尾追加 EOT")
    parser.add_argument("--streaming", action="store_true", help="使用 streaming 读取（省内存）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    files = _collect_files(args.input, args.format)
    data_format = args.format
    if data_format == "auto":
        data_format = _infer_format(files)

    dataset = _load_dataset(files, data_format, streaming=args.streaming)
    enc = tiktoken.get_encoding(args.tokenizer)
    eot_token = getattr(enc, "eot_token", None)

    train_buf = []
    val_buf = []
    test_buf = []
    train_shard_idx = 0
    val_shard_idx = 0
    test_shard_idx = 0

    num_docs = 0
    num_skipped = 0
    num_train_tokens = 0
    num_val_tokens = 0
    num_test_tokens = 0

    total = None if args.streaming else len(dataset)
    pbar = tqdm(dataset, total=total, desc="处理样本")

    for item in pbar:
        text = item.get(args.text_field, None)
        if text is None:
            num_skipped += 1
            continue
        text = str(text).strip()
        if len(text) < args.min_chars:
            num_skipped += 1
            continue
        if args.max_chars > 0 and len(text) > args.max_chars:
            text = text[:args.max_chars]

        tokens = enc.encode(text)
        if args.add_eot:
            if eot_token is None:
                raise ValueError("当前 tokenizer 不支持 eot_token，请关闭 --add_eot")
            tokens.append(eot_token)

        send_to_val = False
        send_to_test = False
        if args.split_field:
            split_value = item.get(args.split_field, None)
            if split_value == args.val_split:
                send_to_val = True
            elif split_value == args.test_split:
                send_to_test = True
            elif split_value == args.train_split:
                send_to_val = False
            else:
                send_to_val = False
        else:
            if args.val_ratio > 0 and rng.random() < args.val_ratio:
                send_to_val = True

        if send_to_test:
            test_buf.extend(tokens)
            num_test_tokens += len(tokens)
            while len(test_buf) >= args.shard_tokens:
                _save_shard(args.output_dir, args.prefix, "test", test_shard_idx, test_buf[:args.shard_tokens])
                del test_buf[:args.shard_tokens]
                test_shard_idx += 1
        elif send_to_val:
            val_buf.extend(tokens)
            num_val_tokens += len(tokens)
            while len(val_buf) >= args.shard_tokens:
                _save_shard(args.output_dir, args.prefix, "val", val_shard_idx, val_buf[:args.shard_tokens])
                del val_buf[:args.shard_tokens]
                val_shard_idx += 1
        else:
            train_buf.extend(tokens)
            num_train_tokens += len(tokens)
            while len(train_buf) >= args.shard_tokens:
                _save_shard(args.output_dir, args.prefix, "train", train_shard_idx, train_buf[:args.shard_tokens])
                del train_buf[:args.shard_tokens]
                train_shard_idx += 1

        num_docs += 1
        if num_docs % 1000 == 0:
            pbar.set_postfix({
                "train_tokens": num_train_tokens,
                "val_tokens": num_val_tokens,
                "test_tokens": num_test_tokens,
                "skipped": num_skipped,
            })

    if train_buf:
        _save_shard(args.output_dir, args.prefix, "train", train_shard_idx, train_buf)
        train_shard_idx += 1
    if val_buf:
        _save_shard(args.output_dir, args.prefix, "val", val_shard_idx, val_buf)
        val_shard_idx += 1
    if test_buf:
        _save_shard(args.output_dir, args.prefix, "test", test_shard_idx, test_buf)
        test_shard_idx += 1

    print("预处理完成")
    print(f"总样本数: {num_docs}")
    print(f"跳过样本数: {num_skipped}")
    print(f"训练 tokens: {num_train_tokens}")
    print(f"验证 tokens: {num_val_tokens}")
    print(f"测试 tokens: {num_test_tokens}")
    print(f"训练 shards: {train_shard_idx}")
    print(f"验证 shards: {val_shard_idx}")
    print(f"测试 shards: {test_shard_idx}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
