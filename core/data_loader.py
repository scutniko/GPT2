"""
数据加载模块
包含token加载和数据迭代器
"""

import os
import torch
import numpy as np


def _get_token_count(filename):
    """获取 shard 中 token 数量（mmap 方式，避免完整加载）。"""
    npt = np.load(filename, mmap_mode="r")
    if npt.ndim != 1:
        raise ValueError(f"token shard 必须是1维数组: {filename}, got shape={npt.shape}")
    return int(npt.shape[0])


def list_shards(data_root, split):
    """
    列出指定 split 的 shard 文件路径（按文件名排序）。
    """
    if data_root is None:
        raise ValueError("data_root 未设置，请在训练时显式传入 --data_root")
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root 不存在或不是目录: {data_root}")
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    return shards


def validate_shard_lengths(data_root, split, min_tokens_required):
    """
    训练启动前的 fail-fast 校验。
    要求每个 shard 的 token 数量都不少于 min_tokens_required。
    """
    shards = list_shards(data_root, split)
    if len(shards) == 0:
        raise ValueError(f"no shards found for split {split} under {data_root}")

    too_short = []
    min_len = None
    max_len = None
    for shard in shards:
        token_count = _get_token_count(shard)
        if min_len is None or token_count < min_len:
            min_len = token_count
        if max_len is None or token_count > max_len:
            max_len = token_count
        if token_count < min_tokens_required:
            too_short.append((shard, token_count))

    if too_short:
        samples = "\n".join(
            [f"- {path} (tokens={cnt})" for path, cnt in too_short[:10]]
        )
        more = ""
        if len(too_short) > 10:
            more = f"\n... 另外还有 {len(too_short) - 10} 个过短 shard"
        raise ValueError(
            f"split={split} 存在过短 shard（要求 tokens >= {min_tokens_required}）\n"
            f"{samples}{more}"
        )

    return {
        "num_shards": len(shards),
        "min_tokens": min_len,
        "max_tokens": max_len,
    }


def load_tokens(filename):
    """
    加载tokens
    
    Args:
        filename: 文件名
        
    Returns:
        tensor，shape是[num_tokens]
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # 将numpy数组转换为int32类型
    ptt = torch.tensor(npt, dtype=torch.long)  # 将numpy数组转换为torch.long类型
    return ptt


class DataLoaderLite:
    """
    数据加载器
    
    Args:
        B: batch size
        T: 序列长度
        process_rank: 进程排名
        num_processes: 进程数量
        split: 数据集类型 ('train' 或 'val')
        master_process: 是否为主进程
    
    Returns:
        x, y: 输入序列和目标序列
    """
    
    def __init__(self, B, T, process_rank, num_processes, split, master_process=True, data_root=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.min_tokens_required = self.B * self.T * self.num_processes + 1
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = list_shards(data_root, split)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self._shard_len_cache = {}
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def _get_shard_len(self, shard_idx):
        if shard_idx not in self._shard_len_cache:
            self._shard_len_cache[shard_idx] = _get_token_count(self.shards[shard_idx])
        return self._shard_len_cache[shard_idx]

    def _find_next_valid_shard(self, start_idx):
        """找到下一个长度满足要求的 shard。"""
        checked = 0
        invalid = []
        idx = start_idx % len(self.shards)
        while checked < len(self.shards):
            shard_len = self._get_shard_len(idx)
            if shard_len >= self.min_tokens_required:
                return idx
            invalid.append((self.shards[idx], shard_len))
            idx = (idx + 1) % len(self.shards)
            checked += 1

        details = "\n".join([f"- {path} (tokens={cnt})" for path, cnt in invalid[:10]])
        raise ValueError(
            "所有 shard 都过短，无法组成一个 batch。\n"
            f"要求 tokens >= {self.min_tokens_required}\n"
            f"{details}"
        )

    def _load_shard_for_rank(self, shard_idx):
        self.current_shard = shard_idx
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        """重置数据加载器状态"""
        # state, init at shard zero（若过短则自动跳过）
        shard_idx = self._find_next_valid_shard(start_idx=0)
        self._load_shard_for_rank(shard_idx)

    def next_batch(self):
        """获取下一个batch"""
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            shard_idx = self._find_next_valid_shard(start_idx=self.current_shard + 1)
            self._load_shard_for_rank(shard_idx)

        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        if buf.numel() < (B * T + 1):
            shard_idx = self._find_next_valid_shard(start_idx=self.current_shard + 1)
            self._load_shard_for_rank(shard_idx)
            buf = self.tokens[self.current_position : self.current_position+B*T+1]
            if buf.numel() < (B * T + 1):
                raise RuntimeError(
                    "切换 shard 后仍无法取到完整 batch，"
                    f"shard={self.current_shard}, position={self.current_position}, "
                    f"need={B * T + 1}, got={buf.numel()}"
                )

        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            shard_idx = self._find_next_valid_shard(start_idx=self.current_shard + 1)
            self._load_shard_for_rank(shard_idx)
        return x, y

