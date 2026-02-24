"""
数据加载模块
包含token加载和数据迭代器
"""

import os
import torch
import numpy as np


def _is_split_shard(filename, split):
    """
    严格判断 shard 是否属于某个 split。
    规则：文件名（不含扩展名）按 "_" / "-" 分词后包含 split。
    例如 dataset_train_000001.npy、my-val-000123.npy。
    """
    stem, ext = os.path.splitext(os.path.basename(filename).lower())
    if ext != ".npy":
        return False
    tokens = stem.replace("-", "_").split("_")
    return split in tokens


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
        assert split in {'train', 'val'}

        # get the shard filenames
        if data_root is None:
            raise ValueError("data_root 未设置，请在训练时显式传入 --data_root")
        shards = os.listdir(data_root)
        shards = [s for s in shards if _is_split_shard(s, split)]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def _advance_shard(self):
        """切换到下一个shard并重置当前位置。"""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _ensure_current_shard_has_batch(self):
        """确保当前shard可为当前rank提供至少一个完整batch。"""
        need = self.B * self.T + 1
        visited = 0
        while self.current_position + need > len(self.tokens):
            self._advance_shard()
            visited += 1
            if visited > len(self.shards):
                raise RuntimeError(
                    "所有 shard 对当前配置都不足以提供一个完整 batch，"
                    "请检查 shard 大小或减小 B/T。"
                )

    def reset(self):
        """重置数据加载器状态"""
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        self._ensure_current_shard_has_batch()

    def next_batch(self):
        """获取下一个batch"""
        B, T = self.B, self.T
        self._ensure_current_shard_has_batch()
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self._advance_shard()
            self._ensure_current_shard_has_batch()
        return x, y

