"""
MoE MLP 模块（Top-k Switch/Router）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mlp import MLP, ReLUMLP, SiLUMLP, SwiGLUMLP, GeGLUMLP


class MoEMLP(nn.Module):
    """
    Top-k Switch/Router MoE
    将每个 token 路由到 top-k 个专家并加权聚合
    """

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_experts = int(getattr(config, "n_experts", 4))
        self.top_k = int(getattr(config, "moe_top_k", 1))
        self.capacity_factor = float(getattr(config, "moe_capacity_factor", 0.0))
        self.router_noise = float(getattr(config, "moe_router_noise", 0.0))
        self.expert_type = str(getattr(config, "moe_expert_type", "mlp")).lower()

        assert self.top_k >= 1 and self.top_k <= self.n_experts, "moe_top_k 必须在 [1, n_experts] 内"

        # 路由器：为每个 token 预测专家分布
        self.router = nn.Linear(self.n_embd, self.n_experts, bias=False)

        # 专家 MLP 列表
        expert_cls = self._resolve_expert_class(self.expert_type)
        self.experts = nn.ModuleList([expert_cls(config) for _ in range(self.n_experts)])

        # 记录最近一次前向的辅助损失
        self.last_aux_loss = None

    @staticmethod
    def _resolve_expert_class(expert_type):
        mapping = {
            "mlp": MLP,
            "relu": ReLUMLP,
            "silu": SiLUMLP,
            "swiglu": SwiGLUMLP,
            "geglu": GeGLUMLP,
        }
        return mapping.get(expert_type, MLP)

    def _compute_aux_loss(self, probs, topk_idx):
        # 负载均衡损失（Switch 风格）
        # probs: [N, E], topk_idx: [N, K]
        n_tokens = probs.size(0)
        n_experts = probs.size(1)
        if n_tokens == 0:
            return probs.new_tensor(0.0)

        importance = probs.sum(dim=0)  # [E]
        load = torch.zeros(n_experts, device=probs.device, dtype=probs.dtype)
        for e in range(n_experts):
            load[e] = (topk_idx == e).sum()

        importance = importance / (importance.sum() + 1e-9)
        load = load / (load.sum() + 1e-9)
        return (importance * load).sum() * n_experts

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            y: [B, T, C]
        """
        bsz, seq_len, dim = x.size()
        x_flat = x.reshape(-1, dim)  # [N, C]

        # 路由 logits & 概率
        logits = self.router(x_flat)
        if self.router_noise > 0 and self.training:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = F.softmax(logits, dim=-1)

        # top-k 选择
        topk_probs, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        if self.top_k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 计算辅助损失
        self.last_aux_loss = self._compute_aux_loss(probs, topk_idx)

        # 容量限制（可选）
        capacity = None
        if self.capacity_factor > 0:
            capacity = int(self.capacity_factor * (x_flat.size(0) / self.n_experts))
            capacity = max(capacity, 1)

        # 聚合输出
        out = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_experts):
            mask = topk_idx == expert_idx  # [N, K]
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=False)
            token_idx = indices[:, 0]
            slot_idx = indices[:, 1]

            if capacity is not None and token_idx.numel() > capacity:
                weights = topk_probs[token_idx, slot_idx]
                top = torch.topk(weights, k=capacity, dim=0)
                keep = top.indices
                token_idx = token_idx[keep]
                slot_idx = slot_idx[keep]
                weights = weights[keep]
            else:
                weights = topk_probs[token_idx, slot_idx]

            x_e = x_flat.index_select(0, token_idx)
            y_e = self.experts[expert_idx](x_e)
            y_e = y_e.to(out.dtype)
            weights = weights.to(out.dtype).unsqueeze(-1)
            out.index_add_(0, token_idx, y_e * weights)

        return out.view(bsz, seq_len, dim)
