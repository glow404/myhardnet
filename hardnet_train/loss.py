"""HardNet 的 top-k hardest-in-batch triplet loss。

作用：
    给定一个 batch 的正样本 patch 对 `(A_i, P_i)`，网络分别输出 anchor
    描述子 `a_i` 和 positive 描述子 `p_i`。本文件计算 batch 内所有
    `a_i` 与 `p_j` 的距离矩阵，并为每个正样本寻找 top-k 难负样本。

为什么需要 point_group：
    指纹数据中同一个真实物理点可能出现在多张图、多条 CSV 正样本记录里。
    如果不屏蔽同一物理点，hardest negative 很容易选到“其实也是正样本”的
    伪负样本，导致网络一边被要求拉近、一边又被要求推远。
"""

from __future__ import annotations

import torch
from torch import nn


HARD_NEGATIVE_STRATEGY_ALIASES = {
    "same_finger_allowed": "same_finger_allowed",
    "point_group": "same_finger_allowed",
    "connected_point": "same_finger_allowed",
    "different_finger": "different_finger",
    "cross_finger_only": "different_finger",
}


def normalize_hard_negative_strategy(strategy: str | None) -> str:
    """规范化 hardest negative 候选策略名称。"""

    if strategy is None:
        return "same_finger_allowed"
    key = str(strategy).strip().lower()
    if not key:
        return "same_finger_allowed"
    if key not in HARD_NEGATIVE_STRATEGY_ALIASES:
        supported = ", ".join(sorted(HARD_NEGATIVE_STRATEGY_ALIASES))
        raise ValueError(f"Unsupported hard negative strategy: {strategy!r}. Supported: {supported}")
    return HARD_NEGATIVE_STRATEGY_ALIASES[key]


def pairwise_l2_for_unit_vectors(anchor: torch.Tensor, positive: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算两组单位向量描述子之间的两两 L2 距离。

    对单位向量 x、y，有：
        ||x - y||_2 = sqrt(2 - 2 * dot(x, y))

    这样比显式广播相减更省显存，尤其 batch_size 较大时很重要。
    """
    similarity = anchor @ positive.t()
    distance_sq = torch.clamp(2.0 - 2.0 * similarity, min=eps)
    return torch.sqrt(distance_sq)


class HardNetLoss(nn.Module):
    """HardNet 的 top-k hardest-in-batch triplet margin loss。"""

    def __init__(
        self,
        margin: float = 1.0,
        hard_negative_strategy: str = "same_finger_allowed",
        hard_negative_top_k: int = 3,
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_strategy = normalize_hard_negative_strategy(hard_negative_strategy)
        self.hard_negative_top_k = int(hard_negative_top_k)
        if self.hard_negative_top_k < 1:
            raise ValueError(f"hard_negative_top_k must be >= 1, got {hard_negative_top_k!r}.")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        point_group: torch.Tensor | None = None,
        finger_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算一个 batch 的 HardNet loss。

        参数：
            anchor:
                anchor 分支输出的描述子，形状 `[B, 128]`。
            positive:
                positive 分支输出的描述子，形状 `[B, 128]`。
            point_group:
                每条正样本对应的物理点组编号。同组样本不能互相当负样本。
            finger_group:
                每条正样本对应的手指编号。`different_finger` 策略下，同一手指
                的所有候选都会被屏蔽。

        返回：
            loss:
                标量损失。
            stats:
                训练日志使用的正样本距离、top-k 负样本平均距离、有效样本 mask
                以及每条样本实际选中的负样本数量。
        """
        if anchor.shape != positive.shape:
            raise ValueError(f"anchor/positive shape mismatch: {anchor.shape} vs {positive.shape}")
        if anchor.ndim != 2:
            raise ValueError("HardNetLoss expects descriptor tensors shaped [batch, descriptor_dim].")

        batch_size = anchor.size(0)
        # distances[i, j] = d(a_i, p_j)。对角线 i==j 是正样本距离。
        distances = pairwise_l2_for_unit_vectors(anchor, positive)
        positive_dist = distances.diag()

        # i==j 必须屏蔽，因为它是正样本，不是负样本。
        invalid = torch.eye(batch_size, dtype=torch.bool, device=distances.device)
        if self.hard_negative_strategy == "different_finger":
            if finger_group is None:
                raise ValueError("finger_group is required when hard_negative_strategy='different_finger'.")
            group = finger_group.to(device=distances.device).view(-1)
            if group.numel() != batch_size:
                raise ValueError("finger_group length must match batch size.")
            # 策略一：最难负样本只能来自不同手指。
            invalid = invalid | group[:, None].eq(group[None, :])

        if point_group is not None:
            group = point_group.to(device=distances.device).view(-1)
            if group.numel() != batch_size:
                raise ValueError("point_group length must match batch size.")
            # 策略二依赖这个屏蔽：A-B、B-C 会经 union-find 合并，
            # 即使表里没有 A-C，也不能把 A-C 当作负样本。
            invalid = invalid | group[:, None].eq(group[None, :])

        # 用一个很大的距离替换无效候选，这样 top-k 时不会优先选到它们。
        large_value = torch.finfo(distances.dtype).max / 16.0

        # 保留原来的双向难负样本定义：
        #   1. 对 anchor a_i，候选是所有非匹配 positive p_j；
        #   2. 对 positive p_i，候选是所有非匹配 anchor a_j。
        # 合并两个方向后取距离最小的 k 个。k=1 时与原来的 hardest-negative 等价。
        anchor_candidates = distances.masked_fill(invalid, large_value)
        positive_candidates = distances.t().masked_fill(invalid.t(), large_value)
        negative_candidates = torch.cat([anchor_candidates, positive_candidates], dim=1)
        selected_k = min(self.hard_negative_top_k, negative_candidates.size(1))
        topk_negative_dist = torch.topk(
            negative_candidates,
            k=selected_k,
            dim=1,
            largest=False,
            sorted=True,
        ).values

        # 若合法候选少于配置的 k，只使用实际存在的候选，不让填充值参与损失。
        valid_topk = topk_negative_dist < large_value / 2.0
        negative_count = valid_topk.sum(dim=1)
        valid = negative_count > 0
        negative_sum = torch.where(valid_topk, topk_negative_dist, torch.zeros_like(topk_negative_dist)).sum(dim=1)
        mean_negative_dist = negative_sum / negative_count.clamp_min(1)
        mean_negative_dist = torch.where(
            valid,
            mean_negative_dist,
            torch.full_like(mean_negative_dist, large_value),
        )

        # 极端情况下，一个 batch 里所有候选都被屏蔽，会没有有效负样本。
        if not torch.any(valid):
            zero = positive_dist.sum() * 0.0
            stats = {
                "pos_dist": positive_dist.detach(),
                "neg_dist": mean_negative_dist.detach(),
                "valid_triplets": valid.detach(),
                "hard_negative_count": negative_count.detach(),
            }
            return zero, stats

        # 每个 top-k 负样本各自计算 triplet margin loss；先在单条正样本内部
        # 对有效负样本求均值，再在有效正样本间求均值，避免候选较少的样本权重变低。
        per_negative = torch.clamp(
            self.margin + positive_dist[:, None] - topk_negative_dist,
            min=0.0,
        )
        per_sample = torch.where(valid_topk, per_negative, torch.zeros_like(per_negative)).sum(dim=1)
        per_sample = per_sample / negative_count.clamp_min(1)
        loss = per_sample[valid].mean()
        stats = {
            "pos_dist": positive_dist.detach(),
            "neg_dist": mean_negative_dist.detach(),
            "valid_triplets": valid.detach(),
            "hard_negative_count": negative_count.detach(),
        }
        return loss, stats
