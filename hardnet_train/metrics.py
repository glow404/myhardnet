"""训练运行均值与固定描述子验证指标。"""

from __future__ import annotations

import math

import torch


class RunningMean:
    """按样本数加权的运行均值。"""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        """加入一个均值观测值及其对应样本数。"""
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def value(self) -> float:
        """返回当前累计均值；没有观测值时返回 0。"""
        if self.count == 0:
            return 0.0
        return self.total / self.count


def pair_l2_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算末维单位描述子的一一对应 L2 距离，支持前置维度广播。"""
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(f"Descriptor dimension mismatch: {left.shape} vs {right.shape}")
    similarity = (left.float() * right.float()).sum(dim=-1)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))


def _finite_flatten(values: torch.Tensor, name: str) -> torch.Tensor:
    flattened = values.detach().float().cpu().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not torch.all(torch.isfinite(flattened)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return flattened


def _binary_roc(positive_dist: torch.Tensor, negative_dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回按距离阈值递增的 `(FPR, TPR)`，相同距离按同一阈值处理。"""
    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    distances = torch.cat([positives, negatives])
    labels = torch.cat(
        [
            torch.ones(positives.numel(), dtype=torch.int64),
            torch.zeros(negatives.numel(), dtype=torch.int64),
        ]
    )
    order = torch.argsort(distances, stable=True)
    sorted_distances = distances[order]
    sorted_labels = labels[order]

    # 每组相同距离只在组尾生成 ROC 点，避免人为规定并列样本先后顺序。
    group_end = torch.ones(sorted_distances.numel(), dtype=torch.bool)
    group_end[:-1] = sorted_distances[:-1] != sorted_distances[1:]
    true_positives = sorted_labels.cumsum(dim=0)[group_end].double()
    false_positives = (1 - sorted_labels).cumsum(dim=0)[group_end].double()
    tpr = true_positives / float(positives.numel())
    fpr = false_positives / float(negatives.numel())
    return torch.cat([torch.zeros(1, dtype=torch.double), fpr]), torch.cat(
        [torch.zeros(1, dtype=torch.double), tpr]
    )


def _eer(fpr: torch.Tensor, tpr: torch.Tensor) -> float:
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossing = torch.nonzero(difference >= 0.0, as_tuple=False)
    if crossing.numel() == 0:
        return float((fpr[-1] + fnr[-1]).item() / 2.0)
    right = int(crossing[0].item())
    if right == 0:
        return float((fpr[0] + fnr[0]).item() / 2.0)
    left = right - 1
    left_difference = float(difference[left].item())
    right_difference = float(difference[right].item())
    denominator = right_difference - left_difference
    weight = 0.0 if denominator == 0.0 else -left_difference / denominator
    interpolated_fpr = float(fpr[left].item()) + weight * float((fpr[right] - fpr[left]).item())
    interpolated_fnr = float(fnr[left].item()) + weight * float((fnr[right] - fnr[left]).item())
    return (interpolated_fpr + interpolated_fnr) / 2.0


def _distance_metrics(positive_dist: torch.Tensor, negative_dist: torch.Tensor) -> dict[str, float]:
    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    fpr, tpr = _binary_roc(positives, negatives)

    at_tpr95 = torch.nonzero(tpr >= 0.95, as_tuple=False)
    fpr_at_tpr95 = float(fpr[int(at_tpr95[0].item())].item()) if at_tpr95.numel() else 1.0
    allowed_fpr = fpr <= 1e-4
    tpr_at_fpr_1e_4 = float(tpr[allowed_fpr].max().item()) if torch.any(allowed_fpr) else 0.0
    auc = float(torch.trapz(tpr, fpr).item())

    return {
        "pos_mean": float(positives.mean().item()),
        "fpr_at_tpr95": fpr_at_tpr95,
        "tpr_at_fpr_1e_4": tpr_at_fpr_1e_4,
        "roc_auc": auc,
        "eer": _eer(fpr, tpr),
        "neg_mean": float(negatives.mean().item()),
        "neg_p01": float(torch.quantile(negatives, 0.01).item()),
        "neg_p05": float(torch.quantile(negatives, 0.05).item()),
    }


def _selected_topk_metrics(
    positive_dist: torch.Tensor,
    negative_dist: torch.Tensor,
    negative_anchor_index: torch.Tensor,
    margin: float,
) -> dict[str, float]:
    """按每个 anchor 实际选中的可变数量 top-k 候选计算 loss 诊断。"""

    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    anchor_index = negative_anchor_index.detach().long().cpu().reshape(-1)
    if anchor_index.numel() != negatives.numel():
        raise ValueError("negative_anchor_index must match negative_dist length.")
    if torch.any(anchor_index < 0) or torch.any(anchor_index >= positives.numel()):
        raise ValueError("negative_anchor_index contains an out-of-range anchor.")

    count = torch.zeros(positives.numel(), dtype=torch.long)
    count.index_add_(0, anchor_index, torch.ones_like(anchor_index))
    included = count > 0
    if not torch.any(included):
        return {
            "active_triplet_ratio": math.nan,
            "loss": math.nan,
        }

    positive_for_negative = positives[anchor_index]
    violations = torch.clamp(
        float(margin) + positive_for_negative - negatives,
        min=0.0,
    )
    violation_sum = torch.zeros(positives.numel(), dtype=torch.float32)
    violation_sum.index_add_(0, anchor_index, violations)
    per_anchor_loss = violation_sum / count.clamp_min(1)

    active_count = torch.zeros(positives.numel(), dtype=torch.long)
    active_count.index_add_(0, anchor_index, violations.gt(0.0).long())
    return {
        "active_triplet_ratio": float(
            active_count[included].gt(0).float().mean().item()
        ),
        "loss": float(per_anchor_loss[included].mean().item()),
    }


def in_batch_descriptor_validation_metrics(
    positive_dist: torch.Tensor,
    candidate_dist: torch.Tensor,
    candidate_mask: torch.Tensor,
    selected_negative_dist: torch.Tensor,
    selected_negative_mask: torch.Tensor,
    selected_negative_anchor_index: torch.Tensor,
    margin: float,
    *,
    valid_anchor_count: int,
    skipped_anchor_count: int,
) -> dict[str, float]:
    """用完整合法候选池计算总体指标，用动态 top-k 子集计算 loss 诊断。"""

    positives = _finite_flatten(positive_dist, "positive_dist")
    if candidate_dist.shape != candidate_mask.shape:
        raise ValueError("Candidate distances and masks must align.")
    if selected_negative_dist.shape != selected_negative_mask.shape:
        raise ValueError("Selected negative distances and masks must align.")
    candidates = _finite_flatten(candidate_dist[candidate_mask], "candidate_dist")
    selected = _finite_flatten(
        selected_negative_dist[selected_negative_mask],
        "selected_negative_dist",
    )
    selected_anchor_index = (
        selected_negative_anchor_index[selected_negative_mask]
        .detach()
        .long()
        .cpu()
        .reshape(-1)
    )
    if positives.numel() != int(valid_anchor_count):
        raise ValueError(
            "positive_dist length must equal valid_anchor_count: "
            f"{positives.numel()} vs {valid_anchor_count}."
        )
    if selected_anchor_index.numel() != selected.numel():
        raise ValueError("Selected negative anchor indices must match selected distances.")

    result = {
        "pos_mean": float(positives.mean().item()),
        "pos_p95": float(torch.quantile(positives, 0.95).item()),
        "pos_p99": float(torch.quantile(positives, 0.99).item()),
        "valid_anchor_count": float(valid_anchor_count),
        "skipped_anchor_count": float(skipped_anchor_count),
        "valid_anchor_ratio": float(valid_anchor_count)
        / float(valid_anchor_count + skipped_anchor_count),
    }
    result.update(_distance_metrics(positives, candidates))
    result.update(
        _selected_topk_metrics(
            positives,
            selected,
            selected_anchor_index,
            margin,
        )
    )
    return result


def fpr_at_recall(positive_dist: torch.Tensor, negative_dist: torch.Tensor, recall: float = 0.95) -> float:
    """兼容旧调用的 FPR@TPR；新日志字段使用明确的 `fpr_at_tpr95`。"""
    if not 0.0 <= float(recall) <= 1.0:
        raise ValueError(f"recall must be in [0, 1], got {recall}.")
    try:
        fpr, tpr = _binary_roc(positive_dist, negative_dist)
    except ValueError:
        return math.nan
    indices = torch.nonzero(tpr >= float(recall), as_tuple=False)
    return float(fpr[int(indices[0].item())].item()) if indices.numel() else 1.0