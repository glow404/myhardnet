"""训练与验证指标工具。

作用：
    1. 提供 RunningMean，用于在 epoch 内累计 loss/距离均值。
    2. 提供 FPR@Recall 指标，评估描述子正负距离分布是否分开。

FPR@95TPR 的含义：
    先找一个距离阈值，让 95% 正样本距离都小于该阈值；
    再统计有多少负样本距离也小于该阈值。这个比例越低越好。
"""

from __future__ import annotations

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


def fpr_at_recall(positive_dist: torch.Tensor, negative_dist: torch.Tensor, recall: float = 0.95) -> float:
    """计算 FPR@Recall。

    参数：
        positive_dist:
            正样本距离，越小越相似。
        negative_dist:
            负样本距离，越大越好。
        recall:
            正样本召回率，HardNet/patch descriptor 常用 0.95。
    """
    positives = positive_dist.detach().float().cpu()
    negatives = negative_dist.detach().float().cpu()
    # 过滤无穷大或 NaN，避免异常 batch 影响 quantile。
    positives = positives[torch.isfinite(positives)]
    negatives = negatives[torch.isfinite(negatives)]
    if positives.numel() == 0 or negatives.numel() == 0:
        return 0.0
    # 阈值越大，召回的正样本越多；95% recall 对应正样本距离的 95% 分位点。
    threshold = torch.quantile(positives, float(recall))
    # 负样本距离小于阈值，就会被系统当作匹配对接收，即 false positive。
    return float((negatives <= threshold).float().mean().item())
