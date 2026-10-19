"""二值描述子训练的组合损失。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hardnet_train.binary_model import BinaryDescriptorFeatures
from hardnet_train.loss import HardNetLoss
from hardnet_train.negative_sampling import (
    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
)


BINARY_METRIC_LOSS_DEFAULTS = {
    "hard_negative_top1_weight": 0.6,
    "positive_tail_loss_weight": 0.05,
    "positive_tail_p95_weight": 0.7,
    "positive_tail_p99_weight": 0.3,
    "positive_tail_p95_target": 0.60,
    "positive_tail_p99_target": 0.80,
}


class BinaryDescriptorLoss(nn.Module):
    """联合优化匹配、teacher 关系保持、量化、bit 均衡与去相关。"""

    def __init__(
        self,
        *,
        margin: float = 0.8,
        hard_negative_strategy: str = "same_finger_allowed",
        hard_negative_top_k: int = 3,
        hard_negative_top1_weight: float = 0.6,
        positive_tail_loss_weight: float = 0.05,
        positive_tail_p95_weight: float = 0.7,
        positive_tail_p99_weight: float = 0.3,
        positive_tail_p95_target: float = 0.60,
        positive_tail_p99_target: float = 0.80,
        same_finger_min_coordinate_separation_px: float = (
            DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX
        ),
        metric_weight: float = 1.0,
        teacher_similarity_weight: float = 0.5,
        positive_consistency_weight: float = 0.1,
        quantization_weight: float = 0.1,
        balance_weight: float = 0.02,
        decorrelation_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.metric_loss = HardNetLoss(
            margin=margin,
            hard_negative_strategy=hard_negative_strategy,
            hard_negative_top_k=hard_negative_top_k,
            hard_negative_top1_weight=hard_negative_top1_weight,
            positive_tail_loss_weight=positive_tail_loss_weight,
            positive_tail_p95_weight=positive_tail_p95_weight,
            positive_tail_p99_weight=positive_tail_p99_weight,
            positive_tail_p95_target=positive_tail_p95_target,
            positive_tail_p99_target=positive_tail_p99_target,
            same_finger_min_coordinate_separation_px=(
                same_finger_min_coordinate_separation_px
            ),
        )
        self.weights = {
            "metric": float(metric_weight),
            "teacher_similarity": float(teacher_similarity_weight),
            "positive_consistency": float(positive_consistency_weight),
            "quantization": float(quantization_weight),
            "balance": float(balance_weight),
            "decorrelation": float(decorrelation_weight),
        }
        negative = {name: value for name, value in self.weights.items() if value < 0.0}
        if negative:
            raise ValueError(f"Binary loss weights must be non-negative, got {negative}.")
        if not any(value > 0.0 for value in self.weights.values()):
            raise ValueError("At least one binary loss weight must be positive.")

    @classmethod
    def from_config(
        cls,
        training_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
    ) -> BinaryDescriptorLoss:
        """从训练与 binary_loss 两个配置段构建组合损失。"""

        return cls(
            margin=float(training_config.get("margin", 0.8)),
            hard_negative_strategy=str(
                training_config.get(
                    "hard_negative_strategy",
                    "same_finger_allowed",
                )
            ),
            hard_negative_top_k=int(
                training_config.get("hard_negative_top_k", 3)
            ),
            hard_negative_top1_weight=float(
                training_config.get(
                    "hard_negative_top1_weight",
                    BINARY_METRIC_LOSS_DEFAULTS["hard_negative_top1_weight"],
                )
            ),
            positive_tail_loss_weight=float(
                training_config.get(
                    "positive_tail_loss_weight",
                    BINARY_METRIC_LOSS_DEFAULTS["positive_tail_loss_weight"],
                )
            ),
            positive_tail_p95_weight=float(
                training_config.get(
                    "positive_tail_p95_weight",
                    BINARY_METRIC_LOSS_DEFAULTS["positive_tail_p95_weight"],
                )
            ),
            positive_tail_p99_weight=float(
                training_config.get(
                    "positive_tail_p99_weight",
                    BINARY_METRIC_LOSS_DEFAULTS["positive_tail_p99_weight"],
                )
            ),
            positive_tail_p95_target=float(
                training_config.get(
                    "positive_tail_p95_target",
                    BINARY_METRIC_LOSS_DEFAULTS["positive_tail_p95_target"],
                )
            ),
            positive_tail_p99_target=float(
                training_config.get(
                    "positive_tail_p99_target",
                    BINARY_METRIC_LOSS_DEFAULTS["positive_tail_p99_target"],
                )
            ),
            same_finger_min_coordinate_separation_px=float(
                training_config.get(
                    "same_finger_min_coordinate_separation_px",
                    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
                )
            ),
            metric_weight=float(loss_config.get("metric_weight", 1.0)),
            teacher_similarity_weight=float(
                loss_config.get("teacher_similarity_weight", 0.5)
            ),
            positive_consistency_weight=float(
                loss_config.get("positive_consistency_weight", 0.1)
            ),
            quantization_weight=float(
                loss_config.get("quantization_weight", 0.1)
            ),
            balance_weight=float(loss_config.get("balance_weight", 0.02)),
            decorrelation_weight=float(
                loss_config.get("decorrelation_weight", 0.02)
            ),
        )

    @staticmethod
    def _relation_loss(
        teacher: torch.Tensor,
        student: torch.Tensor,
    ) -> torch.Tensor:
        """保持 batch 内两两余弦关系，不要求 teacher 与 hash 维度相同。"""

        teacher_unit = F.normalize(teacher.detach().float(), p=2, dim=1)
        student_unit = F.normalize(student.float(), p=2, dim=1)
        teacher_similarity = teacher_unit @ teacher_unit.t()
        student_similarity = student_unit @ student_unit.t()
        squared_error = (student_similarity - teacher_similarity).square()
        count = int(squared_error.shape[0])
        if count <= 1:
            return squared_error.sum() * 0.0
        off_diagonal_sum = squared_error.sum() - squared_error.diagonal().sum()
        return off_diagonal_sum / float(count * (count - 1))

    @staticmethod
    def _decorrelation_loss(values: torch.Tensor) -> torch.Tensor:
        """惩罚 bit 间协方差，避免多个 bit 学到重复判别方向。"""

        values = values.float()
        sample_count, bit_count = values.shape
        if sample_count <= 1 or bit_count <= 1:
            return values.sum() * 0.0
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.t() @ centered / float(sample_count - 1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        return off_diagonal.square().sum() / float(bit_count * (bit_count - 1))

    def forward(
        self,
        anchor: BinaryDescriptorFeatures,
        positive: BinaryDescriptorFeatures,
        point_group: torch.Tensor | None = None,
        finger_group: torch.Tensor | None = None,
        anchor_xy: torch.Tensor | None = None,
        positive_xy: torch.Tensor | None = None,
        anchor_coordinate_frame_group: torch.Tensor | None = None,
        positive_coordinate_frame_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if anchor.logits.shape != positive.logits.shape:
            raise ValueError(
                "anchor/positive binary descriptor shape mismatch: "
                f"{anchor.logits.shape} vs {positive.logits.shape}."
            )
        if anchor.logits.ndim != 2:
            raise ValueError("BinaryDescriptorLoss expects [batch, hash_bits] tensors.")

        anchor_metric = F.normalize(anchor.quantized.float(), p=2, dim=1)
        positive_metric = F.normalize(positive.quantized.float(), p=2, dim=1)
        metric_loss, metric_stats = self.metric_loss(
            anchor_metric,
            positive_metric,
            point_group=point_group,
            finger_group=finger_group,
            anchor_xy=anchor_xy,
            positive_xy=positive_xy,
            anchor_coordinate_frame_group=anchor_coordinate_frame_group,
            positive_coordinate_frame_group=positive_coordinate_frame_group,
        )

        all_teacher = torch.cat([anchor.teacher_f, positive.teacher_f], dim=0)
        all_continuous = torch.cat(
            [anchor.continuous.float(), positive.continuous.float()],
            dim=0,
        )
        teacher_similarity_loss = self._relation_loss(
            all_teacher,
            all_continuous,
        )
        positive_consistency_loss = (
            0.5
            * (1.0 - anchor.continuous.float() * positive.continuous.float())
        ).mean()
        quantization_loss = (1.0 - all_continuous.abs()).square().mean()
        bit_means = all_continuous.mean(dim=0)
        balance_loss = bit_means.square().mean()
        decorrelation_loss = self._decorrelation_loss(all_continuous)

        components = {
            "metric": metric_loss,
            "teacher_similarity": teacher_similarity_loss,
            "positive_consistency": positive_consistency_loss,
            "quantization": quantization_loss,
            "balance": balance_loss,
            "decorrelation": decorrelation_loss,
        }
        total = sum(
            self.weights[name] * component
            for name, component in components.items()
        )
        hard_positive_hamming = (
            0.5
            * (1.0 - anchor.quantized.float() * positive.quantized.float())
        ).mean(dim=1)
        stats = {
            "loss": total.detach(),
            **{
                f"loss_{name}": component.detach()
                for name, component in components.items()
            },
            "positive_hamming": hard_positive_hamming.detach(),
            "mean_abs_bit_balance": bit_means.abs().mean().detach(),
            "mean_abs_continuous": all_continuous.abs().mean().detach(),
            **metric_stats,
        }
        return total, stats
