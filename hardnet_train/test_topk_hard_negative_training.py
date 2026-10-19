"""Top-k hard-negative training contracts shared by float architectures."""

from __future__ import annotations

import pytest
import torch

from hardnet_train.loss import HardNetLoss, select_hard_negatives
from hardnet_train.model import build_descriptor_model


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
def test_float_architecture_trains_with_top3_hard_negatives(
    architecture: str,
) -> None:
    model = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": 128}
    )
    anchor = model(torch.randn(4, 1, 32, 32))
    positive = model(torch.randn(4, 1, 32, 32))
    groups = torch.arange(4)
    criterion = HardNetLoss(
        hard_negative_strategy="different_finger",
        hard_negative_top_k=3,
        hard_negative_top1_weight=0.6,
        positive_tail_loss_weight=0.0,
    )

    loss, stats = criterion(
        anchor,
        positive,
        point_group=groups,
        finger_group=groups,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["selected_negative_dist"].shape == (4, 3)
    assert torch.all(stats["selected_negative_mask"])
    assert torch.all(stats["hard_negative_count"] == 3)
    assert torch.all(
        stats["selected_negative_dist"][:, :-1]
        <= stats["selected_negative_dist"][:, 1:]
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_topk_selection_uses_available_candidate_count() -> None:
    anchor = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    positive = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    groups = torch.arange(2)

    selection = select_hard_negatives(
        anchor,
        positive,
        hard_negative_strategy="different_finger",
        hard_negative_top_k=10,
        same_finger_min_coordinate_separation_px=16,
        point_group=groups,
        finger_group=groups,
    )

    # 双向候选矩阵宽度为 2B；k 不得超过候选矩阵宽度。
    assert selection.topk_negative_dist.shape == (2, 4)
    assert torch.all(selection.negative_count == 2)


def test_top1_keeps_configured_weight_when_only_two_candidates_are_valid() -> None:
    torch.manual_seed(7)
    anchor = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    positive = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    groups = torch.arange(2)
    criterion = HardNetLoss(
        margin=2.0,
        hard_negative_strategy="different_finger",
        hard_negative_top_k=3,
        hard_negative_top1_weight=0.6,
        positive_tail_loss_weight=0.0,
    )

    loss, stats = criterion(
        anchor,
        positive,
        point_group=groups,
        finger_group=groups,
    )
    violations = torch.clamp(
        criterion.margin
        + stats["pos_dist"][:, None]
        - stats["selected_negative_dist"][:, :2],
        min=0.0,
    )
    expected = (0.6 * violations[:, 0] + 0.4 * violations[:, 1]).mean()

    torch.testing.assert_close(loss, expected)
