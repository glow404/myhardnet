"""固定验证 batch 上的二值描述子 Hamming 验证。"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hardnet_train.binary_model import BinaryDescriptorModel
from hardnet_train.loss import select_hard_negatives
from hardnet_train.validation import (
    FixedValidationBatchPlan,
    InBatchValidationDistances,
    detach_validation_distances_to_cpu,
    summarize_in_batch_validation,
)


def pair_normalized_hamming(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算末维 0/1 码的一一对应归一化 Hamming 距离。"""

    if left.shape[-1] != right.shape[-1]:
        raise ValueError(
            f"Binary descriptor dimension mismatch: {left.shape} vs {right.shape}"
        )
    return left.ne(right).float().mean(dim=-1)


def _metadata_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in (
            "point_group",
            "finger_group",
            "anchor_xy",
            "positive_xy",
            "anchor_coordinate_frame_group",
            "positive_coordinate_frame_group",
        )
    }


@torch.inference_mode()
def evaluate_binary_fixed_validation_batches(
    model: BinaryDescriptorModel,
    loader: DataLoader,
    plan: FixedValidationBatchPlan,
    device: torch.device,
    margin: float,
    hard_negative_strategy: str,
    hard_negative_top_k: int,
    same_finger_min_coordinate_separation_px: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> dict[str, float]:
    """用训练同款候选掩码/top-k 选择，并以真实 Hamming 距离计分。"""

    model.eval()
    distance_batches: list[InBatchValidationDistances] = []
    all_codes: list[torch.Tensor] = []
    for batch in loader:
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        if channels_last:
            anchor = anchor.contiguous(memory_format=torch.channels_last)
            positive = positive.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            anchor_codes = model.encode_binary(anchor, packed=False).bool()
            positive_codes = model.encode_binary(positive, packed=False).bool()

        # 二值训练的 metric loss 使用归一化后的 {-1, +1} code。其 L2
        # 排序与 Hamming 距离单调一致，因此候选索引与训练语义相同。
        anchor_metric = F.normalize(
            anchor_codes.float().mul(2.0).sub(1.0),
            p=2,
            dim=1,
        )
        positive_metric = F.normalize(
            positive_codes.float().mul(2.0).sub(1.0),
            p=2,
            dim=1,
        )
        selection = select_hard_negatives(
            anchor_metric,
            positive_metric,
            hard_negative_strategy=hard_negative_strategy,
            hard_negative_top_k=hard_negative_top_k,
            same_finger_min_coordinate_separation_px=(
                same_finger_min_coordinate_separation_px
            ),
            **_metadata_to_device(batch, device),
        )

        pairwise_hamming = anchor_codes[:, None, :].ne(
            positive_codes[None, :, :]
        ).float().mean(dim=-1)
        bidirectional_hamming = torch.cat(
            [pairwise_hamming, pairwise_hamming.t()],
            dim=1,
        )
        selected_hamming = torch.gather(
            bidirectional_hamming,
            1,
            selection.topk_candidate_index,
        )
        distance_batches.append(
            detach_validation_distances_to_cpu(
                InBatchValidationDistances(
                    positive_dist=pairwise_hamming.diag(),
                    candidate_dist=bidirectional_hamming,
                    candidate_mask=selection.valid_candidate,
                    selected_negative_dist=selected_hamming,
                    selected_negative_mask=selection.valid_topk,
                    valid_anchor=selection.valid_anchor,
                )
            )
        )
        all_codes.extend(
            [anchor_codes.detach().cpu(), positive_codes.detach().cpu()]
        )

    metrics = summarize_in_batch_validation(
        distance_batches,
        margin=float(margin),
        plan=plan,
    )
    codes = torch.cat(all_codes, dim=0)
    bit_one_fraction = codes.float().mean(dim=0)
    metrics.update(
        {
            "bit_one_fraction": float(bit_one_fraction.mean().item()),
            "bit_balance_error": float(
                (bit_one_fraction - 0.5).abs().mean().item()
            ),
            "constant_bit_ratio": float(
                ((bit_one_fraction == 0.0) | (bit_one_fraction == 1.0))
                .float()
                .mean()
                .item()
            ),
        }
    )
    return metrics
