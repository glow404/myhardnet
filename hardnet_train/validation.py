"""与训练采样、候选掩码和 top-k 规则对齐的固定验证 batch。"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from hardnet_train.data import (
    FingerprintPairDataset,
    PairRecord,
    allocate_finger_pair_counts,
    build_finger_image_pair_index,
    sample_unique_point_indices,
)
from hardnet_train.loss import HardNetLoss
from hardnet_train.metrics import in_batch_descriptor_validation_metrics


AUTO_VALIDATION_FINGER_COUNT = "auto"
VALIDATION_PLAN_WARNING_POSITIVE_SLOTS = 100_000
VALIDATION_PLAN_MAX_POSITIVE_SLOTS = 10_000_000


def normalize_validation_finger_count(value: Any) -> int | str:
    """把验证手指数规范化为正整数或 `auto`。"""

    text = str(value if value is not None else AUTO_VALIDATION_FINGER_COUNT).strip().lower()
    if text in {"", AUTO_VALIDATION_FINGER_COUNT}:
        return AUTO_VALIDATION_FINGER_COUNT
    try:
        finger_count = int(text)
    except ValueError as exc:
        raise ValueError(
            "validation.finger_count must be a positive integer or 'auto'."
        ) from exc
    if finger_count < 1:
        raise ValueError("validation.finger_count must be >= 1 or 'auto'.")
    return finger_count


@dataclass(frozen=True)
class FixedValidationBatchPlan:
    """训练启动时一次性固化、之后每个 epoch 原样复用的验证 batch 索引。"""

    batches: tuple[tuple[int, ...], ...]
    selected_finger_ids: tuple[str, ...]
    scheduled_image_pair_ids: tuple[tuple[str, ...], ...]
    batch_size: int
    seed: int

    @property
    def finger_count(self) -> int:
        return len(self.selected_finger_ids)

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def positive_count(self) -> int:
        return sum(len(batch) for batch in self.batches)

    @property
    def unique_image_pair_counts(self) -> tuple[int, ...]:
        return tuple(
            len(set(schedule)) for schedule in self.scheduled_image_pair_ids
        )

    @property
    def repeated_image_pair_finger_count(self) -> int:
        return sum(
            unique_count < self.batch_count
            for unique_count in self.unique_image_pair_counts
        )


class FixedValidationBatchSampler(Sampler[list[int]]):
    """按固定顺序重复产出同一组验证 batch。"""

    def __init__(self, plan: FixedValidationBatchPlan) -> None:
        self.plan = plan

    def __len__(self) -> int:
        return self.plan.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.plan.batches:
            yield list(batch)


def _build_repeating_shuffle_schedule(
    values: list[str],
    count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """优先无重复遍历全部值；预算更大时重新洗牌后继续循环。"""

    if not values:
        raise ValueError("Cannot build a validation schedule from empty values.")
    scheduled: list[str] = []
    while len(scheduled) < int(count):
        cycle = list(values)
        rng.shuffle(cycle)
        scheduled.extend(cycle)
    return tuple(scheduled[: int(count)])


def build_fixed_validation_batch_plan(
    records: list[PairRecord],
    *,
    finger_count: int | str,
    batch_count: int,
    batch_size: int,
    seed: int,
) -> FixedValidationBatchPlan:
    """按训练 sampler 的结构构造确定性验证 batch，不预分配负样本。"""

    requested_finger_count = normalize_validation_finger_count(finger_count)
    batch_count = int(batch_count)
    batch_size = int(batch_size)
    seed = int(seed)
    if not records:
        raise ValueError("Fixed validation batch plan requires at least one record.")
    if batch_count < 1:
        raise ValueError("validation.batch_count must be >= 1.")
    if batch_size < 2:
        raise ValueError(
            "validation.batch_size must be >= 2 so an in-batch negative candidate "
            "can exist."
        )
    planned_positive_slots = batch_count * batch_size
    if planned_positive_slots > VALIDATION_PLAN_MAX_POSITIVE_SLOTS:
        raise ValueError(
            "Validation plan is too large: "
            f"batch_count({batch_count}) * batch_size({batch_size}) = "
            f"{planned_positive_slots:,} positive slots, exceeding the safety "
            f"limit {VALIDATION_PLAN_MAX_POSITIVE_SLOTS:,}. Reduce "
            "validation.batch_count or validation.batch_size."
        )
    if planned_positive_slots > VALIDATION_PLAN_WARNING_POSITIVE_SLOTS:
        warnings.warn(
            "Large fixed validation plan: "
            f"{batch_count} batches x {batch_size} pairs = "
            f"{planned_positive_slots:,} positive slots. This normally does not "
            "fail because image pairs are cycled, but validation time and CPU "
            "metric memory grow approximately linearly with this value.",
            RuntimeWarning,
            stacklevel=2,
        )

    groups = build_finger_image_pair_index(records)
    available_fingers = sorted(
        finger_id for finger_id, image_pairs in groups.items() if image_pairs
    )
    if requested_finger_count == AUTO_VALIDATION_FINGER_COUNT:
        finger_count = min(len(available_fingers), batch_size)
    else:
        finger_count = int(requested_finger_count)
    if finger_count < 1:
        raise ValueError(
            "Validation data does not contain any finger with a usable image_pair."
        )
    if finger_count > len(available_fingers):
        raise ValueError(
            f"validation.finger_count={finger_count} exceeds the "
            f"{len(available_fingers)} usable fingers in validation data. Set "
            "validation.finger_count=auto or reduce the configured value."
        )
    if finger_count > batch_size:
        raise ValueError(
            f"validation.finger_count={finger_count} exceeds "
            f"validation.batch_size={batch_size}; every selected finger must "
            "contribute at least one positive pair per batch. Set "
            "validation.finger_count=auto, increase batch_size, or reduce "
            "finger_count."
        )
    if finger_count == 1:
        warnings.warn(
            "Validation uses only one finger, so cross-finger metrics and "
            "different_finger hard negatives are unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    if finger_count == batch_size:
        warnings.warn(
            "validation.finger_count equals validation.batch_size, so every "
            "finger contributes only one positive pair per batch and no "
            "same-finger negative can exist inside that batch.",
            RuntimeWarning,
            stacklevel=2,
        )

    rng = random.Random(seed)
    selected_fingers = tuple(rng.sample(available_fingers, finger_count))
    scheduled_image_pairs = tuple(
        _build_repeating_shuffle_schedule(
            sorted(groups[finger_id]),
            batch_count,
            rng,
        )
        for finger_id in selected_fingers
    )

    batches: list[tuple[int, ...]] = []
    for batch_offset in range(batch_count):
        per_finger_counts = allocate_finger_pair_counts(
            batch_size,
            finger_count,
        )
        rng.shuffle(per_finger_counts)
        batch: list[int] = []
        for finger_offset, (finger_id, pair_count) in enumerate(
            zip(selected_fingers, per_finger_counts)
        ):
            image_pair_id = scheduled_image_pairs[finger_offset][batch_offset]
            batch.extend(
                sample_unique_point_indices(
                    records,
                    groups[finger_id][image_pair_id],
                    pair_count,
                    rng,
                )
            )
        if len(batch) != batch_size:
            raise RuntimeError(
                "Fixed validation batch construction produced an incomplete "
                f"batch: expected {batch_size}, got {len(batch)}."
            )
        rng.shuffle(batch)
        batches.append(tuple(batch))

    return FixedValidationBatchPlan(
        batches=tuple(batches),
        selected_finger_ids=selected_fingers,
        scheduled_image_pair_ids=scheduled_image_pairs,
        batch_size=batch_size,
        seed=seed,
    )


def make_fixed_validation_loader(
    dataset: FingerprintPairDataset,
    plan: FixedValidationBatchPlan,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    """构造按固定 batch plan 读取正样本对及训练元数据的 DataLoader。"""

    worker_count = max(0, int(num_workers))
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": FixedValidationBatchSampler(plan),
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory) and torch.cuda.is_available(),
    }
    if worker_count > 0:
        options["persistent_workers"] = bool(persistent_workers)
        options["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**options)


@dataclass(frozen=True)
class InBatchValidationDistances:
    """一个验证 batch 的正距离、完整候选池及 top-k 排名距离。"""

    positive_dist: torch.Tensor
    candidate_dist: torch.Tensor
    candidate_mask: torch.Tensor
    selected_negative_dist: torch.Tensor
    selected_negative_mask: torch.Tensor
    valid_anchor: torch.Tensor


def detach_validation_distances_to_cpu(
    batch: InBatchValidationDistances,
) -> InBatchValidationDistances:
    """验证批次计算结束后立即释放 GPU 指标张量，只缓存 CPU 副本。"""

    return InBatchValidationDistances(
        positive_dist=batch.positive_dist.detach().cpu(),
        candidate_dist=batch.candidate_dist.detach().cpu(),
        candidate_mask=batch.candidate_mask.detach().cpu(),
        selected_negative_dist=batch.selected_negative_dist.detach().cpu(),
        selected_negative_mask=batch.selected_negative_mask.detach().cpu(),
        valid_anchor=batch.valid_anchor.detach().cpu(),
    )


def summarize_in_batch_validation(
    batches: list[InBatchValidationDistances],
    *,
    margin: float,
    plan: FixedValidationBatchPlan,
) -> dict[str, float]:
    """固定候选池计算 ROC，动态 top-k 只计算 margin loss 诊断。"""

    if not batches:
        raise RuntimeError("Fixed validation loader produced no batches.")

    positive_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    candidate_mask_parts: list[torch.Tensor] = []
    selected_parts: list[torch.Tensor] = []
    selected_mask_parts: list[torch.Tensor] = []
    selected_anchor_parts: list[torch.Tensor] = []
    valid_anchor_count = 0
    skipped_anchor_count = 0

    for batch in batches:
        positive = batch.positive_dist.detach().float().cpu().reshape(-1)
        candidate = batch.candidate_dist.detach().float().cpu()
        candidate_mask = batch.candidate_mask.detach().bool().cpu()
        selected = batch.selected_negative_dist.detach().float().cpu()
        selected_mask = batch.selected_negative_mask.detach().bool().cpu()
        valid_anchor = batch.valid_anchor.detach().bool().cpu().reshape(-1)
        batch_size = int(positive.numel())
        if candidate.ndim != 2 or selected.ndim != 2:
            raise ValueError("Candidate and selected distances must be two-dimensional.")
        if candidate_mask.shape != candidate.shape:
            raise ValueError("Candidate distances and masks must align.")
        if selected_mask.shape != selected.shape:
            raise ValueError("Selected negative distances and masks must align.")
        if valid_anchor.numel() != batch_size:
            raise ValueError("valid_anchor length must match positive_dist length.")

        valid_rows = torch.nonzero(valid_anchor, as_tuple=False).reshape(-1)
        skipped_anchor_count += batch_size - int(valid_rows.numel())
        if valid_rows.numel() == 0:
            continue

        global_anchor_index = torch.arange(
            valid_anchor_count,
            valid_anchor_count + int(valid_rows.numel()),
            dtype=torch.long,
        )
        valid_anchor_count += int(valid_rows.numel())
        positive_parts.append(positive[valid_rows])
        candidate_parts.append(candidate[valid_rows])
        candidate_mask_parts.append(candidate_mask[valid_rows])
        selected_parts.append(selected[valid_rows])
        selected_mask_parts.append(selected_mask[valid_rows])
        selected_anchor_parts.append(
            global_anchor_index[:, None].expand_as(selected[valid_rows])
        )

    if valid_anchor_count == 0:
        raise RuntimeError(
            "Every validation anchor has zero legal in-batch negatives. Increase "
            "validation.finger_count or validation.batch_size, or review the "
            "configured hard-negative strategy and spatial separation."
        )

    positive_dist = torch.cat(positive_parts)
    candidate_dist = torch.cat(candidate_parts)
    candidate_mask = torch.cat(candidate_mask_parts)
    selected_negative_dist = torch.cat(selected_parts)
    selected_negative_mask = torch.cat(selected_mask_parts)
    selected_negative_anchor_index = torch.cat(selected_anchor_parts)
    metrics = in_batch_descriptor_validation_metrics(
        positive_dist=positive_dist,
        candidate_dist=candidate_dist,
        candidate_mask=candidate_mask,
        selected_negative_dist=selected_negative_dist,
        selected_negative_mask=selected_negative_mask,
        selected_negative_anchor_index=selected_negative_anchor_index,
        margin=float(margin),
        valid_anchor_count=valid_anchor_count,
        skipped_anchor_count=skipped_anchor_count,
    )
    metrics.update(
        {
            "batch_count": float(plan.batch_count),
            "configured_finger_count": float(plan.finger_count),
            "configured_batch_count": float(plan.batch_count),
            "configured_batch_size": float(plan.batch_size),
            "planned_positive_count": float(plan.positive_count),
        }
    )
    return metrics


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
def evaluate_fixed_validation_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    plan: FixedValidationBatchPlan,
    criterion: HardNetLoss,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> dict[str, float]:
    """对固定 batch 逐批执行与训练完全相同的候选掩码和 top-k loss。"""

    model.eval()
    distance_batches: list[InBatchValidationDistances] = []
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
            anchor_desc = model(anchor)
            positive_desc = model(positive)
        _, stats = criterion(
            anchor_desc.float(),
            positive_desc.float(),
            **_metadata_to_device(batch, device),
        )
        distance_batches.append(
            detach_validation_distances_to_cpu(
                InBatchValidationDistances(
                    positive_dist=stats["pos_dist"],
                    candidate_dist=stats["candidate_dist"],
                    candidate_mask=stats["candidate_mask"],
                    selected_negative_dist=stats["selected_negative_dist"],
                    selected_negative_mask=stats["selected_negative_mask"],
                    valid_anchor=stats["valid_triplets"],
                )
            )
        )

    return summarize_in_batch_validation(
        distance_batches,
        margin=criterion.margin,
        plan=plan,
    )
