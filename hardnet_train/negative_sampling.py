"""同指负样本的空间候选规则。"""

from __future__ import annotations

import math

import torch


DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX = 16.0


def normalize_min_coordinate_separation(value: float | int) -> float:
    """规范化像素间隔，并拒绝负数或非有限值。"""

    separation = float(value)
    if not math.isfinite(separation) or separation < 0.0:
        raise ValueError(
            "same_finger_min_coordinate_separation_px must be a finite value >= 0, "
            f"got {value!r}."
        )
    return separation


def is_outside_square_neighborhood(
    anchor_xy: tuple[float, float],
    candidate_xy: tuple[float, float],
    min_coordinate_separation_px: float,
) -> bool:
    """判断候选是否位于锚点中心方形邻域之外。

    规则为 ``max(abs(dx), abs(dy)) >= threshold``。因此阈值为 16 时，
    只屏蔽横纵坐标差都小于 16px 的候选；任意一轴达到 16px 即可作为负样本。
    """

    separation = normalize_min_coordinate_separation(
        min_coordinate_separation_px
    )
    dx = abs(float(candidate_xy[0]) - float(anchor_xy[0]))
    dy = abs(float(candidate_xy[1]) - float(anchor_xy[1]))
    return max(dx, dy) >= separation


def same_finger_nearby_mask(
    point_xy: torch.Tensor,
    finger_group: torch.Tensor,
    coordinate_frame_group: torch.Tensor,
    min_coordinate_separation_px: float,
) -> torch.Tensor:
    """返回批内应屏蔽的同指空间近邻矩阵。

    `point_xy[i]` 是第 i 个正样本物理点在统一参考图中的坐标。
    仅同一手指且使用同一参考图坐标系的点才比较坐标；不同参考图或跨指候选
    不受此规则影响。
    """

    separation = normalize_min_coordinate_separation(
        min_coordinate_separation_px
    )
    if point_xy.ndim != 2 or point_xy.shape[1] != 2:
        raise ValueError(
            "point_xy must have shape [batch, 2], "
            f"got {tuple(point_xy.shape)}."
        )
    batch_size = int(point_xy.shape[0])
    finger = finger_group.to(device=point_xy.device).view(-1)
    frame = coordinate_frame_group.to(device=point_xy.device).view(-1)
    if finger.numel() != batch_size:
        raise ValueError("finger_group length must match point_xy batch size.")
    if frame.numel() != batch_size:
        raise ValueError(
            "coordinate_frame_group length must match point_xy batch size."
        )
    if separation == 0.0:
        return torch.zeros(
            (batch_size, batch_size),
            dtype=torch.bool,
            device=point_xy.device,
        )

    coordinate_gap = (
        point_xy.float()[:, None, :] - point_xy.float()[None, :, :]
    ).abs().amax(dim=-1)
    same_finger = finger[:, None].eq(finger[None, :])
    same_frame = frame[:, None].eq(frame[None, :])
    return same_finger & same_frame & coordinate_gap.lt(separation)