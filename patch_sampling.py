"""HardNet 统一局部 patch 采样。

训练数据、注册模板、在线 query 和验证流程必须使用同一套坐标语义；否则即使
参数相同，插值次数、边界处理或角度方向不同，也会让 HardNet 输入分布漂移。

路线 2 直接把局部旋转、裁剪和尺寸缩放合并到一次 ``warpAffine``：

    原图 --(局部逆映射，一次 INTER_LINEAR)--> out_size x out_size patch

矩阵只作用于输出 patch 的局部窗口，不再生成整张旋转图。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def overlap_ratio(
    image_shape: tuple[int, int],
    x: float,
    y: float,
    crop_size: int,
) -> float:
    """计算未旋转局部窗口与原图的有效重叠比例。"""

    height, width = image_shape
    half = float(crop_size) / 2.0
    x0, y0, x1, y1 = float(x) - half, float(y) - half, float(x) + half, float(y) + half
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return intersection / max(float(crop_size * crop_size), 1.0)


def extract_local_affine_patch(
    image: np.ndarray,
    x: float,
    y: float,
    angle: float,
    crop_size: int,
    out_size: int,
) -> np.ndarray:
    """围绕关键点一次性采样方向对齐的 HardNet patch。

    ``angle`` 沿用 OpenCV/SIFT 的正角度语义，与原有 ``getRotationMatrix2D``
    路线保持一致。输出像素通过局部逆映射直接从原图采样，因此不再执行“整图
    旋转 + 局部裁剪 + resize”的三段式处理。
    """

    crop_size = int(crop_size)
    out_size = int(out_size)
    if image.ndim != 2:
        raise ValueError(f"patch sampler expects a grayscale image, got {image.shape}")
    if crop_size <= 0 or out_size <= 0:
        raise ValueError(f"crop_size and out_size must be positive, got {crop_size}, {out_size}")

    # warpAffine 的矩阵描述 source -> destination；默认模式会在内部反解，
    # 因此这里把 source 坐标映射到 patch 坐标，避免显式构造整图旋转结果。
    scale = float(crop_size) / float(out_size)
    alpha = float(np.cos(np.deg2rad(float(angle)))) / scale
    beta = float(np.sin(np.deg2rad(float(angle)))) / scale
    center_x = float(x)
    center_y = float(y)
    output_center = (float(out_size) - 1.0) / 2.0
    matrix = np.asarray(
        [
            [alpha, beta, output_center - alpha * center_x - beta * center_y],
            [-beta, alpha, output_center + beta * center_x - alpha * center_y],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        dsize=(out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def extract_keypoint_patch(
    image: np.ndarray,
    keypoint: cv2.KeyPoint,
    crop_size: int,
    out_size: int,
) -> np.ndarray:
    """以 OpenCV ``KeyPoint`` 为输入的统一局部仿射采样入口。"""

    angle = float(keypoint.angle if keypoint.angle >= 0.0 else 0.0)
    return extract_local_affine_patch(
        image,
        x=float(keypoint.pt[0]),
        y=float(keypoint.pt[1]),
        angle=angle,
        crop_size=crop_size,
        out_size=out_size,
    )


def normalize_patch(patch: np.ndarray, normalize: bool = True) -> np.ndarray:
    """按 HardNet 约定把灰度 patch 转为 float32 输入。"""

    values = patch.astype(np.float32)
    if not normalize:
        return values / 255.0
    std = max(float(values.std()), 1e-6)
    return (values - float(values.mean())) / std


def _patch_value(config: Mapping[str, Any], key: str, legacy_key: str, default: Any) -> Any:
    """读取统一 patch 配置，并兼容 pair_build 的旧字段名。"""

    patch_config = config.get("patch", {})
    if not isinstance(patch_config, Mapping):
        return default
    return patch_config.get(key, patch_config.get(legacy_key, default))


def patchable_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    config: Mapping[str, Any],
) -> tuple[list[cv2.KeyPoint], np.ndarray, list[int]]:
    """过滤越界关键点并用统一采样器构建对齐 patch 数组。"""

    crop_size = int(_patch_value(config, "crop_size", "patch_crop_size", 32))
    out_size = int(_patch_value(config, "out_size", "patch_out_size", 32))
    min_overlap = float(_patch_value(config, "min_overlap_ratio", "min_overlap_ratio", 0.55))
    normalize = bool(_patch_value(config, "normalize", "normalize", True))
    selected_keypoints: list[cv2.KeyPoint] = []
    selected_indices: list[int] = []
    patches: list[np.ndarray] = []
    for index, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        if overlap_ratio(image.shape[:2], float(x), float(y), crop_size) < min_overlap:
            continue
        patches.append(
            normalize_patch(
                extract_keypoint_patch(image, keypoint, crop_size, out_size),
                normalize=normalize,
            )
        )
        selected_keypoints.append(keypoint)
        selected_indices.append(index)
    if not patches:
        return [], np.zeros((0, out_size, out_size), dtype=np.float32), []
    return selected_keypoints, np.stack(patches).astype(np.float32), selected_indices