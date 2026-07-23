"""SIFT/RANSAC/几何扩展正样本构建阶段。

作用：
- 读取 split_assignments.csv，在同一 split、同一 finger_id 内枚举图像对。
- 对每张原始小局部指纹图提取 SIFT/RootSIFT 特征并缓存。
- 对每个 image_pair 执行：
  1. SIFT descriptor ratio 匹配，得到 seed matches
  2. RANSAC 估计 A->B 的仿射/部分仿射变换
  3. 保留 RANSAC 内点作为 `source=sift_inlier`
  4. 使用几何矩阵 T 将 A 图所有 keypoint 投影到 B 图，扩展出
     `source=geometry_expanded` 正样本
  5. 根据投影误差、方向、尺度、纹理、响应等指标打分、去重、截断到 K

输出：
- all_positive_pairs.csv：HardNet 训练所需的正样本 pair 元数据。
- match_diagnostics.csv：每个 image_pair 的匹配数量、RANSAC 内点、失败原因。
- positive_build_summary.json：总体统计。

重要约束：
- 第一版不维护全局 physical_point_id。
- 通过 `image_pair_id`、`correspondence_id` 和每对图最多 K 个样本，为后续 batch sampler
  降低伪负样本风险提供元数据基础。
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import (
    angle_diff_mod180,
    clamp,
    ensure_dir,
    exp_score,
    get_nested,
    imread_grayscale,
    read_csv_rows,
    read_json,
    resolve_path,
    stable_id,
    write_json,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


POSITIVE_FIELDNAMES = [
    # 训练主表字段。后续 patch_extractor 会复用这份字段顺序，
    # 因此新增字段时要同步考虑 patch_extractor 和训练 Dataset。
    "pair_id",
    "correspondence_id",
    "finger_id",
    "image_pair_id",
    "split",
    "source",
    "image_a_path",
    "image_b_path",
    "image_a_id",
    "image_b_id",
    "kp_a_idx",
    "kp_b_idx",
    "x_a",
    "y_a",
    "size_a",
    "angle_a",
    "response_a",
    "x_b",
    "y_b",
    "size_b",
    "angle_b",
    "response_b",
    "descriptor_dist",
    "ratio",
    "ransac_inlier",
    "reproj_error",
    "angle_diff_mod180",
    "scale_ratio",
    "stability_score",
    "transform_matrix",
    "transform_rotation_deg",
    "patch_a_path",
    "patch_p_path",
]


DIAGNOSTIC_FIELDNAMES = [
    # 每个 image_pair 一行，用于 QA 观察哪些图像对匹配失败、失败在哪个阶段。
    "split",
    "finger_id",
    "image_pair_id",
    "image_a_id",
    "image_b_id",
    "image_a_path",
    "image_b_path",
    "keypoints_a",
    "keypoints_b",
    "ratio_kept",
    "mutual_kept",
    "ransac_inliers",
    "seed_candidates",
    "expanded_candidates",
    "final_pairs",
    "mean_reproj_error",
    "transform_rotation_deg",
    "transform_matrix",
    "skip_reason",
]


def _build_sift(config: dict[str, Any]) -> cv2.SIFT:
    """根据配置创建 OpenCV SIFT 实例。"""

    return cv2.SIFT_create(
        nfeatures=int(get_nested(config, "sift", "nfeatures", default=300)),
        nOctaveLayers=int(get_nested(config, "sift", "nOctaveLayers", default=4)),
        contrastThreshold=float(get_nested(config, "sift", "contrastThreshold", default=0.03)),
        edgeThreshold=float(get_nested(config, "sift", "edgeThreshold", default=17.5)),
        sigma=float(get_nested(config, "sift", "sigma", default=1.70)),
    )


def _preprocess_image(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """提 SIFT 前的轻量预处理。

    CLAHE 用于增强指纹脊线对比度；轻微 blur 用于降低噪声点对关键点定位和方向的影响。
    """

    output = image
    if bool(get_nested(config, "sift", "enable_clahe", default=True)):
        tile = get_nested(config, "sift", "clahe_tile", default=[8, 8])
        clahe = cv2.createCLAHE(clipLimit=float(get_nested(config, "sift", "clahe_clip", default=4.0)), tileGridSize=(int(tile[0]), int(tile[1])))
        output = clahe.apply(output)
    if bool(get_nested(config, "sift", "enable_blur", default=True)):
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=float(get_nested(config, "sift", "blur_sigma", default=0.8)))
    return output


def _rootsift(descriptors: np.ndarray | None) -> np.ndarray:
    """把 SIFT descriptor 转成 RootSIFT。

    RootSIFT = L1 normalize + sqrt + L2 normalize，通常能提升 L2 距离匹配稳定性。
    """

    if descriptors is None or len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    values = descriptors.astype(np.float32)
    values /= values.sum(axis=1, keepdims=True) + 1e-12
    values = np.sqrt(values)
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    return values.astype(np.float32)


def _serialize_keypoints(keypoints: tuple[cv2.KeyPoint, ...] | list[cv2.KeyPoint]) -> list[dict[str, float | int]]:
    """把 OpenCV KeyPoint 转成可序列化字典。

    后续所有计算都使用普通 dict，避免跨阶段保存 OpenCV 对象。
    """

    return [
        {
            "x": float(kp.pt[0]),
            "y": float(kp.pt[1]),
            "size": float(kp.size),
            "angle": float(kp.angle if kp.angle >= 0 else 0.0),
            "response": float(kp.response),
            "octave": int(kp.octave),
            "class_id": int(kp.class_id),
        }
        for kp in keypoints
    ]


def _extract_features(
    row: dict[str, str],
    config: dict[str, Any],
    sift: cv2.SIFT,
    cache: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    """读取图像并提取 SIFT/RootSIFT 特征。

    feature_cache 以 image_path 为 key，确保同一张图在大量两两组合中只提一次特征。
    """

    path = row["image_path"]
    if path in cache:
        return cache[path]

    image = imread_grayscale(path)
    if image is None:
        logger.warning("图像读取失败，跳过特征: %s", path)
        feature = {"ok": False, "image": None, "keypoints": [], "descriptors": np.zeros((0, 128), dtype=np.float32)}
        cache[path] = feature
        return feature

    processed = _preprocess_image(image, config)
    keypoints, descriptors = sift.detectAndCompute(processed, None)
    descriptors = _rootsift(descriptors) if bool(get_nested(config, "sift", "enable_rootsift", default=True)) else (descriptors if descriptors is not None else np.zeros((0, 128), dtype=np.float32))
    feature = {
        "ok": True,
        "image": image,
        "keypoints": _serialize_keypoints(keypoints),
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "response_p90": float(np.percentile([kp.response for kp in keypoints], 90)) if keypoints else 1.0,
    }
    cache[path] = feature
    return feature


def _collect_ratio_matches(desc_a: np.ndarray, desc_b: np.ndarray, ratio_thresh: float) -> list[dict[str, float | int]]:
    """执行 BFMatcher + knnMatch(k=2) + Lowe ratio test。

    返回的是 A->B 的单向候选匹配；是否做双向一致性由外层配置控制。
    """

    if len(desc_a) == 0 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    matches: list[dict[str, float | int]] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        ratio = float(best.distance / max(second.distance, 1e-12))
        if ratio <= ratio_thresh:
            matches.append(
                {
                    "kp_a_idx": int(best.queryIdx),
                    "kp_b_idx": int(best.trainIdx),
                    "descriptor_dist": float(best.distance),
                    "ratio": ratio,
                }
            )
    return matches


def _estimate_transform(src_points: np.ndarray, dst_points: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """用 RANSAC 估计 A->B 几何变换。

    partial_affine 对小局部指纹图通常比 homography 更稳，因为自由度更低，不容易被少量错误点带偏。
    """

    if len(src_points) < 3 or len(dst_points) < 3:
        return None, None
    threshold = float(get_nested(config, "matching", "ransac_reproj_thresh", default=5.0))
    model = str(get_nested(config, "matching", "ransac_model", default="partial_affine")).lower()
    if model in {"affine", "full_affine"}:
        matrix, mask = cv2.estimateAffine2D(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=threshold, confidence=0.99, refineIters=10)
    else:
        matrix, mask = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=threshold, confidence=0.99, refineIters=10)
    if matrix is None or mask is None:
        return None, None
    return matrix.astype(np.float32), mask.reshape(-1).astype(bool)


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """使用 2x3 仿射矩阵投影二维点。"""

    points_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (matrix @ points_h.T).T


def _reprojection_errors(matrix: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """计算每个匹配点的重投影误差。"""

    projected = _project(matrix, src_points)
    return np.linalg.norm(projected - dst_points, axis=1)


def _transform_rotation_deg(matrix: np.ndarray) -> float:
    """从仿射矩阵中估计整体旋转角。"""

    return float(np.degrees(np.arctan2(float(matrix[1, 0]), float(matrix[0, 0]))))


def _transform_scale(matrix: np.ndarray) -> float:
    """从仿射矩阵中估计整体尺度变化。

    使用线性部分行列式的平方根作为面积尺度的近似。
    """

    linear = matrix[:, :2].astype(np.float64)
    det = float(np.linalg.det(linear))
    if abs(det) < 1e-8:
        return 1.0
    return float(np.sqrt(abs(det)))


def _border_ok(kp: dict[str, Any], image_shape: tuple[int, int], margin: int) -> bool:
    """检查关键点是否离图像边界足够远。"""

    height, width = image_shape
    return margin <= float(kp["x"]) < width - margin and margin <= float(kp["y"]) < height - margin


def _texture_score(image: np.ndarray | None, x: float, y: float, radius: int, min_std: float) -> tuple[float, float]:
    """计算关键点附近的纹理分数和原始标准差。"""

    if image is None:
        return 0.5, 0.0
    xi = int(round(x))
    yi = int(round(y))
    patch = image[max(0, yi - radius) : min(image.shape[0], yi + radius + 1), max(0, xi - radius) : min(image.shape[1], xi + radius + 1)]
    if patch.size == 0:
        return 0.0, 0.0
    std = float(np.std(patch.astype(np.float32)))
    return clamp(std / max(min_std * 3.0, 1e-6)), std


def _descriptor_ratio_to_candidate(desc: np.ndarray, desc_b: np.ndarray, candidate_idx: int) -> tuple[float, float]:
    """计算几何扩展候选的 descriptor 距离和近似 ratio。

    几何扩展样本不是通过 ratio test 产生的，但仍记录该候选相对最近其他点的 ratio，
    方便 QA 和后续分析。
    """

    if len(desc_b) == 0:
        return 0.0, 1.0
    distances = np.linalg.norm(desc_b - desc[None, :], axis=1)
    selected = float(distances[candidate_idx])
    if len(distances) == 1:
        return selected, 1.0
    sorted_indices = np.argsort(distances)
    second_idx = int(sorted_indices[1] if int(sorted_indices[0]) == candidate_idx else sorted_indices[0])
    ratio = selected / max(float(distances[second_idx]), 1e-12)
    return selected, float(ratio)


def _build_candidate(
    source: str,
    split_name: str,
    finger_id: str,
    image_pair_id: str,
    sample_a: dict[str, str],
    sample_b: dict[str, str],
    kp_a_idx: int,
    kp_b_idx: int,
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    matrix: np.ndarray,
    transform_rotation: float,
    transform_scale: float,
    descriptor_dist: float,
    ratio: float,
    reproj_error: float,
    ransac_inlier: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """把一个候选 keypoint 对转换成可训练正样本元数据。

    这里是正样本质量控制的核心：
    - seed inlier 和 geometry expanded 都会经过边界、纹理、打分。
    - geometry expanded 额外要求方向误差和尺度比例在阈值内。
    - 返回 None 表示该候选被过滤。
    """

    kp_a = feature_a["keypoints"][kp_a_idx]
    kp_b = feature_b["keypoints"][kp_b_idx]
    border_margin = int(get_nested(config, "matching", "border_margin", default=8))
    # 两侧 keypoint 都必须远离边界，否则后续 64x64 -> 32x32 裁剪会产生大量黑边。
    if not _border_ok(kp_a, feature_a["image"].shape[:2], border_margin) or not _border_ok(kp_b, feature_b["image"].shape[:2], border_margin):
        return None

    expected_b_angle = float(kp_a["angle"]) + transform_rotation
    angle_error = angle_diff_mod180(float(kp_b["angle"]), expected_b_angle)
    angle_thresh = float(get_nested(config, "matching", "angle_thresh_deg", default=30.0))
    # 几何扩展样本比 seed inlier 更容易引入错配，因此扩展样本必须满足方向一致性。
    if source == "geometry_expanded" and angle_error > angle_thresh:
        return None

    expected_size = max(float(kp_a["size"]) * transform_scale, 1e-6)
    scale_ratio = float(kp_b["size"]) / expected_size
    scale_min = float(get_nested(config, "matching", "scale_ratio_min", default=0.7))
    scale_max = float(get_nested(config, "matching", "scale_ratio_max", default=1.4))
    # 尺度差太大时，即便坐标投影接近，也很可能不是同一个局部结构。
    if source == "geometry_expanded" and not (scale_min <= scale_ratio <= scale_max):
        return None

    radius = int(get_nested(config, "matching", "texture_window_radius", default=10))
    min_std = float(get_nested(config, "matching", "min_texture_std", default=5.0))
    texture_a, texture_std_a = _texture_score(feature_a["image"], float(kp_a["x"]), float(kp_a["y"]), radius, min_std)
    texture_b, texture_std_b = _texture_score(feature_b["image"], float(kp_b["x"]), float(kp_b["y"]), radius, min_std)
    # 两侧局部纹理都必须足够丰富，否则 HardNet 学到的描述子监督会很弱。
    if min(texture_std_a, texture_std_b) < min_std:
        return None

    weights = get_nested(config, "matching", "score_weights", default={})
    # 将不同量纲的误差/质量指标转成 0-1 分数后加权。
    # stability_score 只用于排序和去重，不作为训练标签。
    descriptor_score = exp_score(descriptor_dist, float(get_nested(config, "matching", "descriptor_distance_scale", default=0.7)))
    reproj_score = exp_score(reproj_error, float(get_nested(config, "matching", "reproj_error_scale", default=5.0)))
    angle_score = exp_score(angle_error, float(get_nested(config, "matching", "angle_error_scale", default=30.0)))
    scale_score = clamp(1.0 - abs(scale_ratio - 1.0) / max(scale_max - 1.0, 1.0 - scale_min, 1e-6))
    response_a = clamp(float(kp_a["response"]) / max(float(feature_a["response_p90"]), 1e-6))
    response_b = clamp(float(kp_b["response"]) / max(float(feature_b["response_p90"]), 1e-6))
    response_score = 0.5 * (response_a + response_b)
    texture = 0.5 * (texture_a + texture_b)
    source_score = 1.0 if source == "sift_inlier" else 0.0
    stability_score = (
        float(weights.get("source_seed", 0.08)) * source_score
        + float(weights.get("descriptor", 0.22)) * descriptor_score
        + float(weights.get("reproj", 0.24)) * reproj_score
        + float(weights.get("angle", 0.18)) * angle_score
        + float(weights.get("scale", 0.10)) * scale_score
        + float(weights.get("response", 0.08)) * response_score
        + float(weights.get("texture", 0.10)) * texture
    )
    stability_score = clamp(stability_score)

    pair_id = stable_id(finger_id, sample_a["image_id"], sample_b["image_id"], kp_a_idx, kp_b_idx, source)
    correspondence_id = stable_id(image_pair_id, kp_a_idx, kp_b_idx, source)
    return {
        "pair_id": pair_id,
        "correspondence_id": correspondence_id,
        "finger_id": finger_id,
        "image_pair_id": image_pair_id,
        "split": split_name,
        "source": source,
        "image_a_path": sample_a["image_path"],
        "image_b_path": sample_b["image_path"],
        "image_a_id": sample_a["image_id"],
        "image_b_id": sample_b["image_id"],
        "kp_a_idx": int(kp_a_idx),
        "kp_b_idx": int(kp_b_idx),
        "x_a": float(kp_a["x"]),
        "y_a": float(kp_a["y"]),
        "size_a": float(kp_a["size"]),
        "angle_a": float(kp_a["angle"]),
        "response_a": float(kp_a["response"]),
        "x_b": float(kp_b["x"]),
        "y_b": float(kp_b["y"]),
        "size_b": float(kp_b["size"]),
        "angle_b": float(kp_b["angle"]),
        "response_b": float(kp_b["response"]),
        "descriptor_dist": float(descriptor_dist),
        "ratio": float(ratio),
        "ransac_inlier": int(ransac_inlier),
        "reproj_error": float(reproj_error),
        "angle_diff_mod180": float(angle_error),
        "scale_ratio": float(scale_ratio),
        "stability_score": float(stability_score),
        "transform_matrix": json.dumps(np.asarray(matrix, dtype=float).tolist(), ensure_ascii=False),
        "transform_rotation_deg": float(transform_rotation),
        "patch_a_path": "",
        "patch_p_path": "",
    }


def _expand_by_geometry(
    split_name: str,
    finger_id: str,
    image_pair_id: str,
    sample_a: dict[str, str],
    sample_b: dict[str, str],
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    matrix: np.ndarray,
    transform_rotation: float,
    transform_scale: float,
    existing_pairs: set[tuple[int, int]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """使用 RANSAC 几何矩阵扩展正样本。

    做法：
    1. 把 A 图所有 keypoint 坐标通过 T 投影到 B 图。
    2. 在 B 图 keypoints 中搜索投影点半径 projection_radius 内的候选。
    3. 若有多个候选，优先选择“投影距离近 + 方向误差小”的那个。
    4. 再交给 _build_candidate 做边界、方向、尺度、纹理和打分过滤。
    """

    if not bool(get_nested(config, "matching", "enable_geometry_expansion", default=True)):
        return []
    if len(feature_a["keypoints"]) == 0 or len(feature_b["keypoints"]) == 0:
        return []

    points_a = np.asarray([[kp["x"], kp["y"]] for kp in feature_a["keypoints"]], dtype=np.float32)
    points_b = np.asarray([[kp["x"], kp["y"]] for kp in feature_b["keypoints"]], dtype=np.float32)
    projected = _project(matrix, points_a)
    projection_radius = float(get_nested(config, "matching", "projection_radius", default=5.0))
    angle_thresh = float(get_nested(config, "matching", "angle_thresh_deg", default=30.0))
    candidates: list[dict[str, Any]] = []

    for kp_a_idx, projected_xy in enumerate(projected):
        # 第一层筛选：只看几何投影附近有没有 B 图 keypoint。
        deltas = points_b - projected_xy[None, :]
        distances = np.linalg.norm(deltas, axis=1)
        valid_indices = np.where(distances <= projection_radius)[0]
        if len(valid_indices) == 0:
            continue

        scored: list[tuple[float, int, float]] = []
        kp_a = feature_a["keypoints"][kp_a_idx]
        expected_angle = float(kp_a["angle"]) + transform_rotation
        for kp_b_idx in valid_indices:
            kp_b = feature_b["keypoints"][int(kp_b_idx)]
            angle_error = angle_diff_mod180(float(kp_b["angle"]), expected_angle)
            # 候选排序分数：坐标距离是主项，方向误差是软惩罚。
            # 真正的硬阈值过滤在 _build_candidate 中完成。
            total = float(distances[kp_b_idx]) + projection_radius * min(angle_error / max(angle_thresh, 1e-6), 2.0)
            scored.append((total, int(kp_b_idx), float(distances[kp_b_idx])))
        scored.sort(key=lambda item: item[0])
        _, kp_b_idx, reproj_error = scored[0]
        if (kp_a_idx, kp_b_idx) in existing_pairs:
            continue

        descriptor_dist, ratio = _descriptor_ratio_to_candidate(feature_a["descriptors"][kp_a_idx], feature_b["descriptors"], kp_b_idx)
        candidate = _build_candidate(
            source="geometry_expanded",
            split_name=split_name,
            finger_id=finger_id,
            image_pair_id=image_pair_id,
            sample_a=sample_a,
            sample_b=sample_b,
            kp_a_idx=kp_a_idx,
            kp_b_idx=kp_b_idx,
            feature_a=feature_a,
            feature_b=feature_b,
            matrix=matrix,
            transform_rotation=transform_rotation,
            transform_scale=transform_scale,
            descriptor_dist=descriptor_dist,
            ratio=ratio,
            reproj_error=reproj_error,
            ransac_inlier=0,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _deduplicate(candidates: list[dict[str, Any]], min_distance: float) -> list[dict[str, Any]]:
    """对同一 image_pair 内空间过近的候选做去重。

    目的：
    - 避免同一局部纹理区域被多个 SIFT keypoint 重复采样。
    - 降低后续 batch 内把近重复 patch 当成负样本的风险。
    """

    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["stability_score"], reverse=True):
        duplicate = False
        for chosen in kept:
            dist_a = float(np.hypot(candidate["x_a"] - chosen["x_a"], candidate["y_a"] - chosen["y_a"]))
            dist_b = float(np.hypot(candidate["x_b"] - chosen["x_b"], candidate["y_b"] - chosen["y_b"]))
            if dist_a < min_distance or dist_b < min_distance:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _select_final(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """从去重后的候选中选择最终导出的 K 个正样本。

    为了让第一版数据集确实包含几何扩展样本，会优先保留少量高分
    `geometry_expanded`，再按总分补齐到 max_positive_per_image_pair。
    """

    max_count = int(get_nested(config, "matching", "max_positive_per_image_pair", default=16))
    min_expanded = int(get_nested(config, "matching", "min_geometry_expanded_per_image_pair", default=4))
    sorted_candidates = sorted(candidates, key=lambda item: item["stability_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    expanded = [item for item in sorted_candidates if item["source"] == "geometry_expanded"]
    for item in expanded[: min(min_expanded, max_count)]:
        selected.append(item)
        selected_ids.add(item["pair_id"])

    for item in sorted_candidates:
        if len(selected) >= max_count:
            break
        if item["pair_id"] in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item["pair_id"])
    return sorted(selected, key=lambda item: item["stability_score"], reverse=True)


def _build_single_image_pair(
    split_name: str,
    finger_id: str,
    sample_a: dict[str, str],
    sample_b: dict[str, str],
    feature_a: dict[str, Any],
    feature_b: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """处理单个 image_pair。

    这个函数串起完整的单对图像流程：ratio match -> RANSAC -> seed inlier
    -> geometry expansion -> dedup -> top K。
    """

    image_pair_id = stable_id(finger_id, sample_a["image_id"], sample_b["image_id"], readable_parts=3)
    diagnostics: dict[str, Any] = {
        "split": split_name,
        "finger_id": finger_id,
        "image_pair_id": image_pair_id,
        "image_a_id": sample_a["image_id"],
        "image_b_id": sample_b["image_id"],
        "image_a_path": sample_a["image_path"],
        "image_b_path": sample_b["image_path"],
        "keypoints_a": len(feature_a["keypoints"]),
        "keypoints_b": len(feature_b["keypoints"]),
        "ratio_kept": 0,
        "mutual_kept": 0,
        "ransac_inliers": 0,
        "seed_candidates": 0,
        "expanded_candidates": 0,
        "final_pairs": 0,
        "mean_reproj_error": 0.0,
        "transform_rotation_deg": 0.0,
        "transform_matrix": "",
        "skip_reason": "",
    }
    if not feature_a["ok"] or not feature_b["ok"]:
        diagnostics["skip_reason"] = "image_read_failed"
        return [], diagnostics

    ratio_thresh = float(get_nested(config, "matching", "ratio_thresh", default=0.85))
    # 先用 descriptor 做粗匹配，得到足够干净的 seed matches 来估计几何关系。
    matches = _collect_ratio_matches(feature_a["descriptors"], feature_b["descriptors"], ratio_thresh)
    diagnostics["ratio_kept"] = len(matches)
    if bool(get_nested(config, "matching", "strict_two_way", default=False)):
        # 可选双向一致性：更严格但会减少 seed matches。
        reverse = _collect_ratio_matches(feature_b["descriptors"], feature_a["descriptors"], ratio_thresh)
        reverse_pairs = {(int(item["kp_b_idx"]), int(item["kp_a_idx"])) for item in reverse}
        matches = [item for item in matches if (int(item["kp_a_idx"]), int(item["kp_b_idx"])) in reverse_pairs]
    diagnostics["mutual_kept"] = len(matches)

    if len(matches) < int(get_nested(config, "matching", "min_seed_matches", default=3)):
        diagnostics["skip_reason"] = "not_enough_seed_matches"
        return [], diagnostics

    src = np.asarray([[feature_a["keypoints"][int(item["kp_a_idx"])]["x"], feature_a["keypoints"][int(item["kp_a_idx"])]["y"]] for item in matches], dtype=np.float32)
    dst = np.asarray([[feature_b["keypoints"][int(item["kp_b_idx"])]["x"], feature_b["keypoints"][int(item["kp_b_idx"])]["y"]] for item in matches], dtype=np.float32)
    # RANSAC 只基于 seed matches 估计 T。后续所有几何扩展都依赖这个 T 的可信度。
    matrix, mask = _estimate_transform(src, dst, config)
    if matrix is None or mask is None:
        diagnostics["skip_reason"] = "ransac_failed"
        return [], diagnostics

    errors = _reprojection_errors(matrix, src, dst)
    inlier_count = int(mask.sum())
    diagnostics["ransac_inliers"] = inlier_count
    diagnostics["mean_reproj_error"] = float(errors[mask].mean()) if inlier_count > 0 else 0.0
    transform_rotation = _transform_rotation_deg(matrix)
    transform_scale = _transform_scale(matrix)
    diagnostics["transform_rotation_deg"] = float(transform_rotation)
    diagnostics["transform_matrix"] = json.dumps(np.asarray(matrix, dtype=float).tolist(), ensure_ascii=False)
    if inlier_count < int(get_nested(config, "matching", "min_inliers_per_image_pair", default=8)):
        diagnostics["skip_reason"] = "not_enough_ransac_inliers"
        return [], diagnostics

    candidates: list[dict[str, Any]] = []
    existing_pairs: set[tuple[int, int]] = set()
    # RANSAC 内点作为最高置信的正样本来源。
    for index, item in enumerate(matches):
        if not bool(mask[index]):
            continue
        kp_a_idx = int(item["kp_a_idx"])
        kp_b_idx = int(item["kp_b_idx"])
        existing_pairs.add((kp_a_idx, kp_b_idx))
        candidate = _build_candidate(
            source="sift_inlier",
            split_name=split_name,
            finger_id=finger_id,
            image_pair_id=image_pair_id,
            sample_a=sample_a,
            sample_b=sample_b,
            kp_a_idx=kp_a_idx,
            kp_b_idx=kp_b_idx,
            feature_a=feature_a,
            feature_b=feature_b,
            matrix=matrix,
            transform_rotation=transform_rotation,
            transform_scale=transform_scale,
            descriptor_dist=float(item["descriptor_dist"]),
            ratio=float(item["ratio"]),
            reproj_error=float(errors[index]),
            ransac_inlier=1,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)

    diagnostics["seed_candidates"] = len(candidates)
    # 几何引导扩展：用可信 T 从所有 SIFT keypoints 中补充更多对应点。
    expanded = _expand_by_geometry(
        split_name=split_name,
        finger_id=finger_id,
        image_pair_id=image_pair_id,
        sample_a=sample_a,
        sample_b=sample_b,
        feature_a=feature_a,
        feature_b=feature_b,
        matrix=matrix,
        transform_rotation=transform_rotation,
        transform_scale=transform_scale,
        existing_pairs=existing_pairs,
        config=config,
    )
    diagnostics["expanded_candidates"] = len(expanded)
    candidates.extend(expanded)
    if not candidates:
        diagnostics["skip_reason"] = "no_valid_candidates"
        return [], diagnostics

    deduped = _deduplicate(candidates, min_distance=float(get_nested(config, "matching", "dedup_min_distance", default=12.0)))
    final = _select_final(deduped, config)
    diagnostics["final_pairs"] = len(final)
    if not final:
        diagnostics["skip_reason"] = "no_final_pairs"
    return final, diagnostics


def _iter_with_optional_tqdm(items: list[Any], description: str):
    """如果安装了 tqdm，就显示进度条；否则退化成普通迭代。"""

    if tqdm is None:
        return items
    return tqdm(items, desc=description, unit="pair")


def build_positive_pairs(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """执行整个 build_positive 阶段。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    split_assignments_path = output_root / "split_assignments.csv"
    if not split_assignments_path.exists():
        raise FileNotFoundError(f"缺少 split_assignments.csv，请先运行 split: {split_assignments_path}")

    rows = read_csv_rows(split_assignments_path)
    if not rows:
        raise ValueError("split_assignments.csv 为空")

    rows_by_split_finger: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rows_by_split_finger[row["split"]][row["finger_id"]].append(row)

    positive_path = output_root / "all_positive_pairs.csv"
    diagnostics_path = output_root / "match_diagnostics.csv"
    summary_path = output_root / "positive_build_summary.json"
    ensure_dir(output_root)

    sift = _build_sift(config)
    feature_cache: dict[str, dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    skip_counts: Counter[str] = Counter()
    image_pair_count = 0
    positive_count = 0

    max_pairs_per_finger = get_nested(config, "matching", "max_image_pairs_per_finger", default=None)
    max_pairs_per_finger = None if max_pairs_per_finger in (None, "", 0) else int(max_pairs_per_finger)

    # 使用流式写 CSV，避免全量两两组合时把所有正样本和诊断都堆在内存里。
    with positive_path.open("w", encoding="utf-8", newline="") as pos_handle, diagnostics_path.open("w", encoding="utf-8", newline="") as diag_handle:
        pos_writer = csv.DictWriter(pos_handle, fieldnames=POSITIVE_FIELDNAMES)
        diag_writer = csv.DictWriter(diag_handle, fieldnames=DIAGNOSTIC_FIELDNAMES)
        pos_writer.writeheader()
        diag_writer.writeheader()

        for split_name in ["train", "val", "test"]:
            for finger_id, finger_rows in rows_by_split_finger.get(split_name, {}).items():
                ordered = sorted(finger_rows, key=lambda item: item["image_id"])
                image_pairs = list(combinations(ordered, 2))
                if max_pairs_per_finger is not None:
                    image_pairs = image_pairs[:max_pairs_per_finger]

                # 同一 finger 内全部两两组合。若配置了 max_image_pairs_per_finger，
                # 可用于冒烟测试或小规模调参。
                for sample_a, sample_b in _iter_with_optional_tqdm(image_pairs, f"{split_name}:{finger_id}"):
                    feature_a = _extract_features(sample_a, config, sift, feature_cache, logger)
                    feature_b = _extract_features(sample_b, config, sift, feature_cache, logger)
                    current_pairs, diagnostics = _build_single_image_pair(
                        split_name=split_name,
                        finger_id=finger_id,
                        sample_a=sample_a,
                        sample_b=sample_b,
                        feature_a=feature_a,
                        feature_b=feature_b,
                        config=config,
                    )
                    image_pair_count += 1
                    if diagnostics.get("skip_reason"):
                        skip_counts[str(diagnostics["skip_reason"])] += 1
                    diag_writer.writerow({key: diagnostics.get(key, "") for key in DIAGNOSTIC_FIELDNAMES})

                    for row in current_pairs:
                        pos_writer.writerow({key: row.get(key, "") for key in POSITIVE_FIELDNAMES})
                        positive_count += 1
                        split_counts[row["split"]] += 1
                        source_counts[row["source"]] += 1

    split_summary_path = output_root / "split_summary.json"
    if split_summary_path.exists():
        split_summary = read_json(split_summary_path)
        for split_name in ["train", "val", "test"]:
            split_summary.setdefault("splits", {}).setdefault(split_name, {"finger_count": 0, "image_count": 0, "positive_pair_count": 0})
            split_summary["splits"][split_name]["positive_pair_count"] = int(split_counts.get(split_name, 0))
        write_json(split_summary_path, split_summary)

    write_json(
        summary_path,
        {
            "image_pair_count": image_pair_count,
            "positive_pair_count": positive_count,
            "split_positive_counts": dict(split_counts),
            "source_counts": dict(source_counts),
            "skip_counts": dict(skip_counts),
            "feature_cache_count": len(feature_cache),
        },
    )
    logger.info("正样本构建完成 | image_pairs=%d | positives=%d | sources=%s", image_pair_count, positive_count, dict(source_counts))
    return {"all_positive_pairs_path": str(positive_path), "diagnostics_path": str(diagnostics_path), "summary_path": str(summary_path), "positive_pair_count": positive_count}
