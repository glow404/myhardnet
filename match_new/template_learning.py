"""在线模板学习的身份确认、公共区域和内容去重计算。"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import cv2
import numpy as np


def template_content_hash(template: dict[str, Any]) -> str:
    """根据真实模板内容生成哈希，忽略 image_id、路径等元数据。"""

    digest = hashlib.sha256()
    for key in ("overlap_image", "keypoints_xy", "hardnet_descriptors"):
        value = np.ascontiguousarray(np.asarray(template.get(key, np.zeros((0,), dtype=np.uint8))))
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_coverage_mask(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """用局部灰度标准差提取具有脊线纹理的有效区域。

    该掩码与 C 方案的低对比块思想一致，但在像素级生成，随后通过形态学闭运算
    填补脊线之间的小空洞。它用于学习准入，不改变现有 ZNCC 匹配分数。
    """

    gray = np.asarray(image)
    if gray.ndim != 2 or gray.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    source = gray.astype(np.float32)
    window = max(3, int(config.get("coverage_window_size", 9)))
    if window % 2 == 0:
        window += 1
    mean = cv2.boxFilter(source, cv2.CV_32F, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(source * source, cv2.CV_32F, (window, window), normalize=True)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    mask = (local_std >= float(config.get("coverage_min_std", 5.0))).astype(np.uint8)
    kernel_size = max(1, int(config.get("coverage_morph_kernel", 3)))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def common_area_metrics(
    query_image: np.ndarray,
    gallery_image: np.ndarray,
    affine_matrix: np.ndarray | list[list[float]] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    """计算仿射对齐后的真实纹理公共区域。

    分母使用两张有效纹理区域面积的较小值。query 面积会乘仿射线性部分行列式，
    这样移出 gallery 画布的区域仍计入面积，不会因为裁剪而虚增公共区域比例。
    """

    result = {"available": False, "common_pixels": 0, "common_area_ratio": 0.0}
    query_mask = build_coverage_mask(query_image, config)
    gallery_mask = build_coverage_mask(gallery_image, config)
    matrix = np.asarray(affine_matrix, dtype=np.float64) if affine_matrix is not None else np.zeros((0, 0))
    if query_mask.size == 0 or gallery_mask.size == 0 or matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return result

    height, width = gallery_mask.shape
    warped_query = cv2.warpAffine(
        query_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    gallery_valid = gallery_mask.astype(bool)
    common_pixels = int(np.count_nonzero(warped_query & gallery_valid))
    scale_area = abs(float(np.linalg.det(matrix[:, :2])))
    if not math.isfinite(scale_area) or scale_area <= 1e-12:
        return result
    expected_query_area = float(np.count_nonzero(query_mask)) * scale_area
    gallery_area = float(np.count_nonzero(gallery_valid))
    denominator = min(expected_query_area, gallery_area)
    if denominator <= 0.0:
        return result
    result.update(
        {
            "available": True,
            "common_pixels": common_pixels,
            "common_area_ratio": float(np.clip(common_pixels / denominator, 0.0, 1.0)),
        }
    )
    return result


def build_learning_evidence(
    entry: dict[str, Any],
    match_result: dict[str, Any],
    common_area: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """把一次 query-template 匹配转成可解释的学习证据。"""

    score = float(match_result.get("score", 0.0))
    unique = int(match_result.get("unique_inliers", 0))
    texture_available = bool(match_result.get("texture_available", False))
    texture = float(match_result.get("texture_similarity", 0.0))
    common_ratio = float(common_area.get("common_area_ratio", 0.0))
    common_pixels = int(common_area.get("common_pixels", 0))
    common_ok = bool(common_area.get("available", False)) and common_ratio >= float(
        config.get("min_common_area_ratio", 0.35)
    ) and common_pixels >= int(config.get("min_common_area_pixels", 256))
    confirmation_ok = (
        score >= float(config.get("confirm_score_threshold", 0.70))
        and texture_available
        and common_ok
    )
    strict_ok = (
        score >= float(config.get("learn_score_threshold", 0.85))
        and unique >= int(config.get("learn_min_unique_inliers", 12))
        and texture_available
        and texture >= float(config.get("learn_min_texture_similarity", 0.75))
        and common_ok
    )
    return {
        "template_path": str(entry.get("template_path", "")),
        "template_source": str(entry.get("source", "seed")),
        "protected": bool(entry.get("protected", False)),
        "score": score,
        "unique_inliers": unique,
        "texture_available": texture_available,
        "texture_similarity": texture,
        "common_area_ratio": common_ratio,
        "common_pixels": common_pixels,
        "confirmation_ok": confirmation_ok,
        "strict_learning_ok": strict_ok,
    }


def evaluate_learning_decision(
    evidences: list[dict[str, Any]],
    *,
    duplicate_content: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    """根据全部可信模板证据给出最终学习准入结论。"""

    confirmations = [item for item in evidences if bool(item.get("confirmation_ok", False))]
    strict = [item for item in evidences if bool(item.get("strict_learning_ok", False))]
    seed_confirmations = [item for item in confirmations if str(item.get("template_source", "")) == "seed"]
    required = max(1, int(config.get("confirmation_templates", 2)))
    require_seed = bool(config.get("require_seed_confirmation", True))
    reasons: list[str] = []
    if duplicate_content:
        reasons.append("duplicate_template_content")
    if not strict:
        reasons.append("no_strict_high_confidence_match")
    if len(confirmations) < required:
        reasons.append("insufficient_template_confirmations")
    if require_seed and not seed_confirmations:
        reasons.append("missing_seed_confirmation")
    return {
        "accepted": not reasons,
        "reasons": reasons or ["accepted"],
        "num_confirmations": len(confirmations),
        "num_seed_confirmations": len(seed_confirmations),
        "num_strict_matches": len(strict),
        "best_score": max((float(item.get("score", 0.0)) for item in evidences), default=0.0),
        "best_common_area_ratio": max((float(item.get("common_area_ratio", 0.0)) for item in evidences), default=0.0),
    }
