"""HardNet L2 图像级匹配器。工具脚本

作用：
    对一张 query 指纹图模板和一张 gallery/template 指纹图模板进行匹配，
    输出图像级分数和诊断指标。这个文件是 match_new 的核心：

    1. 不再使用 Hadamard/Hamming 二值化；
    2. 直接在 HardNet 128 维连续 descriptor 上计算 L2 距离；
    3. 用 top-k / ratio 生成较宽松候选；
    4. 候选过多时做方向软排序截断，而不是方向峰硬裁剪；
    5. 用 RANSAC 估计局部仿射变换；
    6. 对 RANSAC inliers 做 one-to-one 去重；
    7. 融合归一化 unique inlier 分数与脊线纹理分数，得到图像级 match score。

输入：
    query/gallery 都是 `.npz` 模板读出的 dict，至少包含：
        - keypoints_xy
        - keypoints_angle
        - hardnet_descriptors

输出：
    一个 dict，包含 score、unique_inliers、raw_inliers、mean_l2_distance、
    mean_reproj_error、orientation_consistency 等字段。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchCandidate:
    """一个 HardNet 候选匹配。

    query_idx / gallery_idx 指向两张图中的 keypoint 下标；
    distance 是 HardNet L2 距离，越小越像；
    angle_delta 只用于软排序和诊断，不作为前置硬门槛。
    """

    query_idx: int
    gallery_idx: int
    distance: float
    angle_delta: float


@dataclass(frozen=True)
class ScoredInlier:
    """RANSAC inlier 的去重排序信息。"""

    candidate: MatchCandidate
    reproj_error: float
    sort_score: float


def wrap_angle_deg(angle: float) -> float:
    """把角度规整到 (-180, 180]，方便比较方向差。"""

    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return float(angle)


def select_l2_descriptors(template: dict[str, Any], descriptor_source: str) -> np.ndarray:
    """从模板中取出指定 descriptor，并做 L2 归一化保护。

    `hardnet` 使用 `hardnet_descriptors`；`sift` 使用 `sift_descriptors`。
    """

    source = str(descriptor_source).lower()
    if source == "hardnet":
        key = "hardnet_descriptors"
        has_flag = "has_hardnet"
    elif source in {"sift", "rootsift"}:
        key = "sift_descriptors"
        has_flag = "has_sift"
    else:
        raise ValueError(f"unsupported descriptor_source: {descriptor_source}")
    if has_flag in template and not bool(template.get(has_flag)):
        template_path = template.get("template_path", "<unknown>")
        raise ValueError(
            f"template has no {key}: {template_path}. "
            "请使用对应的单描述子模板目录，不要复用双描述子或错误类型的模板。"
        )
    desc = np.asarray(template.get(key, np.zeros((0, 128))), dtype=np.float32)
    if desc.ndim == 1:
        desc = desc.reshape(1, -1)
    if desc.size == 0:
        return np.zeros((0, 128), dtype=np.float32)
    if desc.ndim != 2 or desc.shape[1] != 128:
        raise ValueError(f"{key} must be [N,128], got {desc.shape}")
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    return desc / np.maximum(norms, 1e-12)


def select_hardnet_descriptors(template: dict[str, Any]) -> np.ndarray:
    """兼容旧调用：取 HardNet descriptor 并做 L2 归一化。"""

    return select_l2_descriptors(template, "hardnet")


def keypoint_angles(template: dict[str, Any], n: int) -> np.ndarray:
    """取出与 descriptor 行数对齐的 keypoint angle 数组。"""

    angles = np.asarray(template.get("keypoints_angle", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
    if angles.shape[0] < n:
        padded = np.zeros((n,), dtype=np.float32)
        padded[: angles.shape[0]] = angles
        return padded
    return angles[:n]


def pairwise_l2(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """计算 query descriptors 到 gallery descriptors 的全量 L2 距离矩阵。

    对 L2-normalized descriptor 来说，L2 排序与 cosine 排序等价；
    但 L2 距离范围更直观，便于设置 abs_distance_threshold。
    """

    q2 = np.sum(query * query, axis=1, keepdims=True)
    g2 = np.sum(gallery * gallery, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + g2 - 2.0 * query @ gallery.T, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """返回最小的 count 个下标。

    使用 argpartition 避免每一行都完整排序，gallery keypoint 较多时更快。
    """

    if values.size == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    count = min(int(count), int(values.size))
    if count == values.size:
        order = np.argsort(values)
    else:
        part = np.argpartition(values, kth=count - 1)[:count]
        order = part[np.argsort(values[part])]
    return order.astype(np.int64)


def build_l2_candidates(query: dict[str, Any], gallery: dict[str, Any], cfg: dict[str, Any], descriptor_source: str = "hardnet") -> list[MatchCandidate]:
    """生成 RANSAC 前的 HardNet L2 候选。

    支持三种策略：
        - ratio_only：只要最近邻通过 Lowe ratio；
        - topk_only：保留 top-k 中距离足够近的候选；
        - topk_or_ratio：ratio 通过则取最近邻，否则退回 top-k。

    当 bidirectional_ratio_test=true 时，经 ratio 分支保留的最近邻还必须满足：
        1. gallery -> query 方向也通过 Lowe ratio；
        2. 两个方向的最近邻互相指向同一对关键点。

    这里默认允许 many-to-one，因为指纹局部纹理重复，过早强制一对一会删掉真匹配。
    """

    desc_q = select_l2_descriptors(query, descriptor_source)
    desc_g = select_l2_descriptors(gallery, descriptor_source)
    if desc_q.shape[0] == 0 or desc_g.shape[0] == 0:
        return []

    q_angles = keypoint_angles(query, desc_q.shape[0])
    g_angles = keypoint_angles(gallery, desc_g.shape[0])
    distances = pairwise_l2(desc_q, desc_g)
    policy = str(cfg.get("candidate_policy", "topk_or_ratio")).lower()
    top_k = max(1, int(cfg.get("top_k", 5)))
    ratio_threshold = float(cfg.get("ratio_threshold", 0.95))
    bidirectional_ratio = bool(cfg.get("bidirectional_ratio_test", False))
    abs_threshold = float(cfg.get("abs_distance_threshold", 1.10))
    margin = float(cfg.get("distance_margin", 0.12))
    allow_many = bool(cfg.get("allow_many_to_one_before_ransac", True))

    # 双向 Lowe 验证需要预先得到每个 gallery 描述子的 query 侧最近邻和次近邻。
    # 只在开关启用且当前策略包含 ratio 分支时计算；topk_only 不受此开关影响。
    reverse_best_query = np.full((distances.shape[1],), -1, dtype=np.int64)
    reverse_ratio_ok = np.zeros((distances.shape[1],), dtype=bool)
    if bidirectional_ratio and policy in {"ratio_only", "topk_or_ratio"}:
        for gallery_idx in range(distances.shape[1]):
            column = distances[:, gallery_idx]
            reverse_order = top_indices(column, min(2, column.size))
            if reverse_order.size == 0:
                continue
            reverse_best_idx = int(reverse_order[0])
            reverse_best_dist = float(column[reverse_best_idx])
            reverse_second_dist = float(column[int(reverse_order[1])]) if reverse_order.size > 1 else float("inf")
            reverse_best_query[gallery_idx] = reverse_best_idx
            reverse_ratio_ok[gallery_idx] = bool(
                reverse_second_dist > 1e-12
                and reverse_best_dist < ratio_threshold * reverse_second_dist
            )

    # candidates 保存所有进入后续几何验证的候选；
    # best_gallery 只在 allow_many=false 时使用，用于提前做 gallery 侧去重。
    candidates: list[MatchCandidate] = []
    best_gallery: dict[int, MatchCandidate] = {}
    for query_idx in range(distances.shape[0]):
        row = distances[query_idx]
        need = min(max(top_k, 2), row.size)
        order = top_indices(row, need)
        if order.size == 0:
            continue
        best_idx = int(order[0])
        best_dist = float(row[best_idx])
        second_dist = float(row[int(order[1])]) if order.size > 1 else float("inf")
        ratio_ok = bool(second_dist > 1e-12 and best_dist < ratio_threshold * second_dist)
        if ratio_ok and bidirectional_ratio:
            ratio_ok = bool(
                reverse_best_query[best_idx] == query_idx
                and reverse_ratio_ok[best_idx]
            )
        adaptive_limit = min(abs_threshold, best_dist + margin)

        # 自适应距离上限防止 top-k 把明显过远的候选也塞进 RANSAC。
        keep: list[int] = []
        if policy == "ratio_only":
            if ratio_ok and best_dist <= abs_threshold:
                keep = [best_idx]
        elif policy == "topk_only":
            keep = [int(i) for i in order[:top_k] if float(row[int(i)]) <= adaptive_limit]
        elif policy == "topk_or_ratio":
            if ratio_ok and best_dist <= abs_threshold:
                keep = [best_idx]
            else:
                keep = [int(i) for i in order[:top_k] if float(row[int(i)]) <= adaptive_limit]
        else:
            raise ValueError(f"unsupported candidate_policy: {policy}")

        for gallery_idx in keep:
            candidate = MatchCandidate(
                query_idx=int(query_idx),
                gallery_idx=int(gallery_idx),
                distance=float(row[gallery_idx]),
                angle_delta=wrap_angle_deg(float(g_angles[gallery_idx]) - float(q_angles[query_idx])),
            )
            if allow_many:
                candidates.append(candidate)
            else:
                previous = best_gallery.get(candidate.gallery_idx)
                if previous is None or candidate.distance < previous.distance:
                    best_gallery[candidate.gallery_idx] = candidate

    return candidates if allow_many else list(best_gallery.values())


def dominant_angle_delta(candidates: list[MatchCandidate], bin_deg: float = 10.0) -> float:
    """估计候选集的主方向差。

    主方向差只作为软排序和统计诊断使用，不直接删除候选。
    """

    if not candidates:
        return 0.0
    bin_deg = max(float(bin_deg), 1e-6)
    num_bins = max(1, int(math.ceil(360.0 / bin_deg)))
    hist = np.zeros((num_bins,), dtype=np.int32)
    for candidate in candidates:
        shifted = wrap_angle_deg(candidate.angle_delta) + 180.0
        idx = int(np.floor(shifted / bin_deg))
        idx = max(0, min(num_bins - 1, idx))
        hist[idx] += 1
    best = int(np.argmax(hist))
    return wrap_angle_deg(-180.0 + (best + 0.5) * bin_deg)


def soft_gate_candidates(candidates: list[MatchCandidate], cfg: dict[str, Any]) -> tuple[list[MatchCandidate], float]:
    """候选过多时做轻量截断。

    如果候选数未超过 max_candidates_for_ransac，原样返回。
    如果超过，则按“归一化 L2 距离 + 少量方向惩罚”排序，只保留前 N 个。
    方向惩罚权重很小，目的是让 RANSAC 少看一点明显方向离群的候选，
    而不是复刻 SIFT 方向峰硬过滤。
    """

    max_candidates = int(cfg.get("max_candidates_for_ransac", 250))
    dominant = dominant_angle_delta(candidates, float(cfg.get("orientation_hist_bin_deg", 10.0)))
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates, dominant

    if bool(cfg.get("orientation_soft_gate", True)):
        distances = np.asarray([item.distance for item in candidates], dtype=np.float32)
        lo = float(np.min(distances))
        hi = float(np.max(distances))
        denom = max(hi - lo, 1e-6)
        weight = float(cfg.get("orientation_weight", 0.15))
        scored = []
        for idx, candidate in enumerate(candidates):
            norm_dist = (candidate.distance - lo) / denom
            angle_penalty = min(abs(wrap_angle_deg(candidate.angle_delta - dominant)), 45.0) / 45.0
            scored.append((norm_dist + weight * angle_penalty, idx))
        keep = [idx for _, idx in sorted(scored, key=lambda item: item[0])[:max_candidates]]
    else:
        keep = np.argsort([item.distance for item in candidates])[:max_candidates].tolist()
    return [candidates[int(i)] for i in keep], dominant


def points_from_candidates(
    candidates: list[MatchCandidate],
    query_xy: np.ndarray,
    gallery_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把候选下标转换为 RANSAC 需要的坐标点对。"""

    src = np.asarray([query_xy[c.query_idx] for c in candidates], dtype=np.float32)
    dst = np.asarray([gallery_xy[c.gallery_idx] for c in candidates], dtype=np.float32)
    return src.reshape(-1, 2), dst.reshape(-1, 2)


def estimate_partial_affine_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """用最小二乘重新估计 partial affine。

    OpenCV 的 RANSAC 阶段负责找 inlier；去重后的 final inliers 更干净，
    所以这里用一个简单的相似/partial affine 线性模型重新拟合最终矩阵。
    模型形式：
        u = a*x - b*y + tx
        v = b*x + a*y + ty
    """

    if src.shape[0] < 2 or dst.shape[0] < 2:
        return None
    rows = []
    values = []
    for (x, y), (u, v) in zip(src.astype(np.float64), dst.astype(np.float64)):
        rows.append([x, -y, 1.0, 0.0])
        values.append(u)
        rows.append([y, x, 0.0, 1.0])
        values.append(v)
    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(values, dtype=np.float64)
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    scale_a, scale_b, tx, ty = params.tolist()
    return np.asarray([[scale_a, -scale_b, tx], [scale_b, scale_a, ty]], dtype=np.float64)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """对一批二维点应用 2x3 affine 矩阵。"""

    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def reprojection_errors(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """计算 src 经 affine 投影到 dst 后的像素误差。"""

    pred = apply_affine(matrix, src)
    return np.linalg.norm(pred - dst.astype(np.float64), axis=1).astype(np.float32)


def partial_affine_scale(matrix: np.ndarray) -> float:
    """从 estimateAffinePartial2D 的 2x3 矩阵中提取等比例缩放因子。

    partial affine 的线性部分形式近似为：
        [[a, -b],
         [b,  a]]
    因此 scale = sqrt(a^2 + b^2)。这里不限制旋转角，只限制尺度范围。
    """

    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (2, 3):
        return float("nan")
    a = float(affine[0, 0])
    b = float(affine[1, 0])
    return float(math.sqrt(a * a + b * b))


def scale_in_allowed_range(matrix: np.ndarray, cfg: dict[str, Any]) -> tuple[bool, float]:
    """检查 partial affine 的尺度是否落在配置允许范围内。"""

    scale = partial_affine_scale(matrix)
    min_scale = float(cfg.get("min_scale", 0.0))
    max_scale = float(cfg.get("max_scale", float("inf")))
    if not math.isfinite(scale):
        return False, scale
    return bool(min_scale <= scale <= max_scale), scale


def ridge_texture_similarity(
    query_image: np.ndarray,
    gallery_image: np.ndarray,
    matrix: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """计算仿射对齐后的局部脊线纹理相似度。

    ``matrix`` 必须把 query 坐标映射到 gallery 坐标。函数将 query 灰度图
    warp 到 gallery 坐标系，仅在真实落入 gallery 的区域按块计算 ZNCC。
    低对比块不包含稳定脊线信息，因此不参与最终中位数统计。
    """

    result = {
        "available": False,
        "similarity": 0.0,
        "overlap_fraction": 0.0,
        "valid_blocks": 0,
    }
    query_gray = np.asarray(query_image)
    gallery_gray = np.asarray(gallery_image)
    affine = np.asarray(matrix, dtype=np.float64)
    if query_gray.ndim != 2 or gallery_gray.ndim != 2 or query_gray.size == 0 or gallery_gray.size == 0:
        return result
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
        return result

    query_float = query_gray.astype(np.float32)
    gallery_float = gallery_gray.astype(np.float32)
    blur_sigma = max(0.0, float(cfg.get("blur_sigma", 0.8)))
    if blur_sigma > 0.0:
        query_float = cv2.GaussianBlur(query_float, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        gallery_float = cv2.GaussianBlur(gallery_float, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    height, width = gallery_float.shape
    warped_query = cv2.warpAffine(
        query_float,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_support = cv2.warpAffine(
        np.ones(query_float.shape, dtype=np.uint8),
        affine,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    overlap_fraction = float(np.mean(warped_support)) if warped_support.size else 0.0
    result["overlap_fraction"] = overlap_fraction
    if overlap_fraction < float(cfg.get("min_overlap_fraction", 0.20)):
        return result

    block_size = max(4, int(cfg.get("block_size", 16)))
    min_block_valid_fraction = float(cfg.get("min_block_valid_fraction", 0.60))
    min_block_std = max(0.0, float(cfg.get("min_block_std", 5.0)))
    block_scores: list[float] = []
    for top in range(0, height, block_size):
        bottom = min(top + block_size, height)
        for left in range(0, width, block_size):
            right = min(left + block_size, width)
            valid = warped_support[top:bottom, left:right]
            if valid.size == 0 or float(np.mean(valid)) < min_block_valid_fraction:
                continue
            query_values = warped_query[top:bottom, left:right][valid].astype(np.float64)
            gallery_values = gallery_float[top:bottom, left:right][valid].astype(np.float64)
            if query_values.size < 16:
                continue
            query_centered = query_values - float(np.mean(query_values))
            gallery_centered = gallery_values - float(np.mean(gallery_values))
            query_energy = float(np.dot(query_centered, query_centered))
            gallery_energy = float(np.dot(gallery_centered, gallery_centered))
            if math.sqrt(query_energy / query_values.size) < min_block_std:
                continue
            if math.sqrt(gallery_energy / gallery_values.size) < min_block_std:
                continue
            denominator = math.sqrt(query_energy * gallery_energy)
            if denominator <= 1e-12:
                continue
            correlation = float(np.dot(query_centered, gallery_centered) / denominator)
            # 负相关不构成同一脊线纹理的正证据，统一截断为 0。
            block_scores.append(float(np.clip(correlation, 0.0, 1.0)))

    result["valid_blocks"] = len(block_scores)
    if len(block_scores) < max(1, int(cfg.get("min_valid_blocks", 4))):
        return result
    result["available"] = True
    result["similarity"] = float(np.median(np.asarray(block_scores, dtype=np.float32)))
    return result


def compute_fused_match_score(
    query: dict[str, Any],
    gallery: dict[str, Any],
    matrix: np.ndarray,
    unique_count: int,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """按 C 方案融合归一化几何分数与脊线纹理相似度。"""

    cfg = dict(config.get("texture_verification", {}))
    low_unique = int(cfg.get("low_unique_inliers", 4))
    saturation_unique = float(cfg.get("geometry_saturation_inliers", 12.0))
    if low_unique < 2:
        raise ValueError(f"texture_verification.low_unique_inliers must be >= 2, got {low_unique}")
    if saturation_unique <= 0.0:
        raise ValueError(
            "texture_verification.geometry_saturation_inliers must be > 0, "
            f"got {saturation_unique}"
        )
    geometry_similarity = float(np.clip(unique_count / saturation_unique, 0.0, 1.0))
    diagnostics = {
        "texture_enabled": bool(cfg.get("enabled", False)),
        "texture_evaluated": False,
        "texture_available": False,
        "texture_similarity": 0.0,
        "texture_overlap_fraction": 0.0,
        "texture_valid_blocks": 0,
        "geometry_similarity": geometry_similarity,
        "geometry_weight": 1.0,
        "texture_weight": 0.0,
        "texture_decision": "disabled",
        "texture_similarity_ms": 0.0,
    }
    if unique_count < low_unique:
        diagnostics["texture_decision"] = "geometry_below_low_gate"
        return 0.0, diagnostics
    if not diagnostics["texture_enabled"]:
        diagnostics["texture_decision"] = "geometry_only"
        return geometry_similarity, diagnostics

    query_image = np.asarray(query.get("overlap_image", np.zeros((0, 0), dtype=np.uint8)))
    gallery_image = np.asarray(gallery.get("overlap_image", np.zeros((0, 0), dtype=np.uint8)))
    if query_image.size == 0 or gallery_image.size == 0:
        query_path = query.get("template_path", query.get("image_id", "<query>"))
        gallery_path = gallery.get("template_path", gallery.get("image_id", "<gallery>"))
        raise ValueError(
            "texture verification requires overlap_image in both templates. "
            f"Rebuild templates instead of using --skip-template-build: query={query_path}, gallery={gallery_path}"
        )

    diagnostics["texture_evaluated"] = True
    texture_started = time.perf_counter()
    texture = ridge_texture_similarity(query_image, gallery_image, matrix, cfg)
    diagnostics["texture_similarity_ms"] = (
        time.perf_counter() - texture_started
    ) * 1000.0
    diagnostics["texture_available"] = bool(texture["available"])
    diagnostics["texture_similarity"] = float(texture["similarity"])
    diagnostics["texture_overlap_fraction"] = float(texture["overlap_fraction"])
    diagnostics["texture_valid_blocks"] = int(texture["valid_blocks"])
    if not diagnostics["texture_available"]:
        diagnostics["texture_decision"] = "insufficient_texture_area"
        return 0.0, diagnostics

    geometry_weight = float(cfg.get("geometry_weight", 0.70))
    texture_weight = float(cfg.get("texture_weight", 0.30))
    if geometry_weight < 0.0 or texture_weight < 0.0 or geometry_weight + texture_weight <= 0.0:
        raise ValueError(
            "texture_verification geometry_weight and texture_weight must be non-negative "
            "with a positive sum"
        )
    weight_sum = geometry_weight + texture_weight
    geometry_weight /= weight_sum
    texture_weight /= weight_sum
    diagnostics["geometry_weight"] = geometry_weight
    diagnostics["texture_weight"] = texture_weight
    diagnostics["texture_decision"] = "fused_score"
    match_score = (
        geometry_weight * geometry_similarity
        + texture_weight * diagnostics["texture_similarity"]
    )
    return float(np.clip(match_score, 0.0, 1.0)), diagnostics


def unique_inliers(
    inliers: list[MatchCandidate],
    query_xy: np.ndarray,
    gallery_xy: np.ndarray,
    matrix: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[list[MatchCandidate], np.ndarray]:
    """把 RANSAC 原始 inliers 清理成 one-to-one inliers。

    RANSAC 输入允许 many-to-one，所以原始 inlier 中可能出现同一个 query 点
    对多个 gallery 点，或者多个 query 点挤到同一个 gallery 点。
    这里按“descriptor 距离 + 重投影误差惩罚”从好到坏贪心保留，
    最终每个 query/gallery keypoint 最多出现一次。
    """

    if not inliers:
        return [], np.zeros((0,), dtype=np.float32)
    src, dst = points_from_candidates(inliers, query_xy, gallery_xy)
    errors = reprojection_errors(matrix, src, dst)
    weight = float(cfg.get("reproj_error_weight", 0.05))
    scored = [
        ScoredInlier(candidate=candidate, reproj_error=float(error), sort_score=float(candidate.distance + weight * float(error)))
        for candidate, error in zip(inliers, errors)
    ]
    used_q: set[int] = set()
    used_g: set[int] = set()
    kept: list[MatchCandidate] = []
    kept_errors: list[float] = []
    for item in sorted(scored, key=lambda x: x.sort_score):
        q = item.candidate.query_idx
        g = item.candidate.gallery_idx
        if q in used_q or g in used_g:
            continue
        used_q.add(q)
        used_g.add(g)
        kept.append(item.candidate)
        kept_errors.append(item.reproj_error)
    return kept, np.asarray(kept_errors, dtype=np.float32)


def orientation_consistency(candidates: list[MatchCandidate], dominant: float, threshold_deg: float) -> float:
    """统计最终 inliers 中方向差接近主方向差的比例。"""

    if not candidates:
        return 0.0
    ok = [abs(wrap_angle_deg(item.angle_delta - dominant)) <= threshold_deg for item in candidates]
    return float(np.mean(ok))


def candidate_debug_rows(candidates: list[MatchCandidate], query_xy: np.ndarray, gallery_xy: np.ndarray, errors: np.ndarray | None = None) -> list[dict[str, Any]]:
    """把候选匹配转成可视化脚本使用的轻量字典。"""

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        qx, qy = query_xy[candidate.query_idx].tolist()
        gx, gy = gallery_xy[candidate.gallery_idx].tolist()
        row = {
            "query_idx": int(candidate.query_idx),
            "gallery_idx": int(candidate.gallery_idx),
            "query_xy": [float(qx), float(qy)],
            "gallery_xy": [float(gx), float(gy)],
            "distance": float(candidate.distance),
            "angle_delta": float(candidate.angle_delta),
        }
        if errors is not None and index < len(errors):
            row["reproj_error"] = float(errors[index])
        rows.append(row)
    return rows


def empty_debug_matches() -> dict[str, Any]:
    """构造空调试匹配信息。"""

    return {"candidates": [], "raw_inliers": [], "unique_inliers": []}


def empty_result(query: dict[str, Any], gallery: dict[str, Any]) -> dict[str, Any]:
    """构造失败或空匹配时的默认返回值。"""

    nq = int(np.asarray(query.get("keypoints_xy", np.zeros((0, 2)))).shape[0])
    ng = int(np.asarray(gallery.get("keypoints_xy", np.zeros((0, 2)))).shape[0])
    return {
        "score": 0.0,
        "raw_score": 0.0,
        "quality_score": 0.0,
        "num_keypoints_q": nq,
        "num_keypoints_g": ng,
        "num_candidates": 0,
        "num_raw_matches": 0,
        "num_inliers": 0,
        "raw_inliers": 0,
        "unique_inliers": 0,
        "unique_query_inliers": 0,
        "unique_gallery_inliers": 0,
        "inlier_ratio": 0.0,
        "mean_l2_distance": 0.0,
        "mean_reproj_error": 0.0,
        "orientation_consistency": 0.0,
        "dominant_angle_delta": 0.0,
        "mean_similarity": 0.0,
        "affine_matrix": None,
        "affine_scale": 0.0,
        "scale_rejected": False,
        "texture_enabled": False,
        "texture_evaluated": False,
        "texture_available": False,
        "texture_similarity": 0.0,
        "texture_overlap_fraction": 0.0,
        "texture_valid_blocks": 0,
        "geometry_similarity": 0.0,
        "geometry_weight": 1.0,
        "texture_weight": 0.0,
        "texture_decision": "not_reached",
    }


def match_templates_descriptor_l2(query: dict[str, Any], gallery: dict[str, Any], config: dict[str, Any], descriptor_source: str = "hardnet", include_debug: bool = False) -> dict[str, Any]:
    """使用指定 descriptor 匹配一对图像模板并返回图像级分数。

    主流程：
        1. 读取并归一化 descriptor；
        2. 生成 L2/top-k/ratio 候选；
        3. 候选过多时做方向软门控；
        4. 用 cv2.estimateAffinePartial2D 做 RANSAC；
        5. 对 RANSAC inliers 做 one-to-one 去重；
        6. 用 unique_inliers 作为主分数。
    """

    match_started = time.perf_counter()
    timings = {
        "descriptor_prepare_ms": 0.0,
        "candidate_generation_ms": 0.0,
        "candidate_filter_ms": 0.0,
        "ransac_ms": 0.0,
        "inlier_refinement_ms": 0.0,
        "texture_similarity_ms": 0.0,
        "score_fusion_ms": 0.0,
        "postprocess_ms": 0.0,
        "image_match_total_ms": 0.0,
    }
    cfg = dict(config.get("matching", {}))
    base = empty_result(query, gallery)
    base["texture_enabled"] = bool(dict(config.get("texture_verification", {})).get("enabled", False))
    if base["texture_enabled"]:
        base["texture_decision"] = "geometry_not_available"
    source = str(descriptor_source).lower()

    def finish(result: dict[str, Any], debug: dict[str, Any] | None = None) -> dict[str, Any]:
        timings["image_match_total_ms"] = (
            time.perf_counter() - match_started
        ) * 1000.0
        result.update(timings)
        if include_debug:
            result = {**result, "debug_matches": debug or empty_debug_matches()}
        return result

    stage_started = time.perf_counter()
    desc_q = select_l2_descriptors(query, source)
    desc_g = select_l2_descriptors(gallery, source)
    timings["descriptor_prepare_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_keypoints_q"] = int(desc_q.shape[0])
    base["num_keypoints_g"] = int(desc_g.shape[0])
    base["descriptor_source"] = source
    if desc_q.shape[0] < 2 or desc_g.shape[0] < 2:
        return finish(base)

    query_xy = np.asarray(query["keypoints_xy"], dtype=np.float32)
    gallery_xy = np.asarray(gallery["keypoints_xy"], dtype=np.float32)

    # 阶段 1：L2 候选生成。
    stage_started = time.perf_counter()
    candidates = build_l2_candidates(query, gallery, cfg, descriptor_source=source)
    timings["candidate_generation_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_raw_matches"] = int(len(candidates))

    # 阶段 2：候选过多时做软截断，并记录主方向差作为诊断指标。
    stage_started = time.perf_counter()
    candidates_for_ransac, dominant = soft_gate_candidates(candidates, cfg)
    timings["candidate_filter_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_candidates"] = int(len(candidates_for_ransac))
    base["dominant_angle_delta"] = float(dominant)
    # partial affine 至少需要两对坐标；这是算法的固定数学条件，不是打分阈值。
    if len(candidates_for_ransac) < 2:
        return finish(base, {"candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy), "raw_inliers": [], "unique_inliers": []})

    # 阶段 3：RANSAC 几何验证。这里的输入仍允许 many-to-one。
    stage_started = time.perf_counter()
    src, dst = points_from_candidates(candidates_for_ransac, query_xy, gallery_xy)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.get("ransac_reproj_threshold", 5.0)),
        maxIters=int(cfg.get("ransac_max_iters", 3000)),
        confidence=float(cfg.get("ransac_confidence", 0.995)),
        refineIters=10,
    )
    timings["ransac_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    if matrix is None or inlier_mask is None:
        return finish(base, {"candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy), "raw_inliers": [], "unique_inliers": []})

    stage_started = time.perf_counter()
    scale_ok, affine_scale = scale_in_allowed_range(np.asarray(matrix, dtype=np.float64), cfg)
    base["affine_scale"] = float(affine_scale) if math.isfinite(affine_scale) else 0.0
    if not scale_ok:
        base["scale_rejected"] = True
        timings["inlier_refinement_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        return finish(base, {"candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy), "raw_inliers": [], "unique_inliers": []})

    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    raw_inliers = [candidate for candidate, keep in zip(candidates_for_ransac, mask) if bool(keep)]
    base["num_inliers"] = int(len(raw_inliers))
    base["raw_inliers"] = int(len(raw_inliers))
    if len(raw_inliers) == 0:
        timings["inlier_refinement_ms"] = (
            time.perf_counter() - stage_started
        ) * 1000.0
        return finish(base, {"candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy), "raw_inliers": [], "unique_inliers": []})

    # 阶段 4：把 RANSAC 原始 inliers 清理成真正的一对一 inliers。
    kept, stage1_errors = unique_inliers(raw_inliers, query_xy, gallery_xy, np.asarray(matrix, dtype=np.float64), cfg)
    if len(kept) >= 2:
        kept_src, kept_dst = points_from_candidates(kept, query_xy, gallery_xy)
        refined = estimate_partial_affine_ls(kept_src, kept_dst)
        if refined is not None:
            refined_scale_ok, refined_scale = scale_in_allowed_range(refined, cfg)
            if not refined_scale_ok:
                base["affine_scale"] = float(refined_scale) if math.isfinite(refined_scale) else 0.0
                base["scale_rejected"] = True
                timings["inlier_refinement_ms"] = (
                    time.perf_counter() - stage_started
                ) * 1000.0
                return finish(base, {"candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy), "raw_inliers": [], "unique_inliers": []})
            matrix = refined
            affine_scale = refined_scale
            final_errors = reprojection_errors(matrix, kept_src, kept_dst)
        else:
            final_errors = stage1_errors
    else:
        final_errors = stage1_errors

    unique_count = int(len(kept))
    distances = np.asarray([item.distance for item in kept], dtype=np.float32)
    mean_l2 = float(np.mean(distances)) if distances.size else 0.0
    mean_reproj = float(np.mean(final_errors)) if final_errors.size else 0.0
    inlier_ratio = float(unique_count / max(len(candidates_for_ransac), 1))
    orient = orientation_consistency(
        kept,
        dominant,
        float(cfg.get("orientation_consistency_thresh_deg", 20.0)),
    )
    timings["inlier_refinement_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    # 阶段 5：unique inlier 数仅作为原始诊断值；最终 score 是 [0,1] 连续融合分数。
    raw_score = float(unique_count)
    stage_started = time.perf_counter()
    score, texture_diagnostics = compute_fused_match_score(
        query,
        gallery,
        np.asarray(matrix, dtype=np.float64),
        unique_count,
        config,
    )
    score_elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
    timings["texture_similarity_ms"] = float(
        texture_diagnostics.get("texture_similarity_ms", 0.0)
    )
    timings["score_fusion_ms"] = max(
        0.0,
        score_elapsed_ms - timings["texture_similarity_ms"],
    )
    stage_started = time.perf_counter()
    quality_score = (
        0.55 * min(unique_count / 40.0, 1.0)
        + 0.20 * min(inlier_ratio, 1.0)
        + 0.15 * min(max(1.0 - mean_l2 / 1.4, 0.0), 1.0)
        + 0.10 * min(max(1.0 - mean_reproj / 5.0, 0.0), 1.0)
    )
    raw_src, raw_dst = points_from_candidates(raw_inliers, query_xy, gallery_xy)
    raw_errors = reprojection_errors(np.asarray(matrix, dtype=np.float64), raw_src, raw_dst) if raw_inliers else np.zeros((0,), dtype=np.float32)
    debug_matches = {
        "candidates": candidate_debug_rows(candidates_for_ransac, query_xy, gallery_xy),
        "raw_inliers": candidate_debug_rows(raw_inliers, query_xy, gallery_xy, raw_errors),
        "unique_inliers": candidate_debug_rows(kept, query_xy, gallery_xy, final_errors),
    }
    timings["postprocess_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0

    return finish({
        **base,
        "score": float(score),
        "raw_score": float(raw_score),
        "quality_score": float(quality_score),
        "num_candidates": int(len(candidates_for_ransac)),
        "num_raw_matches": int(len(candidates)),
        "num_inliers": int(len(raw_inliers)),
        "raw_inliers": int(len(raw_inliers)),
        "unique_inliers": unique_count,
        "unique_query_inliers": int(len({item.query_idx for item in kept})),
        "unique_gallery_inliers": int(len({item.gallery_idx for item in kept})),
        "inlier_ratio": inlier_ratio,
        "mean_l2_distance": mean_l2,
        "mean_reproj_error": mean_reproj,
        "orientation_consistency": orient,
        "dominant_angle_delta": float(dominant),
        "mean_similarity": float(np.mean(1.0 - distances / 2.0)) if distances.size else 0.0,
        "affine_matrix": np.asarray(matrix, dtype=float).tolist(),
        "affine_scale": float(affine_scale),
        "scale_rejected": False,
        "descriptor_source": source,
        **texture_diagnostics,
    }, debug_matches)


def match_templates_hardnet_l2(query: dict[str, Any], gallery: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """兼容旧入口：使用 HardNet descriptor 的 L2 匹配。"""

    return match_templates_descriptor_l2(query, gallery, config, descriptor_source="hardnet")
