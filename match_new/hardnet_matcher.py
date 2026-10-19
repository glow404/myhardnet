"""HardNet 图像级匹配器。工具脚本

作用：
    对一张 query 指纹图模板和一张 gallery/template 指纹图模板进行匹配，
    输出图像级分数和诊断指标。距离由模板契约自动选择：

    1. float32 描述子使用 L2；packed_uint8 二值描述子使用归一化 Hamming；
    2. 用 top-k / ratio 生成较宽松候选；
    3. 候选过多时按描述子距离截断；
    4. 用 RANSAC 估计局部仿射变换；
    5. 对 RANSAC inliers 做 one-to-one 去重；
    6. 融合归一化 unique inlier 分数与脊线纹理分数，得到图像级 match score。

输入：
    query/gallery 都是 `.npz` 模板读出的 dict，至少包含：
        - keypoints_xy
        - keypoints_angle
        - hardnet_descriptors

输出：
    一个 dict，包含 score、unique_inliers、raw_inliers、描述子距离、
    mean_reproj_error、orientation_consistency 等字段。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from match_new.descriptor_contract import (
    BINARY_DESCRIPTOR_KIND,
    HAMMING_DISTANCE_METRIC,
    L2_DISTANCE_METRIC,
    DescriptorContract,
    descriptor_columns,
    require_compatible_contracts,
    require_l2_float_contract,
    require_supported_contract,
    resolve_descriptor_contract,
)


@dataclass(frozen=True)
class MatchCandidate:
    """一个 HardNet 候选匹配。

    query_idx / gallery_idx 指向两张图中的 keypoint 下标；
    distance 是当前模板契约对应的描述子距离，越小越像；
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


def descriptor_array_key(descriptor_source: str) -> tuple[str, str]:
    """把描述子来源映射到模板数组字段和可用性标记。"""

    source = str(descriptor_source).strip().lower()
    if source == "hardnet":
        return "hardnet_descriptors", "has_hardnet"
    if source in {"sift", "rootsift"}:
        return "sift_descriptors", "has_sift"
    raise ValueError(f"unsupported descriptor_source: {descriptor_source}")


def select_descriptors(
    template: dict[str, Any],
    descriptor_source: str,
) -> tuple[np.ndarray, DescriptorContract]:
    """按模板契约读取浮点或打包二值描述子。"""

    key, has_flag = descriptor_array_key(descriptor_source)
    if has_flag in template and not bool(template.get(has_flag)):
        template_path = template.get("template_path", "<unknown>")
        raise ValueError(
            f"template has no {key}: {template_path}. "
            "请使用对应的单描述子模板目录，不要复用错误类型的模板。"
        )

    contract = resolve_descriptor_contract(template, descriptor_source)
    label = str(
        template.get(
            "template_path",
            template.get("image_id", "<in-memory-template>"),
        )
    )
    require_supported_contract(contract, label=label)
    dtype = np.uint8 if contract.kind == BINARY_DESCRIPTOR_KIND else np.float32
    raw = template.get(key)
    desc = np.asarray(raw if raw is not None else [], dtype=dtype)
    expected_columns = descriptor_columns(contract)
    if desc.size == 0:
        if desc.ndim == 2 and desc.shape[1] not in {0, expected_columns}:
            raise ValueError(
                f"{key} metadata/array dimension mismatch for {label}: "
                f"contract={contract.dimension}, array={desc.shape}."
            )
        return np.zeros((0, expected_columns), dtype=dtype), contract
    if desc.ndim != 2 or desc.shape[1] != expected_columns:
        raise ValueError(
            f"{key} must be [N,{expected_columns}] for {label}, got {desc.shape}. "
            "Implicit reshape, truncation and padding are not supported."
        )
    if contract.kind != BINARY_DESCRIPTOR_KIND:
        if not np.all(np.isfinite(desc)):
            raise ValueError(f"{key} contains NaN or infinite values: {label}")
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        desc = desc / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(desc), contract


def select_l2_descriptors(template: dict[str, Any], descriptor_source: str) -> np.ndarray:
    """兼容旧调用：读取并归一化连续描述子。"""

    descriptors, contract = select_descriptors(template, descriptor_source)
    require_l2_float_contract(contract, label=str(template.get("template_path", "template")))
    return descriptors


def require_compatible_templates(
    query: dict[str, Any],
    gallery: dict[str, Any],
    descriptor_source: str,
) -> DescriptorContract:
    """验证 query/gallery 契约一致，并返回统一契约。"""

    query_contract = resolve_descriptor_contract(query, descriptor_source)
    gallery_contract = resolve_descriptor_contract(gallery, descriptor_source)
    require_supported_contract(query_contract, label="query")
    require_supported_contract(gallery_contract, label="gallery")
    require_compatible_contracts(query_contract, gallery_contract)
    return query_contract


def require_compatible_l2_templates(
    query: dict[str, Any],
    gallery: dict[str, Any],
    descriptor_source: str,
) -> None:
    """兼容旧调用：验证 query/gallery 都是相同的浮点 L2 契约。"""

    contract = require_compatible_templates(query, gallery, descriptor_source)
    require_l2_float_contract(contract, label="query/gallery")


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


# ---------------------------------------------------------------------------
# 描述子候选后端
# ---------------------------------------------------------------------------

_l2_candidate_device: torch.device | None = None
_HAMMING_BACKENDS = {"auto", "cpu", "cuda"}


def _resolve_l2_candidate_device() -> torch.device:
    """解析浮点 L2 候选生成设备，并缓存结果。"""

    global _l2_candidate_device
    if _l2_candidate_device is None:
        _l2_candidate_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _l2_candidate_device


def _resolve_hamming_backend(raw_backend: str | None) -> str:
    """解析 packed-Hamming 后端。

    ``auto`` 与默认值都选择 OpenCV CPU。当前工程的典型矩阵只有数百行，
    CPU SIMD/POPCNT 明显快于逐模板 CUDA 调度。``cuda`` 仅作为显式实验开关；
    CUDA 不可用时安全回退到 CPU。
    """

    backend = str(raw_backend or "cpu").strip().lower()
    if backend not in _HAMMING_BACKENDS:
        supported = ", ".join(sorted(_HAMMING_BACKENDS))
        raise ValueError(
            f"unsupported Hamming backend: {backend!r}; expected one of {supported}."
        )
    if backend == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _l2_distance_torch(
    query: np.ndarray,
    gallery: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """在 GPU（或 CPU torch）上计算全量 L2 距离矩阵。"""

    q_t = torch.from_numpy(query.astype(np.float32)).to(device)
    g_t = torch.from_numpy(gallery.astype(np.float32)).to(device)
    dist = torch.cdist(q_t, g_t, p=2.0)
    return dist.cpu().numpy().astype(np.float32)


def _l2_reverse_neighbors_torch(
    distances: np.ndarray,
    k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """在 GPU 上并行计算距离矩阵每一列的 top-k 最小值和下标。

    返回 ``(best_indices, best_values)``，形状均为 ``(gallery_count, k)``。
    """

    d_t = torch.from_numpy(distances.astype(np.float32)).to(device)
    topk_vals, topk_indices = torch.topk(
        d_t.T,
        k=min(k, d_t.shape[0]),
        largest=False,
    )
    return (
        topk_indices.cpu().numpy().astype(np.int64),
        topk_vals.cpu().numpy().astype(np.float32),
    )


def _hamming_distance_counts_torch(
    query: np.ndarray,
    gallery: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """直接在 packed uint8 上计算 Hamming bit count 距离矩阵。

    保留紧凑二值表示，不再展开为 float32 位向量。距离矩阵驻留 GPU，供正向
    和反向 top-k 共用，避免完整矩阵在 CPU/GPU 之间往返。
    """

    query_t = torch.from_numpy(
        np.ascontiguousarray(query, dtype=np.uint8)
    ).to(device)
    gallery_t = torch.from_numpy(
        np.ascontiguousarray(gallery, dtype=np.uint8)
    ).to(device)
    xor = torch.bitwise_xor(query_t[:, None, :], gallery_t[None, :, :])
    counts = xor - ((xor >> 1) & 0x55)
    counts = (counts & 0x33) + ((counts >> 2) & 0x33)
    counts = (counts + (counts >> 4)) & 0x0F
    return counts.sum(dim=2, dtype=torch.int32)


def _hamming_topk_torch(
    distance_counts: torch.Tensor,
    hash_bits: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对 GPU 上的 Hamming bit count 矩阵取行方向 top-k。"""

    neighbors = min(max(int(k), 0), int(distance_counts.shape[1]))
    shape = (int(distance_counts.shape[0]), 0)
    if neighbors == 0:
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.int32),
        )
    column_count = int(distance_counts.shape[1])
    tie_break = torch.arange(
        column_count,
        dtype=distance_counts.dtype,
        device=distance_counts.device,
    )
    ranking_keys = distance_counts * (column_count + 1) + tie_break
    _, topk_indices = torch.topk(
        ranking_keys,
        k=neighbors,
        largest=False,
    )
    topk_counts = torch.gather(distance_counts, 1, topk_indices)
    return (
        (topk_counts.to(torch.float32) / float(hash_bits))
        .cpu()
        .numpy()
        .astype(np.float32),
        topk_indices.cpu().numpy().astype(np.int32),
    )


def _hamming_knn_torch(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """在 GPU packed uint8 距离矩阵上返回归一化 Hamming top-k。"""

    distance_counts = _hamming_distance_counts_torch(query, gallery, device)
    return _hamming_topk_torch(distance_counts, hash_bits, k)


def _hamming_bidirectional_knn_torch(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    forward_k: int,
    reverse_k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """计算一次 GPU 距离矩阵，同时取得正向和反向近邻。"""

    distance_counts = _hamming_distance_counts_torch(query, gallery, device)
    forward_distances, forward_indices = _hamming_topk_torch(
        distance_counts,
        hash_bits,
        forward_k,
    )
    if reverse_k <= 0:
        return forward_distances, forward_indices, None, None
    reverse_distances, reverse_indices = _hamming_topk_torch(
        distance_counts.T,
        hash_bits,
        reverse_k,
    )
    return (
        forward_distances,
        forward_indices,
        reverse_distances,
        reverse_indices,
    )


# ---------------------------------------------------------------------------

def pairwise_l2(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """计算 query descriptors 到 gallery descriptors 的全量 L2 距离矩阵。

    对 L2-normalized descriptor 来说，L2 排序与 cosine 排序等价；
    但 L2 距离范围更直观，便于设置 abs_distance_threshold。

    GPU 可用时自动使用 ``torch.cdist`` 加速；CUDA 不可用或失败时回退 NumPy。
    """

    if query.ndim != 2 or gallery.ndim != 2:
        raise ValueError(
            f"pairwise_l2 expects [N,D] arrays, got {query.shape} and {gallery.shape}."
        )
    if query.shape[1] != gallery.shape[1]:
        raise ValueError(
            "pairwise_l2 descriptor dimension mismatch: "
            f"query={query.shape[1]}, gallery={gallery.shape[1]}."
        )
    device = _resolve_l2_candidate_device()
    if device.type == "cuda":
        try:
            return _l2_distance_torch(query, gallery, device)
        except RuntimeError:
            pass
    q2 = np.sum(query * query, axis=1, keepdims=True)
    g2 = np.sum(gallery * gallery, axis=1, keepdims=True).T
    d2 = np.maximum(q2 + g2 - 2.0 * query @ gallery.T, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def _validate_hamming_inputs(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    *,
    function_name: str,
) -> int:
    """校验 packed-Hamming 数组契约并返回规范化 bit 数。"""

    if query.ndim != 2 or gallery.ndim != 2:
        raise ValueError(
            f"{function_name} expects [N,B] arrays, "
            f"got {query.shape} and {gallery.shape}."
        )
    if query.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"{function_name} packed width mismatch: "
            f"query={query.shape[1]}, gallery={gallery.shape[1]}."
        )
    bits = int(hash_bits)
    if bits <= 0 or bits % 8 != 0 or query.shape[1] != bits // 8:
        raise ValueError(
            f"{function_name} requires byte-aligned hash bits: "
            f"hash_bits={bits}, width={query.shape[1]}."
        )
    return bits


def pairwise_hamming(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    *,
    backend: str = "cpu",
) -> np.ndarray:
    """计算归一化 Hamming 全距离矩阵。

    默认使用 OpenCV CPU SIMD/POPCNT。只有显式配置 ``backend=cuda`` 时才走
    packed uint8 CUDA 路径；CUDA 不可用或执行失败时安全回退 CPU。
    """

    bits = _validate_hamming_inputs(
        query,
        gallery,
        hash_bits,
        function_name="pairwise_hamming",
    )
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        return np.zeros(
            (int(query.shape[0]), int(gallery.shape[0])),
            dtype=np.float32,
        )

    if _resolve_hamming_backend(backend) == "cuda":
        try:
            distance_counts = _hamming_distance_counts_torch(
                query,
                gallery,
                torch.device("cuda"),
            )
            return (
                distance_counts.to(torch.float32)
                .div(float(bits))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        except RuntimeError:
            pass

    sorted_distances, sorted_indices = cv2.batchDistance(
        np.ascontiguousarray(query, dtype=np.uint8),
        np.ascontiguousarray(gallery, dtype=np.uint8),
        cv2.CV_32S,
        normType=cv2.NORM_HAMMING,
        K=int(gallery.shape[0]),
    )
    distances = np.empty(sorted_distances.shape, dtype=np.float32)
    np.put_along_axis(
        distances,
        np.asarray(sorted_indices, dtype=np.int32),
        np.asarray(sorted_distances, dtype=np.float32) / float(bits),
        axis=1,
    )
    return distances


def hamming_knn(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    k: int,
    *,
    backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """返回归一化 Hamming top-k 距离和近邻下标矩阵。"""

    bits = _validate_hamming_inputs(
        query,
        gallery,
        hash_bits,
        function_name="hamming_knn",
    )
    neighbors = min(max(int(k), 0), int(gallery.shape[0]))
    if query.shape[0] == 0 or neighbors == 0:
        shape = (int(query.shape[0]), 0)
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.int32),
        )

    if _resolve_hamming_backend(backend) == "cuda":
        try:
            return _hamming_knn_torch(
                query,
                gallery,
                bits,
                neighbors,
                torch.device("cuda"),
            )
        except RuntimeError:
            pass

    distances, indices = cv2.batchDistance(
        np.ascontiguousarray(query, dtype=np.uint8),
        np.ascontiguousarray(gallery, dtype=np.uint8),
        cv2.CV_32S,
        normType=cv2.NORM_HAMMING,
        K=neighbors,
    )
    return (
        np.asarray(distances, dtype=np.float32) / float(bits),
        np.asarray(indices, dtype=np.int32),
    )


def hamming_bidirectional_knn(
    query: np.ndarray,
    gallery: np.ndarray,
    hash_bits: int,
    forward_k: int,
    reverse_k: int,
    *,
    backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """返回正反两个方向的 Hamming 近邻。

    显式 CUDA 路径只计算一次距离矩阵，再分别对行和列取 top-k；CPU 路径继续
    使用 OpenCV packed-Hamming，避免为小矩阵引入 GPU 调度和传输开销。
    """

    bits = _validate_hamming_inputs(
        query,
        gallery,
        hash_bits,
        function_name="hamming_bidirectional_knn",
    )
    forward_neighbors = min(max(int(forward_k), 0), int(gallery.shape[0]))
    reverse_neighbors = min(max(int(reverse_k), 0), int(query.shape[0]))
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        forward_shape = (int(query.shape[0]), 0)
        forward_distances = np.zeros(forward_shape, dtype=np.float32)
        forward_indices = np.zeros(forward_shape, dtype=np.int32)
        if reverse_k <= 0:
            return forward_distances, forward_indices, None, None
        reverse_shape = (int(gallery.shape[0]), 0)
        return (
            forward_distances,
            forward_indices,
            np.zeros(reverse_shape, dtype=np.float32),
            np.zeros(reverse_shape, dtype=np.int32),
        )

    if _resolve_hamming_backend(backend) == "cuda":
        try:
            return _hamming_bidirectional_knn_torch(
                query,
                gallery,
                bits,
                forward_neighbors,
                reverse_neighbors,
                torch.device("cuda"),
            )
        except RuntimeError:
            pass

    forward_distances, forward_indices = hamming_knn(
        query,
        gallery,
        bits,
        forward_neighbors,
        backend="cpu",
    )
    if reverse_k <= 0:
        return forward_distances, forward_indices, None, None
    reverse_distances, reverse_indices = hamming_knn(
        gallery,
        query,
        bits,
        reverse_neighbors,
        backend="cpu",
    )
    return (
        forward_distances,
        forward_indices,
        reverse_distances,
        reverse_indices,
    )


def pairwise_descriptor_distance(
    query: np.ndarray,
    gallery: np.ndarray,
    contract: DescriptorContract,
) -> np.ndarray:
    """按描述子契约选择 L2 或归一化 Hamming 距离。"""

    if contract.metric == L2_DISTANCE_METRIC:
        return pairwise_l2(query, gallery)
    if contract.metric == HAMMING_DISTANCE_METRIC:
        return pairwise_hamming(query, gallery, contract.dimension)
    raise ValueError(f"unsupported descriptor metric: {contract.metric}")


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


def metric_matching_config(
    config: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """校验距离契约，并在 Hamming 模式下应用二值专用阈值。"""

    configured = str(config.get("distance", "auto")).strip().lower()
    if configured == "euclidean":
        configured = L2_DISTANCE_METRIC
    if configured not in {"", "auto", metric}:
        raise ValueError(
            "matching.distance does not match template/checkpoint metric: "
            f"config={configured}, descriptor={metric}."
        )
    overrides = (
        config.get(HAMMING_DISTANCE_METRIC, {})
        if metric == HAMMING_DISTANCE_METRIC
        else {}
    )
    if not isinstance(overrides, dict):
        raise ValueError("matching.hamming must be a mapping.")
    return {**config, **overrides, "distance": metric}


def build_descriptor_candidates(
    query: dict[str, Any],
    gallery: dict[str, Any],
    cfg: dict[str, Any],
    descriptor_source: str = "hardnet",
    diagnostics: dict[str, Any] | None = None,
) -> list[MatchCandidate]:
    """生成 RANSAC 前的描述子候选，支持 L2 与归一化 Hamming。

    支持三种策略：
        - ratio_only：只要最近邻通过 Lowe ratio；
        - topk_only：保留 top-k 中距离足够近的候选；
        - topk_or_ratio：ratio 通过则取最近邻，否则退回 top-k。

    当 bidirectional_ratio_test=true 时，经 ratio 分支保留的最近邻还必须满足：
        1. gallery -> query 方向也通过 Lowe ratio；
        2. 两个方向的最近邻互相指向同一对关键点。

    这里默认允许 many-to-one，因为指纹局部纹理重复，过早强制一对一会删掉真匹配。
    """

    contract = require_compatible_templates(query, gallery, descriptor_source)
    desc_q, query_contract = select_descriptors(query, descriptor_source)
    desc_g, gallery_contract = select_descriptors(gallery, descriptor_source)
    require_compatible_contracts(query_contract, gallery_contract)
    if desc_q.shape[1] != desc_g.shape[1]:
        raise ValueError(
            "Query/gallery descriptor storage width mismatch: "
            f"query={desc_q.shape[1]}, gallery={desc_g.shape[1]}."
        )
    cfg = metric_matching_config(cfg, contract.metric)
    if diagnostics is not None:
        diagnostics.update(
            {
                "num_query_descriptors": int(desc_q.shape[0]),
                "num_gallery_descriptors": int(desc_g.shape[0]),
                "num_abs_distance_pass": 0,
                "num_one_way_ratio_pass": 0,
                "num_ratio_pass": 0,
                "num_queries_with_candidates": 0,
                "candidate_query_coverage": 0.0,
            }
        )
    if desc_q.shape[0] == 0 or desc_g.shape[0] == 0:
        return []

    q_angles = keypoint_angles(query, desc_q.shape[0])
    g_angles = keypoint_angles(gallery, desc_g.shape[0])
    policy = str(cfg.get("candidate_policy", "topk_or_ratio")).lower()
    top_k = max(1, int(cfg.get("top_k", 5)))
    ratio_threshold = float(cfg.get("ratio_threshold", 0.95))
    bidirectional_ratio = bool(cfg.get("bidirectional_ratio_test", False))
    abs_threshold = float(cfg.get("abs_distance_threshold", 1.10))
    margin = float(cfg.get("distance_margin", 0.12))
    allow_many = bool(cfg.get("allow_many_to_one_before_ransac", True))
    forward_k = min(max(top_k, 2), int(desc_g.shape[0]))
    needs_bidirectional_ratio = bool(
        bidirectional_ratio
        and policy in {"ratio_only", "topk_or_ratio"}
    )
    reverse_k = min(2, int(desc_q.shape[0])) if needs_bidirectional_ratio else 0

    reverse_distances: np.ndarray | None = None
    reverse_indices: np.ndarray | None = None
    if contract.metric == HAMMING_DISTANCE_METRIC:
        # CPU 默认直接使用 OpenCV packed-Hamming。显式 CUDA 时只计算一次距离
        # 矩阵，并从同一矩阵取得正向和反向近邻。
        (
            forward_distances,
            forward_indices,
            reverse_distances,
            reverse_indices,
        ) = hamming_bidirectional_knn(
            desc_q,
            desc_g,
            contract.dimension,
            forward_k,
            reverse_k,
            backend=str(cfg.get("backend", "cpu")),
        )
        distances = None
    else:
        distances = pairwise_descriptor_distance(desc_q, desc_g, contract)
        forward_distances = None
        forward_indices = None

    # 双向 Lowe 验证需要预先得到每个 gallery 描述子的 query 侧最近邻和次近邻。
    reverse_best_query = np.full((desc_g.shape[0],), -1, dtype=np.int64)
    reverse_ratio_ok = np.zeros((desc_g.shape[0],), dtype=bool)
    if needs_bidirectional_ratio:
        if contract.metric == HAMMING_DISTANCE_METRIC:
            assert reverse_distances is not None
            assert reverse_indices is not None
            reverse_best_query[:] = reverse_indices[:, 0].astype(np.int64)
            best_vals = reverse_distances[:, 0]
            second_vals = (
                reverse_distances[:, 1]
                if reverse_distances.shape[1] > 1
                else np.full(
                    reverse_distances.shape[0],
                    np.inf,
                    dtype=np.float32,
                )
            )
            reverse_ratio_ok[:] = (
                (second_vals > 1e-12)
                & (best_vals < ratio_threshold * second_vals)
            )
        else:
            assert distances is not None
            device = _resolve_l2_candidate_device()
            if device.type == "cuda":
                try:
                    # GPU 并行：对距离矩阵的每一列（gallery 维度）
                    # 取 top-2 最小值和下标，替代逐列 argpartition。
                    rev_indices, rev_vals = _l2_reverse_neighbors_torch(
                        distances,
                        reverse_k,
                        device,
                    )
                    reverse_best_query[:] = rev_indices[:, 0].astype(np.int64)
                    best_vals = rev_vals[:, 0]
                    second_vals = (
                        rev_vals[:, 1]
                        if rev_indices.shape[1] > 1
                        else np.full(rev_indices.shape[0], np.inf, dtype=np.float32)
                    )
                    reverse_ratio_ok[:] = (
                        (second_vals > 1e-12)
                        & (best_vals < ratio_threshold * second_vals)
                    )
                except RuntimeError:
                    device = torch.device("cpu")
            if device.type != "cuda":
                for gallery_idx in range(distances.shape[1]):
                    column = distances[:, gallery_idx]
                    reverse_order = top_indices(column, reverse_k)
                    reverse_best_idx = int(reverse_order[0])
                    reverse_best_dist = float(column[reverse_best_idx])
                    reverse_second_dist = (
                        float(column[int(reverse_order[1])])
                        if reverse_order.size > 1
                        else float("inf")
                    )
                    reverse_best_query[gallery_idx] = reverse_best_idx
                    reverse_ratio_ok[gallery_idx] = bool(
                        reverse_second_dist > 1e-12
                        and reverse_best_dist
                        < ratio_threshold * reverse_second_dist
                    )

    # candidates 保存所有进入后续几何验证的候选；
    # best_gallery 只在 allow_many=false 时使用，用于提前做 gallery 侧去重。
    candidates: list[MatchCandidate] = []
    best_gallery: dict[int, MatchCandidate] = {}
    num_abs_distance_pass = 0
    num_one_way_ratio_pass = 0
    num_ratio_pass = 0
    for query_idx in range(desc_q.shape[0]):
        if contract.metric == HAMMING_DISTANCE_METRIC:
            assert forward_distances is not None
            assert forward_indices is not None
            order = forward_indices[query_idx]
            ordered_distances = forward_distances[query_idx]
            best_idx = int(order[0])
            best_dist = float(ordered_distances[0])
            second_dist = (
                float(ordered_distances[1])
                if ordered_distances.size > 1
                else float("inf")
            )
        else:
            assert distances is not None
            row = distances[query_idx]
            order = top_indices(row, forward_k)
            best_idx = int(order[0])
            best_dist = float(row[best_idx])
            second_dist = (
                float(row[int(order[1])])
                if order.size > 1
                else float("inf")
            )
        if best_dist <= abs_threshold:
            num_abs_distance_pass += 1
        one_way_ratio_ok = bool(
            second_dist > 1e-12
            and best_dist < ratio_threshold * second_dist
        )
        if one_way_ratio_ok:
            num_one_way_ratio_pass += 1
        ratio_ok = one_way_ratio_ok
        if ratio_ok and bidirectional_ratio:
            ratio_ok = bool(
                reverse_best_query[best_idx] == query_idx
                and reverse_ratio_ok[best_idx]
            )
        if ratio_ok:
            num_ratio_pass += 1
        adaptive_limit = min(abs_threshold, best_dist + margin)

        # 自适应距离上限防止 top-k 把明显过远的候选也塞进 RANSAC。
        keep: list[tuple[int, float]] = []
        if policy == "ratio_only":
            if ratio_ok and best_dist <= abs_threshold:
                keep = [(best_idx, best_dist)]
        else:
            if contract.metric == HAMMING_DISTANCE_METRIC:
                top_neighbors = [
                    (int(gallery_idx), float(distance))
                    for gallery_idx, distance in zip(
                        order[:top_k],
                        ordered_distances[:top_k],
                    )
                ]
            else:
                assert distances is not None
                row = distances[query_idx]
                top_neighbors = [
                    (int(gallery_idx), float(row[int(gallery_idx)]))
                    for gallery_idx in order[:top_k]
                ]
            if policy == "topk_only":
                keep = [
                    (gallery_idx, distance)
                    for gallery_idx, distance in top_neighbors
                    if distance <= adaptive_limit
                ]
            elif policy == "topk_or_ratio":
                if ratio_ok and best_dist <= abs_threshold:
                    keep = [(best_idx, best_dist)]
                else:
                    keep = [
                        (gallery_idx, distance)
                        for gallery_idx, distance in top_neighbors
                        if distance <= adaptive_limit
                    ]
            else:
                raise ValueError(f"unsupported candidate_policy: {policy}")

        for gallery_idx, distance in keep:
            candidate = MatchCandidate(
                query_idx=int(query_idx),
                gallery_idx=int(gallery_idx),
                distance=float(distance),
                angle_delta=wrap_angle_deg(float(g_angles[gallery_idx]) - float(q_angles[query_idx])),
            )
            if allow_many:
                candidates.append(candidate)
            else:
                previous = best_gallery.get(candidate.gallery_idx)
                if previous is None or candidate.distance < previous.distance:
                    best_gallery[candidate.gallery_idx] = candidate

    selected_candidates = candidates if allow_many else list(best_gallery.values())
    if diagnostics is not None:
        num_queries_with_candidates = len(
            {candidate.query_idx for candidate in selected_candidates}
        )
        diagnostics.update(
            {
                "num_abs_distance_pass": num_abs_distance_pass,
                "num_one_way_ratio_pass": num_one_way_ratio_pass,
                "num_ratio_pass": num_ratio_pass,
                "num_queries_with_candidates": num_queries_with_candidates,
                "candidate_query_coverage": float(
                    num_queries_with_candidates / max(desc_q.shape[0], 1)
                ),
            }
        )
    return selected_candidates


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


def truncate_candidates_by_distance(
    candidates: list[MatchCandidate],
    max_candidates: int,
) -> list[MatchCandidate]:
    """候选过多时按描述子距离保留前 N 个；max_candidates<=0 表示不截断。"""

    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    keep = np.argsort([item.distance for item in candidates])[:max_candidates]
    return [candidates[int(i)] for i in keep]


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
        "num_abs_distance_pass": 0,
        "num_one_way_ratio_pass": 0,
        "num_ratio_pass": 0,
        "num_queries_with_candidates": 0,
        "candidate_query_coverage": 0.0,
        "ransac_success": False,
        "num_inliers": 0,
        "raw_inliers": 0,
        "unique_inliers": 0,
        "unique_query_inliers": 0,
        "unique_gallery_inliers": 0,
        "inlier_ratio": 0.0,
        "mean_descriptor_distance": 0.0,
        "mean_l2_distance": 0.0,
        "mean_hamming_distance": 0.0,
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


def match_templates_descriptor(query: dict[str, Any], gallery: dict[str, Any], config: dict[str, Any], descriptor_source: str = "hardnet", include_debug: bool = False) -> dict[str, Any]:
    """使用模板声明的 L2 或 Hamming 距离匹配一对图像模板。

    主流程：
        1. 按描述子契约读取浮点或打包二值 descriptor；
        2. 生成 top-k/ratio 候选；
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
        "unique_inlier_dedup_ms": 0.0,
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
    contract = require_compatible_templates(query, gallery, source)
    desc_q, _query_contract = select_descriptors(query, source)
    desc_g, _gallery_contract = select_descriptors(gallery, source)
    if desc_q.shape[1] != desc_g.shape[1]:
        raise ValueError(
            "Query/gallery descriptor storage width mismatch: "
            f"query={desc_q.shape[1]}, gallery={desc_g.shape[1]}."
        )
    cfg = metric_matching_config(cfg, contract.metric)
    timings["descriptor_prepare_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_keypoints_q"] = int(desc_q.shape[0])
    base["num_keypoints_g"] = int(desc_g.shape[0])
    base["descriptor_source"] = source
    base["descriptor_kind"] = contract.kind
    base["descriptor_metric"] = contract.metric
    base["descriptor_dim"] = contract.dimension
    if desc_q.shape[0] < 2 or desc_g.shape[0] < 2:
        return finish(base)

    query_xy = np.asarray(query["keypoints_xy"], dtype=np.float32)
    gallery_xy = np.asarray(gallery["keypoints_xy"], dtype=np.float32)

    # 阶段 1：按契约生成 L2 或 Hamming 候选。
    stage_started = time.perf_counter()
    candidate_diagnostics: dict[str, Any] = {}
    candidates = build_descriptor_candidates(
        query,
        gallery,
        cfg,
        descriptor_source=source,
        diagnostics=candidate_diagnostics,
    )
    base.update(candidate_diagnostics)
    timings["candidate_generation_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_raw_matches"] = int(len(candidates))

    # 阶段 2：进入 RANSAC 前仅按描述子距离截断。
    stage_started = time.perf_counter()
    candidates_for_ransac = truncate_candidates_by_distance(
        candidates,
        int(cfg.get("max_candidates_for_ransac", 250)),
    )
    timings["candidate_filter_ms"] = (
        time.perf_counter() - stage_started
    ) * 1000.0
    base["num_candidates"] = int(len(candidates_for_ransac))
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

    base["ransac_success"] = True
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
    dedup_started = time.perf_counter()
    kept, stage1_errors = unique_inliers(
        raw_inliers,
        query_xy,
        gallery_xy,
        np.asarray(matrix, dtype=np.float64),
        cfg,
    )
    timings["unique_inlier_dedup_ms"] = (
        time.perf_counter() - dedup_started
    ) * 1000.0
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
    mean_distance = float(np.mean(distances)) if distances.size else 0.0
    mean_reproj = float(np.mean(final_errors)) if final_errors.size else 0.0
    inlier_ratio = float(unique_count / max(len(candidates_for_ransac), 1))
    # 方向只在最终内点上统计，用于诊断，不参与候选截断或打分。
    dominant = dominant_angle_delta(
        kept,
        float(cfg.get("orientation_hist_bin_deg", 10.0)),
    )
    base["dominant_angle_delta"] = float(dominant)
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
    distance_cap = 1.0 if contract.metric == HAMMING_DISTANCE_METRIC else 2.0
    distance_similarity = min(
        max(1.0 - mean_distance / distance_cap, 0.0),
        1.0,
    )
    quality_score = (
        0.55 * min(unique_count / 40.0, 1.0)
        + 0.20 * min(inlier_ratio, 1.0)
        + 0.15 * distance_similarity
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
        "mean_descriptor_distance": mean_distance,
        "mean_l2_distance": (
            mean_distance if contract.metric == L2_DISTANCE_METRIC else 0.0
        ),
        "mean_hamming_distance": (
            mean_distance if contract.metric == HAMMING_DISTANCE_METRIC else 0.0
        ),
        "mean_reproj_error": mean_reproj,
        "orientation_consistency": orient,
        "dominant_angle_delta": float(dominant),
        "mean_similarity": (
            float(np.mean(1.0 - distances / distance_cap))
            if distances.size
            else 0.0
        ),
        "affine_matrix": np.asarray(matrix, dtype=float).tolist(),
        "affine_scale": float(affine_scale),
        "scale_rejected": False,
        "descriptor_source": source,
        **texture_diagnostics,
    }, debug_matches)


def match_templates_descriptor_l2(query: dict[str, Any], gallery: dict[str, Any], config: dict[str, Any], descriptor_source: str = "hardnet", include_debug: bool = False) -> dict[str, Any]:
    """兼容旧调用名；实际距离由模板契约选择。"""

    return match_templates_descriptor(
        query,
        gallery,
        config,
        descriptor_source=descriptor_source,
        include_debug=include_debug,
    )


def build_l2_candidates(
    query: dict[str, Any],
    gallery: dict[str, Any],
    cfg: dict[str, Any],
    descriptor_source: str = "hardnet",
    diagnostics: dict[str, Any] | None = None,
) -> list[MatchCandidate]:
    """兼容旧调用名；实际距离由模板契约选择。"""

    return build_descriptor_candidates(
        query,
        gallery,
        cfg,
        descriptor_source=descriptor_source,
        diagnostics=diagnostics,
    )


def match_templates_hardnet_l2(query: dict[str, Any], gallery: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """兼容旧入口；实际距离由 HardNet 模板契约选择。"""

    return match_templates_descriptor(query, gallery, config, descriptor_source="hardnet")
