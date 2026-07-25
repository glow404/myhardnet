"""离线阈值评估与在线解锁共用的 identity 级匹配实现。

一个已注册手指由多张图像模板组成。本模块负责逐模板计算图像级分数，再按配置
融合为一个 identity 级分数。离线评估必须遍历全部模板以获得真实完整分数；
在线解锁在最大值类融合下可于任一模板达到固定阈值后提前结束，同时保持接受/
拒绝结论与完整遍历一致。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from match_new.hardnet_matcher import match_templates_descriptor_l2
from match_new.template_builder import load_image_template


def load_template_cached(
    path: str | Path,
    cache: dict[str, dict[str, Any]],
    *,
    require: str | None = None,
) -> dict[str, Any]:
    """加载一份图像模板并放入进程内缓存。

    缓存键同时包含文件路径和描述子类型要求，防止同一路径在 HardNet/SIFT
    两种加载约束下错误复用。在线引擎可在启动阶段预加载注册模板，避免把磁盘
    读取时间计入每次解锁。
    """

    key = str(Path(path).expanduser())
    cache_key = f"{key}::{require or 'any'}"
    if cache_key not in cache:
        cache[cache_key] = load_image_template(key, require=require)
    return cache[cache_key]


def fuse_image_results(results: list[dict[str, Any]], method: str) -> float:
    """把同一手指多张模板的图像级分数融合为 identity 级分数。

    ``max`` 与 ``max_quality_tiebreak`` 属于最大值类融合，可以安全使用固定
    阈值早停；``mean`` 和 ``top3_mean`` 必须取得完整分数集合后才能计算。
    """

    if not results:
        return 0.0
    scores = np.asarray([float(item.get("score", 0.0)) for item in results], dtype=np.float32)
    method = str(method).lower()
    if method == "max":
        return float(np.max(scores))
    if method == "mean":
        return float(np.mean(scores))
    if method == "top3_mean":
        top = np.sort(scores)[-min(3, scores.size) :]
        return float(np.mean(top)) if top.size else 0.0
    if method == "max_quality_tiebreak":
        best = max(
            results,
            key=lambda item: (float(item.get("score", 0.0)), float(item.get("quality_score", 0.0))),
        )
        return float(best.get("score", 0.0))
    raise ValueError(f"unsupported fusion method: {method}")


def score_query_against_identity(
    query_template: dict[str, Any],
    identity: dict[str, Any],
    config: dict[str, Any],
    template_cache: dict[str, dict[str, Any]],
    descriptor_source: str = "hardnet",
    *,
    early_stop_threshold: float | None = None,
) -> dict[str, Any]:
    """计算一份查询模板与一个已注册手指模板集合的匹配结果。

    ``early_stop_threshold=None`` 时始终遍历全部注册模板，用于离线阈值扫描，
    从而保留真实 identity 最大分数。在线解锁显式传入离线标定的固定阈值；
    在最大值类融合下，一旦某张模板分数达到阈值，最终“接受”结论已不可逆，
    因此可以提前返回。若全部模板均未达到阈值，则仍会完整遍历并返回“拒绝”。
    """

    identity_started = time.perf_counter()
    paths = [str(path) for path in identity.get("template_paths", [])]
    method = str(dict(config.get("identification", {})).get("fusion_method", "max")).lower()
    if early_stop_threshold is not None:
        if method not in {"max", "max_quality_tiebreak"}:
            raise ValueError(f"early stop requires max-like fusion, got {method}")
        early_stop_threshold = float(early_stop_threshold)
        if not 0.0 <= early_stop_threshold <= 1.0:
            raise ValueError(f"early_stop_threshold must be in [0,1], got {early_stop_threshold}")
    require = "sift" if str(descriptor_source).lower() in {"sift", "rootsift"} else "hardnet"

    results: list[dict[str, Any]] = []
    best_index = -1
    early_stopped = False
    template_load_ms = 0.0
    timing_fields = (
        "descriptor_prepare_ms",
        "candidate_generation_ms",
        "candidate_filter_ms",
        "ransac_ms",
        "inlier_refinement_ms",
        "texture_similarity_ms",
        "score_fusion_ms",
        "postprocess_ms",
        "image_match_total_ms",
    )
    timing_totals = {field: 0.0 for field in timing_fields}
    for index, path in enumerate(paths):
        load_started = time.perf_counter()
        gallery_template = load_template_cached(
            path,
            template_cache,
            require=require,
        )
        template_load_ms += (time.perf_counter() - load_started) * 1000.0
        result = match_templates_descriptor_l2(
            query_template,
            gallery_template,
            config,
            descriptor_source=descriptor_source,
        )
        results.append(result)
        for field in timing_fields:
            timing_totals[field] += float(result.get(field, 0.0))
        if best_index < 0:
            best_index = index
        else:
            best = results[best_index]
            best_key = (float(best.get("score", 0.0)), float(best.get("quality_score", 0.0)))
            result_key = (float(result.get("score", 0.0)), float(result.get("quality_score", 0.0)))
            if result_key > best_key:
                best_index = index
        if early_stop_threshold is not None and float(result.get("score", 0.0)) >= early_stop_threshold:
            early_stopped = True
            break

    if results:
        best_index = max(
            range(len(results)),
            key=lambda idx: (
                float(results[idx].get("score", 0.0)),
                float(results[idx].get("quality_score", 0.0)),
            ),
        )
    best = results[best_index] if best_index >= 0 else {}
    best_template = (
        load_template_cached(paths[best_index], template_cache, require=require)
        if best_index >= 0
        else {}
    )
    fusion_started = time.perf_counter()
    identity_score = float(fuse_image_results(results, method))
    identity_fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
    identity_match_total_ms = (
        time.perf_counter() - identity_started
    ) * 1000.0
    identity_match_overhead_ms = max(
        0.0,
        identity_match_total_ms
        - template_load_ms
        - timing_totals["image_match_total_ms"]
        - identity_fusion_ms,
    )
    return {
        "score": identity_score,
        "best_template_path": paths[best_index] if best_index >= 0 else "",
        "best_template_image_id": str(best_template.get("image_id", "")),
        "best_template_image_path": str(best_template.get("image_path", "")),
        "num_templates": len(paths),
        "num_templates_evaluated": len(results),
        "early_stopped": int(early_stopped),
        "early_stop_threshold": "" if early_stop_threshold is None else float(early_stop_threshold),
        "best_image_score": float(best.get("score", 0.0)),
        "best_quality_score": float(best.get("quality_score", 0.0)),
        "num_raw_matches": int(best.get("num_raw_matches", 0)),
        "num_candidates": int(best.get("num_candidates", 0)),
        "num_inliers": int(best.get("num_inliers", 0)),
        "raw_inliers": int(best.get("raw_inliers", 0)),
        "unique_inliers": int(best.get("unique_inliers", 0)),
        "unique_query_inliers": int(best.get("unique_query_inliers", 0)),
        "unique_gallery_inliers": int(best.get("unique_gallery_inliers", 0)),
        "inlier_ratio": float(best.get("inlier_ratio", 0.0)),
        "mean_l2_distance": float(best.get("mean_l2_distance", 0.0)),
        "mean_reproj_error": float(best.get("mean_reproj_error", 0.0)),
        "orientation_consistency": float(best.get("orientation_consistency", 0.0)),
        "dominant_angle_delta": float(best.get("dominant_angle_delta", 0.0)),
        "mean_similarity": float(best.get("mean_similarity", 0.0)),
        "texture_enabled": int(bool(best.get("texture_enabled", False))),
        "texture_evaluated": int(bool(best.get("texture_evaluated", False))),
        "texture_available": int(bool(best.get("texture_available", False))),
        "texture_similarity": float(best.get("texture_similarity", 0.0)),
        "texture_overlap_fraction": float(best.get("texture_overlap_fraction", 0.0)),
        "texture_valid_blocks": int(best.get("texture_valid_blocks", 0)),
        "geometry_similarity": float(best.get("geometry_similarity", 0.0)),
        "geometry_weight": float(best.get("geometry_weight", 1.0)),
        "texture_weight": float(best.get("texture_weight", 0.0)),
        "texture_decision": str(best.get("texture_decision", "")),
        "registered_template_load_ms": template_load_ms,
        **timing_totals,
        "identity_fusion_ms": identity_fusion_ms,
        "identity_match_overhead_ms": identity_match_overhead_ms,
        "identity_match_total_ms": identity_match_total_ms,
    }
