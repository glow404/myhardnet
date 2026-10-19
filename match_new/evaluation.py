"""HardNet 身份验证评估与失败样本导出。

作用：
    这个脚本负责把“图像级匹配器”扩展成“手机解锁式 identity 级验证实验”。
    HardNet 模板可使用 float/L2 或 binary/Hamming 描述子，后续评估流程保持一致。

    1. 每个 query 对本人 identity 的注册模板打分，得到 genuine attempt；
    2. 同一个 query 对非本人 identity 的注册模板打分，得到 impostor attempts；
    3. identity 级分数默认取注册模板中的最大图像级 score；
    4. 根据 identity 级 score 计算 FAR、FRR、EER 和 AUC；
    5. 在配置的解锁阈值下导出 false reject / false accept 原图和拼接预览图。

注意：
    这里所有 FAR/FRR 都是 identity 级 attempt 统计，不是单图对单图统计。
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from match_new.identity_matcher import (
    compute_end_to_end_core_ms,
    load_template_cached,
    resolve_early_stop_threshold,
    score_query_against_identity,
)
from match_new.template_builder import load_identity_templates
from match_new.template_library import TemplateLibraryManager
from match_new.utils import (
    ensure_dir,
    format_significant_digits,
    read_csv_rows,
    resolve_path,
    safe_id,
    template_filename,
    write_csv_rows,
    write_json,
    write_yaml,
)


SCORE_FIELDNAMES = [
    # verification_scores.csv 的字段。
    # 每一行代表一次 query 对某个 owner_identity 的验证尝试。
    "query_id",
    "query_identity",
    "owner_identity",
    "label",
    "score",
    "accepted_at_selected_threshold",
    "query_image_path",
    "query_template_path",
    "descriptor_source",
    "descriptor_kind",
    "descriptor_metric",
    "descriptor_dim",
    "best_template_path",
    "best_template_image_id",
    "best_template_image_path",
    "num_templates",
    "num_templates_evaluated",
    "early_stopped",
    "early_stop_threshold",
    "best_image_score",
    "best_quality_score",
    "num_raw_matches",
    "num_abs_distance_pass",
    "num_one_way_ratio_pass",
    "num_ratio_pass",
    "num_queries_with_candidates",
    "candidate_query_coverage",
    "templates_with_candidates",
    "templates_with_abs_distance_pass",
    "templates_with_ratio_pass",
    "templates_with_ransac_success",
    "templates_with_unique_inlier_ready",
    "templates_with_texture_evaluated",
    "templates_with_texture_available",
    "templates_with_fused_score",
    "max_candidate_query_coverage",
    "max_num_candidates",
    "max_raw_inliers",
    "max_unique_inliers",
    "max_geometry_similarity",
    "max_texture_similarity",
    "ransac_success",
    "scale_rejected",
    "num_candidates",
    "num_inliers",
    "raw_inliers",
    "unique_inliers",
    "unique_query_inliers",
    "unique_gallery_inliers",
    "inlier_ratio",
    "mean_descriptor_distance",
    "mean_l2_distance",
    "mean_hamming_distance",
    "mean_reproj_error",
    "orientation_consistency",
    "dominant_angle_delta",
    "mean_similarity",
    "texture_enabled",
    "texture_evaluated",
    "texture_available",
    "texture_similarity",
    "texture_overlap_fraction",
    "texture_valid_blocks",
    "geometry_similarity",
    "geometry_weight",
    "texture_weight",
    "texture_decision",
    "registered_template_load_ms",
    "descriptor_prepare_ms",
    "candidate_generation_ms",
    "candidate_filter_ms",
    "ransac_ms",
    "inlier_refinement_ms",
    "unique_inlier_dedup_ms",
    "texture_similarity_ms",
    "score_fusion_ms",
    "postprocess_ms",
    "image_match_total_ms",
    "identity_fusion_ms",
    "identity_match_overhead_ms",
    "identity_match_total_ms",
    "attempt_match_ms",
    "query_template_build_ms",
    "unlock_match_ms",
    "unlock_end_to_end_ms",
    "unlock_end_to_end_core_ms",
]


def parse_float(value: Any) -> float | None:
    """解析 CSV 文本形式的耗时值；空值或非法值返回 ``None``。"""

    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_false_reject(
    row: dict[str, Any],
    threshold: float,
    config: dict[str, Any],
) -> str:
    """按最早失败阶段分类 genuine identity attempt，不改变任何匹配判定。"""

    def integer(name: str) -> int:
        value = parse_float(row.get(name))
        return int(value) if value is not None else 0

    def number(name: str) -> float:
        value = parse_float(row.get(name))
        return float(value) if value is not None else 0.0

    matching_cfg = dict(config.get("matching", {}))
    texture_cfg = dict(config.get("texture_verification", {}))
    candidate_policy = str(
        matching_cfg.get("candidate_policy", "topk_or_ratio")
    ).lower()
    if integer("templates_with_candidates") == 0:
        if integer("templates_with_abs_distance_pass") == 0:
            return "descriptor_distance_reject"
        if (
            candidate_policy in {"ratio_only", "topk_or_ratio"}
            and integer("templates_with_ratio_pass") == 0
        ):
            return "ratio_reject"
        return "candidate_insufficient"
    if integer("templates_with_ransac_success") == 0:
        if integer("scale_rejected") > 0:
            return "ransac_scale_reject"
        return "ransac_failed"

    low_unique = int(texture_cfg.get("low_unique_inliers", 3))
    if integer("templates_with_unique_inlier_ready") == 0:
        return "unique_inlier_insufficient"
    if bool(texture_cfg.get("enabled", False)):
        if (
            integer("templates_with_texture_evaluated") > 0
            and integer("templates_with_texture_available") == 0
        ):
            return "texture_unavailable"
        if (
            integer("templates_with_fused_score") > 0
            and number("max_geometry_similarity") >= threshold
            and number("score") < threshold
        ):
            return "texture_fusion_reject"
    if float(threshold) - number("score") <= 0.05:
        return "threshold_edge"
    return "score_below_threshold"


def timing_percentile(values: list[float], q: float) -> float | None:
    """用最近秩方式计算耗时报告中的百分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(q)))
    index = max(0, min(len(ordered) - 1, index))
    return float(ordered[index])


def summarize_timing(values: list[float]) -> dict[str, Any]:
    """汇总一组毫秒耗时，输出最快、平均、最慢及 P50/P95。"""

    if not values:
        return {"count": 0}
    total = float(sum(values))
    return {
        "count": len(values),
        "fastest_ms": float(min(values)),
        "average_ms": total / len(values),
        "p50_ms": timing_percentile(values, 0.50),
        "p95_ms": timing_percentile(values, 0.95),
        "slowest_ms": float(max(values)),
    }


def summarize_score_components(score_rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    """汇总 C 方案各分量，并对比同一记录上的纯几何阈值判定。"""

    decision_counts: dict[str, int] = {}
    groups: dict[str, dict[str, float]] = {
        "genuine": {"count": 0.0, "evaluated": 0.0, "available": 0.0, "fused": 0.0},
        "impostor": {"count": 0.0, "evaluated": 0.0, "available": 0.0, "fused": 0.0},
    }
    for row in score_rows:
        decision = str(row.get("texture_decision", ""))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        name = "genuine" if int(row.get("label", 0)) == 1 else "impostor"
        group = groups[name]
        score = float(row.get("score", 0.0) or 0.0)
        geometry = float(row.get("geometry_similarity", 0.0) or 0.0)
        texture = float(row.get("texture_similarity", 0.0) or 0.0)
        group["count"] += 1.0
        group["evaluated"] += float(int(row.get("texture_evaluated", 0) or 0))
        group["available"] += float(int(row.get("texture_available", 0) or 0))
        group["fused"] += float(decision == "fused_score")
        group["score_sum"] = group.get("score_sum", 0.0) + score
        group["geometry_sum"] = group.get("geometry_sum", 0.0) + geometry
        group["texture_sum"] = group.get("texture_sum", 0.0) + texture
        group["match_score_accept"] = group.get("match_score_accept", 0.0) + float(score >= threshold)
        group["geometry_only_accept"] = group.get("geometry_only_accept", 0.0) + float(geometry >= threshold)
        group["decision_changed"] = group.get("decision_changed", 0.0) + float(
            (score >= threshold) != (geometry >= threshold)
        )

    summary_groups: dict[str, Any] = {}
    for name, group in groups.items():
        count = int(group["count"])
        denominator = max(count, 1)
        summary_groups[name] = {
            "count": count,
            "texture_evaluated": int(group["evaluated"]),
            "texture_available": int(group["available"]),
            "fused_score_count": int(group["fused"]),
            "mean_match_score": group.get("score_sum", 0.0) / denominator,
            "mean_geometry_similarity": group.get("geometry_sum", 0.0) / denominator,
            "mean_texture_similarity": group.get("texture_sum", 0.0) / denominator,
            "match_score_accept": int(group.get("match_score_accept", 0.0)),
            "geometry_only_accept_on_recorded_best": int(group.get("geometry_only_accept", 0.0)),
            "decision_changed_vs_geometry_only": int(group.get("decision_changed", 0.0)),
        }
    return {
        "threshold": float(threshold),
        "decision_counts": decision_counts,
        **summary_groups,
    }


def summarize_matching_stages(
    score_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """汇总 identity attempt 在候选、RANSAC 和 unique-inlier 阶段的通过率。"""

    low_unique = int(
        dict(config.get("texture_verification", {})).get(
            "low_unique_inliers",
            3,
        )
    )
    groups: dict[str, list[dict[str, str]]] = {"genuine": [], "impostor": []}
    for row in score_rows:
        name = "genuine" if int(row.get("label", 0)) == 1 else "impostor"
        groups[name].append(row)

    def value(row: dict[str, str], name: str) -> float:
        parsed = parse_float(row.get(name))
        return float(parsed) if parsed is not None else 0.0

    summary: dict[str, Any] = {
        "definition": (
            "attempt 级覆盖率；candidate_ready 表示至少一张注册模板产生候选，"
            "不是带点对应标注的 descriptor recall@K。"
        ),
        "low_unique_inliers": low_unique,
    }
    for name, rows in groups.items():
        count = len(rows)
        denominator = max(count, 1)
        summary[name] = {
            "count": count,
            "candidate_ready": sum(
                value(row, "templates_with_candidates") > 0 for row in rows
            ),
            "candidate_ready_rate": sum(
                value(row, "templates_with_candidates") > 0 for row in rows
            )
            / denominator,
            "ratio_ready_rate": sum(
                value(row, "templates_with_ratio_pass") > 0 for row in rows
            )
            / denominator,
            "ransac_success_rate": sum(
                value(row, "templates_with_ransac_success") > 0 for row in rows
            )
            / denominator,
            "unique_inlier_ready_rate": sum(
                value(row, "templates_with_unique_inlier_ready") > 0 for row in rows
            )
            / denominator,
            "texture_available_rate": sum(
                value(row, "templates_with_texture_available") > 0 for row in rows
            )
            / denominator,
            "fused_score_ready_rate": sum(
                value(row, "templates_with_fused_score") > 0 for row in rows
            )
            / denominator,
            "mean_best_candidates": sum(
                value(row, "num_candidates") for row in rows
            )
            / denominator,
            "mean_best_raw_inliers": sum(
                value(row, "raw_inliers") for row in rows
            )
            / denominator,
            "mean_best_unique_inliers": sum(
                value(row, "unique_inliers") for row in rows
            )
            / denominator,
            "mean_max_unique_inliers": sum(
                value(row, "max_unique_inliers") for row in rows
            )
            / denominator,
            "mean_max_candidate_query_coverage": sum(
                value(row, "max_candidate_query_coverage") for row in rows
            )
            / denominator,
            "mean_best_reprojection_error": sum(
                value(row, "mean_reproj_error") for row in rows
            )
            / denominator,
        }
    return summary


def select_query_rows(metadata_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """从 split metadata 中取出 query 图像行。"""

    return [row for row in metadata_rows if str(row.get("split", "")).lower() == "query"]


def rate_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """在指定阈值下计算 FAR/FRR/TAR。

    判定规则：
        score >= threshold 视为接受；
        genuine 被拒绝计入 FRR；
        impostor 被接受计入 FAR。
    """

    genuine = labels == 1
    impostor = labels == 0
    genuine_total = int(np.sum(genuine))
    impostor_total = int(np.sum(impostor))
    genuine_accept = int(np.sum(scores[genuine] >= threshold)) if genuine_total else 0
    impostor_accept = int(np.sum(scores[impostor] >= threshold)) if impostor_total else 0
    return {
        "threshold": float(threshold),
        "unlock_threshold": float(threshold),
        "match_score_threshold": float(threshold),
        "far": float(impostor_accept / impostor_total) if impostor_total else None,
        "frr": float((genuine_total - genuine_accept) / genuine_total) if genuine_total else None,
        "tar": float(genuine_accept / genuine_total) if genuine_total else None,
        "genuine_accept": genuine_accept,
        "genuine_reject": int(genuine_total - genuine_accept),
        "genuine_total": genuine_total,
        "impostor_accept": impostor_accept,
        "impostor_reject": int(impostor_total - impostor_accept),
        "impostor_total": impostor_total,
    }


def build_threshold_curve(labels: np.ndarray, scores: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    """在 [0,1] 连续匹配分数范围内构建阈值曲线。"""

    eval_cfg = dict(config.get("evaluation", {}))
    configured = eval_cfg.get("score_thresholds")
    if configured:
        thresholds = [float(value) for value in configured]
    else:
        thresholds = [0.0, 0.5, 1.0]
    if bool(eval_cfg.get("auto_thresholds", True)):
        step = float(eval_cfg.get("score_threshold_step", 0.01))
        if not 0.0 < step <= 1.0:
            raise ValueError(f"evaluation.score_threshold_step must be in (0,1], got {step}")
        count = int(math.ceil(1.0 / step))
        auto = [min(index * step, 1.0) for index in range(count + 1)]
        thresholds.extend(auto)
    # 网格只用于画曲线，实际分数也必须纳入候选阈值；否则低 FAR 区域可能
    # 被 0.01 的粗粒度跳过，导致选出的阈值无法精确复现。
    thresholds.extend(
        float(score)
        for score in np.asarray(scores).reshape(-1)
        if np.isfinite(score)
    )
    thresholds = sorted({round(value, 10) for value in thresholds if 0.0 <= value <= 1.0})
    return [rate_at_threshold(labels, scores, threshold) for threshold in thresholds]


def resolve_far_frr_table_config(config: dict[str, Any]) -> dict[str, float]:
    """解析固定 FAR/FRR 表格的阈值扫描范围。"""

    eval_cfg = dict(config.get("evaluation", {}))
    table_cfg = dict(eval_cfg.get("far_frr_table", {}))
    min_threshold = float(table_cfg.get("min_threshold", 0.50))
    max_threshold = float(table_cfg.get("max_threshold", 0.85))
    step = float(table_cfg.get("step", 0.01))
    if not 0.0 <= min_threshold <= max_threshold <= 1.0:
        raise ValueError(
            "evaluation.far_frr_table bounds must satisfy "
            "0 <= min_threshold <= max_threshold <= 1, "
            f"got min={min_threshold}, max={max_threshold}."
        )
    if not 0.0 < step <= 1.0:
        raise ValueError(
            f"evaluation.far_frr_table.step must be in (0,1], got {step}."
        )
    return {
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "step": step,
    }


def far_frr_table_filename(min_threshold: float, max_threshold: float) -> str:
    """根据阈值范围生成固定 FAR/FRR 表格文件名。"""

    return (
        f"far_frr_thresholds_{min_threshold:.2f}_{max_threshold:.2f}.csv"
    )


def build_far_frr_table(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_threshold: float = 0.50,
    max_threshold: float = 0.85,
    step: float = 0.01,
) -> list[dict[str, Any]]:
    """构建指定阈值范围内的固定 FAR/FRR 表格。

    默认扫描 ``0.50--0.85``（步长 ``0.01``）。返回值保留接受/拒绝计数，
    便于检查 FAR 和 FRR 的分母是否符合预期。
    """

    if not 0.0 <= min_threshold <= max_threshold <= 1.0:
        raise ValueError(
            "FAR/FRR threshold bounds must satisfy "
            "0 <= min_threshold <= max_threshold <= 1, "
            f"got min={min_threshold}, max={max_threshold}."
        )
    if not 0.0 < step <= 1.0:
        raise ValueError(f"FAR/FRR threshold step must be in (0,1], got {step}.")

    thresholds: list[float] = []
    value = float(min_threshold)
    max_iterations = int(math.ceil((max_threshold - min_threshold) / step)) + 2
    for _ in range(max(1, max_iterations)):
        if value > max_threshold + 1e-12:
            break
        thresholds.append(round(value, 10))
        value = round(value + step, 10)
    if not thresholds or thresholds[-1] < max_threshold - 1e-12:
        thresholds.append(round(max_threshold, 10))
    unique_thresholds = sorted({threshold for threshold in thresholds if 0.0 <= threshold <= 1.0})
    return [rate_at_threshold(labels, scores, threshold) for threshold in unique_thresholds]


def select_zero_far_min_frr_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """选择使全体 impostor FAR=0 且 FRR 最小的精确阈值。

    ``score >= threshold`` 视为接受，因此阈值必须严格大于最大 impostor
    分数。FAR 在阈值升高时单调不增，FRR 单调不减，所以最大 impostor
    分数之上的最小 float64 值就是 FAR=0 约束下 FRR 最小的工作点。
    """

    label_values = np.asarray(labels, dtype=np.int32).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if label_values.size != score_values.size:
        raise ValueError(
            "labels and scores must have the same length: "
            f"labels={label_values.size}, scores={score_values.size}."
        )
    if score_values.size and not np.all(np.isfinite(score_values)):
        raise ValueError("Cannot select a FAR=0 threshold from non-finite scores.")

    impostor_scores = score_values[label_values == 0]
    if impostor_scores.size == 0:
        point = rate_at_threshold(label_values, score_values, 0.0)
        return {
            **point,
            "selection_status": "unavailable_no_impostor_attempts",
            "selection_rule": "global_far_0_then_min_frr",
            "max_impostor_score": None,
        }

    max_impostor_score = float(np.max(impostor_scores))
    threshold = float(np.nextafter(max_impostor_score, np.inf))
    point = rate_at_threshold(label_values, score_values, threshold)
    if point["far"] != 0.0:
        raise RuntimeError(
            "Internal error: selected threshold did not produce FAR=0: "
            f"threshold={threshold}, FAR={point['far']}."
        )
    return {
        **point,
        "selection_status": "selected",
        "selection_rule": "global_far_0_then_min_frr",
        "max_impostor_score": max_impostor_score,
    }


def build_per_finger_far_frr_table(
    score_rows: list[dict[str, Any]],
    operating_point: dict[str, Any],
) -> list[dict[str, Any]]:
    """在同一个全局 FAR=0 工作点下汇总总体及每个注册手指的 FAR/FRR。

    第一行固定为全体统计。每个手指采用 owner-centric 口径：
    ``owner_identity`` 等于该手指的所有尝试构成分母，其中本人 query 是
    genuine，其他手指 query 是 impostor。
    """

    threshold = float(operating_point["threshold"])

    def table_row(
        scope: str,
        identity_id: str,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> dict[str, Any]:
        rates = rate_at_threshold(labels, scores, threshold)
        return {
            "scope": scope,
            "identity_id": identity_id,
            "threshold": rates["threshold"],
            "far": rates["far"],
            "frr": rates["frr"],
            "tar": rates["tar"],
            "genuine_accept": rates["genuine_accept"],
            "genuine_reject": rates["genuine_reject"],
            "genuine_total": rates["genuine_total"],
            "impostor_accept": rates["impostor_accept"],
            "impostor_reject": rates["impostor_reject"],
            "impostor_total": rates["impostor_total"],
            "selection_status": operating_point.get("selection_status", ""),
            "selection_rule": operating_point.get("selection_rule", ""),
            "max_impostor_score_all_fingers": operating_point.get(
                "max_impostor_score"
            ),
        }

    all_labels = np.asarray(
        [int(row["label"]) for row in score_rows], dtype=np.int32
    )
    all_scores = np.asarray(
        [float(row["score"]) for row in score_rows], dtype=np.float64
    )
    table = [table_row("all_fingers", "__ALL__", all_labels, all_scores)]
    rows_by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        rows_by_owner.setdefault(
            str(row.get("owner_identity", "")), []
        ).append(row)
    for identity_id in sorted(rows_by_owner):
        rows = rows_by_owner[identity_id]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        table.append(table_row("finger", identity_id, labels, scores))
    return table


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    selected_threshold: float,
) -> dict[str, Any]:
    """汇总配置阈值下的 FAR/FRR，以及 EER 和 AUC。"""

    metrics: dict[str, Any] = {
        "selected_threshold": float(selected_threshold),
        "fixed_threshold": rate_at_threshold(labels, scores, selected_threshold),
        "auc": None,
        "eer": None,
        "eer_threshold": None,
    }
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    metrics["auc"] = float(auc(fpr, tpr))
    metrics["eer"] = float((fpr[idx] + fnr[idx]) / 2.0)
    metrics["eer_threshold"] = float(thresholds[idx])
    return metrics


def matching_backend_name(descriptor_source: str, config: dict[str, Any]) -> str:
    """生成准确的匹配后端名称，避免报告里出现与实际策略不一致的字样。"""

    source = str(descriptor_source).lower()
    matching = dict(config.get("matching", {}))
    distance = str(matching.get("distance", "l2")).lower()
    policy = str(matching.get("candidate_policy", "topk_or_ratio")).lower()
    texture_suffix = "_texture_fusion" if bool(dict(config.get("texture_verification", {})).get("enabled", False)) else ""
    return f"{source}_{distance}_{policy}_ransac{texture_suffix}"


def _resolve_config_path(config: dict[str, Any], value: Any) -> str | None:
    """把配置里的相对路径解析成绝对路径字符串；空值返回 None。"""

    if value in {None, ""}:
        return None
    try:
        return str(resolve_path(config, value))
    except Exception:
        return str(value)


def build_effective_config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """整理本次评估实际生效的配置，写入 metrics.json / effective_config.yaml。

    重点保留：
        - model.checkpoint
        - matching 全部参数（含双向 Lowe 验证）
        - sift / patch / texture_verification / enrollment / template_management / identification / evaluation
    """

    snapshot: dict[str, Any] = {}
    for key in (
        "output",
        "runtime",
        "data",
        "model",
        "sift",
        "patch",
        "keypoint_filter",
        "matching",
        "texture_verification",
        "enrollment",
        "template_management",
        "identification",
        "evaluation",
        "descriptor_sources",
    ):
        if key in config:
            snapshot[key] = copy.deepcopy(config[key])

    output_cfg = dict(snapshot.get("output", {}))
    output_cfg.pop("_output_dir_from_cli", None)
    if "output_dir" in output_cfg:
        output_cfg["output_dir"] = _resolve_config_path(config, output_cfg.get("output_dir")) or output_cfg.get("output_dir")
    if output_cfg:
        snapshot["output"] = output_cfg

    data_cfg = dict(snapshot.get("data", {}))
    for field in ("image_root",):
        if field in data_cfg:
            data_cfg[field] = _resolve_config_path(config, data_cfg.get(field)) or data_cfg.get(field)
    if data_cfg:
        snapshot["data"] = data_cfg

    model_cfg = dict(snapshot.get("model", {}))
    checkpoint = _resolve_config_path(config, model_cfg.get("checkpoint"))
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        model_cfg["checkpoint"] = checkpoint
        model_cfg["checkpoint_exists"] = checkpoint_path.exists()
        if checkpoint_path.exists():
            model_cfg["checkpoint_size_bytes"] = int(checkpoint_path.stat().st_size)
    if model_cfg:
        snapshot["model"] = model_cfg

    if config.get("_config_path"):
        snapshot["source_config_path"] = str(config.get("_config_path"))
    if config.get("_config_sources"):
        snapshot["source_config_chain"] = list(config.get("_config_sources", []))
    return snapshot


def imread_color(path: str | Path) -> np.ndarray | None:
    """读取彩色图像，兼容 Windows 中文路径。"""

    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def imwrite_image(path: str | Path, image: np.ndarray) -> bool:
    """写图像文件，兼容 Windows 中文路径。"""

    target = Path(path)
    ensure_dir(target.parent)
    ext = target.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def copy_file_if_requested(src: str | Path, dst: str | Path, enabled: bool) -> str:
    """根据配置决定是否复制原图到失败样本目录。"""

    if not enabled:
        return ""
    if not str(src):
        return ""
    source = Path(src)
    if not source.exists() or not source.is_file():
        return ""
    target = Path(dst)
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return str(target)


def make_pair_preview(
    query_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
    title: str,
    score_text: str,
    max_height: int = 420,
) -> str:
    """生成 query 与最佳命中模板的左右拼接预览图。

    preview 只用于人工快速浏览失败案例，不参与指标计算。
    """

    query = imread_color(query_path)
    template = imread_color(template_path)
    if query is None or template is None:
        return ""

    def resize_keep_height(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image
        scale = min(float(max_height) / float(h), 1.0)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    left = resize_keep_height(query)
    right = resize_keep_height(template)
    height = max(left.shape[0], right.shape[0])
    gap = 16
    header = 64
    width = left.shape[1] + right.shape[1] + gap
    canvas = np.full((height + header, width, 3), 255, dtype=np.uint8)
    canvas[header : header + left.shape[0], : left.shape[1]] = left
    canvas[header : header + right.shape[0], left.shape[1] + gap : left.shape[1] + gap + right.shape[1]] = right
    cv2.putText(canvas, title[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, score_text[:120], (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "query", (8, header + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "best template", (left.shape[1] + gap + 8, header + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 160), 1, cv2.LINE_AA)
    return str(output_path) if imwrite_image(output_path, canvas) else ""


def export_failure_cases(
    score_rows: list[dict[str, str]],
    threshold_info: dict[str, Any] | None,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """在配置的解锁阈值下导出失败样本。

    false_reject：
        label=1 且 score < threshold，本人被拒。

    false_accept：
        label=0 且 score >= threshold，非本人被接受。

    每个导出的 case 目录包含：
        - query 原图；
        - best_template 原图；
        - pair_preview.jpg；
        - case.json；
        - 汇总 CSV 中的一行。
    """

    if threshold_info is None:
        return {"enabled": False, "reason": "no_target_threshold"}
    failure_cfg = dict(dict(config.get("evaluation", {})).get("failure_export", {}))
    if not bool(failure_cfg.get("enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    threshold = float(threshold_info["threshold"])
    max_cases = int(failure_cfg.get("max_cases_per_type", 200))
    copy_originals = bool(failure_cfg.get("copy_original_images", True))
    make_preview = bool(failure_cfg.get("make_pair_preview", True))
    threshold_text = format_significant_digits(threshold)
    root = ensure_dir(
        Path(output_dir)
        / "failure_cases"
        / f"threshold_{safe_id(threshold_text)}"
    )

    # 先扫描 verification_scores，找出配置阈值下的失败行。
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(score_rows):
        label = int(row["label"])
        score = float(row["score"])
        failure_type = ""
        if label == 1 and score < threshold:
            failure_type = "false_reject"
        elif label == 0 and score >= threshold:
            failure_type = "false_accept"
        if not failure_type:
            continue
        failure_stage = (
            classify_false_reject(row, threshold, config)
            if failure_type == "false_reject"
            else "impostor_score_above_threshold"
        )
        failures.append(
            {
                "case_index": index,
                "failure_type": failure_type,
                "failure_stage": failure_stage,
                "threshold": threshold,
                "score": score,
                "query_id": row.get("query_id", ""),
                "query_identity": row.get("query_identity", ""),
                "owner_identity": row.get("owner_identity", ""),
                "query_image_path": row.get("query_image_path", ""),
                "query_template_path": row.get("query_template_path", ""),
                "best_template_path": row.get("best_template_path", ""),
                "best_template_image_id": row.get("best_template_image_id", ""),
                "best_template_image_path": row.get("best_template_image_path", ""),
                "best_image_score": row.get("best_image_score", ""),
                "best_quality_score": row.get("best_quality_score", ""),
                "num_raw_matches": row.get("num_raw_matches", ""),
                "num_abs_distance_pass": row.get("num_abs_distance_pass", ""),
                "num_one_way_ratio_pass": row.get("num_one_way_ratio_pass", ""),
                "num_ratio_pass": row.get("num_ratio_pass", ""),
                "num_queries_with_candidates": row.get("num_queries_with_candidates", ""),
                "candidate_query_coverage": row.get("candidate_query_coverage", ""),
                "templates_with_candidates": row.get("templates_with_candidates", ""),
                "templates_with_abs_distance_pass": row.get("templates_with_abs_distance_pass", ""),
                "templates_with_ratio_pass": row.get("templates_with_ratio_pass", ""),
                "templates_with_ransac_success": row.get("templates_with_ransac_success", ""),
                "ransac_success": row.get("ransac_success", ""),
                "scale_rejected": row.get("scale_rejected", ""),
                "num_candidates": row.get("num_candidates", ""),
                "raw_inliers": row.get("raw_inliers", ""),
                "unique_inliers": row.get("unique_inliers", ""),
                "mean_descriptor_distance": row.get(
                    "mean_descriptor_distance", ""
                ),
                "mean_l2_distance": row.get("mean_l2_distance", ""),
                "mean_hamming_distance": row.get(
                    "mean_hamming_distance", ""
                ),
                "mean_reproj_error": row.get("mean_reproj_error", ""),
                "orientation_consistency": row.get("orientation_consistency", ""),
                "geometry_similarity": row.get("geometry_similarity", ""),
                "texture_available": row.get("texture_available", ""),
                "texture_similarity": row.get("texture_similarity", ""),
                "texture_decision": row.get("texture_decision", ""),
            }
        )

    write_csv_rows(root / "failure_cases_all.csv", failures)
    # false_reject 按分数从高到低排，优先看“差一点通过”的本人失败；
    # false_accept 也按分数从高到低排，优先看最危险的误接受。
    ordered = {
        "false_reject": sorted([item for item in failures if item["failure_type"] == "false_reject"], key=lambda item: (-float(item["score"]), item["query_id"])),
        "false_accept": sorted([item for item in failures if item["failure_type"] == "false_accept"], key=lambda item: (-float(item["score"]), item["query_id"])),
    }

    # 将前 max_cases_per_type 个 case 落盘，避免全量实验时复制太多图片。
    exported: list[dict[str, Any]] = []
    for failure_type, items in ordered.items():
        type_dir = ensure_dir(root / f"{failure_type}s")
        for local_idx, item in enumerate(items[: max_cases if max_cases > 0 else None], start=1):
            score_text = format_significant_digits(float(item["score"]))
            case_id = (
                f"{failure_type}_{local_idx:05d}"
                f"__score_{safe_id(score_text)}"
                f"__q_{safe_id(item['query_id'])}"
                f"__owner_{safe_id(item['owner_identity'])}"
            )
            case_dir = ensure_dir(type_dir / case_id)
            query_src = item["query_image_path"]
            template_src = item["best_template_image_path"]
            query_ext = Path(query_src).suffix or ".png"
            template_ext = Path(template_src).suffix or ".png"
            query_copy = copy_file_if_requested(query_src, case_dir / f"query{query_ext}", copy_originals)
            template_copy = copy_file_if_requested(template_src, case_dir / f"best_template{template_ext}", copy_originals)
            preview = ""
            if make_preview and query_src and template_src:
                preview = make_pair_preview(
                    query_src,
                    template_src,
                    case_dir / "pair_preview.jpg",
                    title=f"{failure_type} | query={item['query_identity']} owner={item['owner_identity']}",
                    score_text=(
                        f"score={score_text} threshold={threshold_text} "
                        f"unique={item['unique_inliers']} raw={item['raw_inliers']}"
                    ),
                )
            enriched = {
                **item,
                "case_dir": str(case_dir),
                "query_copy": query_copy,
                "best_template_copy": template_copy,
                "pair_preview": preview,
            }
            write_json(case_dir / "case.json", enriched)
            exported.append(enriched)

    write_csv_rows(root / "failure_cases_exported.csv", exported)
    failure_stage_counts: dict[str, int] = {}
    for item in failures:
        stage = str(item.get("failure_stage", "unknown"))
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1
    summary = {
        "enabled": True,
        "threshold": threshold,
        "target_satisfied": bool(threshold_info.get("satisfied", False)),
        "failure_stage_counts": failure_stage_counts,
        "num_false_rejects": len(ordered["false_reject"]),
        "num_false_accepts": len(ordered["false_accept"]),
        "num_exported_false_rejects": min(len(ordered["false_reject"]), max_cases) if max_cases > 0 else len(ordered["false_reject"]),
        "num_exported_false_accepts": min(len(ordered["false_accept"]), max_cases) if max_cases > 0 else len(ordered["false_accept"]),
        "failure_dir": str(root),
        "all_cases_csv": str(root / "failure_cases_all.csv"),
        "exported_cases_csv": str(root / "failure_cases_exported.csv"),
    }
    write_json(root / "failure_summary.json", summary)
    return summary


def write_plots(
    labels: np.ndarray,
    scores: np.ndarray,
    curve: list[dict[str, Any]],
    selected_threshold: float,
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """写出分数分布图，以及标注 EER 和配置阈值的 FAR/FRR 曲线图。"""

    plots = ensure_dir(output_dir / "plots")
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    bins = np.linspace(0.0, 1.0, 51)
    plt.figure(figsize=(8, 5))
    if impostor.size:
        plt.hist(impostor, bins=bins, alpha=0.65, label="impostor", density=True)
    if genuine.size:
        plt.hist(genuine, bins=bins, alpha=0.65, label="genuine", density=True)
    plt.legend()
    plt.xlabel("match score")
    plt.tight_layout()
    plt.savefig(plots / "score_distribution.png", dpi=150)
    plt.close()

    thresholds = [float(item["threshold"]) for item in curve]
    far = [float(item["far"] or 0.0) for item in curve]
    frr = [float(item["frr"] or 0.0) for item in curve]
    far_color = "tab:blue"
    frr_color = "tab:orange"
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far, color=far_color, label="FAR")
    plt.plot(thresholds, frr, color=frr_color, label="FRR")

    fixed = dict(metrics.get("fixed_threshold") or {})
    selected_far = float(fixed.get("far") or 0.0)
    selected_frr = float(fixed.get("frr") or 0.0)
    plt.axvline(
        selected_threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="configured threshold",
    )
    plt.scatter(
        [selected_threshold, selected_threshold],
        [selected_far, selected_frr],
        color=[far_color, frr_color],
        s=36,
        zorder=5,
    )
    plt.annotate(
        (
            f"Configured threshold={selected_threshold:.4f}\n"
            f"FAR={selected_far:.4%}\n"
            f"FRR={selected_frr:.4%}"
        ),
        xy=(selected_threshold, max(selected_far, selected_frr)),
        xytext=(-8, 28),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": "gray", "linewidth": 0.8},
    )

    eer = metrics.get("eer")
    eer_threshold = metrics.get("eer_threshold")
    if eer is not None and eer_threshold is not None:
        eer_value = float(eer)
        eer_x = float(eer_threshold)
        plt.axvline(
            eer_x,
            color="tab:green",
            linestyle=":",
            linewidth=1,
            label="EER threshold",
        )
        plt.scatter(
            [eer_x],
            [eer_value],
            color="tab:green",
            marker="o",
            s=42,
            zorder=5,
        )
        plt.annotate(
            f"EER threshold={eer_x:.4f}\nEER={eer_value:.4%}",
            xy=(eer_x, eer_value),
            xytext=(8, -38),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            arrowprops={
                "arrowstyle": "->",
                "color": "tab:green",
                "linewidth": 0.8,
            },
        )

    plt.legend()
    plt.xlabel("match score threshold")
    plt.ylabel("error rate")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plots / "far_frr_vs_threshold.png", dpi=150)
    plt.close()


def write_unlock_timing_report(score_rows: list[dict[str, str]], output_dir: str | Path) -> dict[str, Any]:
    """从本人评估记录中生成总体耗时汇总。

    离线评估不生成逐次解锁 CSV；推理阶段耗时已记录在模板构建计时中。
    """

    total_genuine_attempts = 0
    failed_genuine_attempts = 0
    successful_unlocks_timed = 0
    match_values: list[float] = []
    end_to_end_values: list[float] = []
    end_to_end_core_values: list[float] = []
    for row in score_rows:
        if int(row.get("label", 0)) != 1:
            continue
        total_genuine_attempts += 1
        if int(row.get("accepted_at_selected_threshold", 0)) != 1:
            failed_genuine_attempts += 1
            continue
        successful_unlocks_timed += 1
        match_ms = parse_float(row.get("unlock_match_ms"))
        end_to_end_ms = parse_float(row.get("unlock_end_to_end_ms"))
        end_to_end_core_ms = parse_float(row.get("unlock_end_to_end_core_ms"))
        if match_ms is not None:
            match_values.append(match_ms)
        if end_to_end_core_ms is not None:
            end_to_end_core_values.append(end_to_end_core_ms)
        elif end_to_end_ms is not None:
            end_to_end_core_values.append(
                compute_end_to_end_core_ms(end_to_end_ms, row)
            )
        if end_to_end_ms is not None:
            end_to_end_values.append(end_to_end_ms)
    out = ensure_dir(output_dir)
    (Path(out) / "unlock_timing.csv").unlink(missing_ok=True)
    summary = {
        "definition": {
            "timing_scope": "只统计在 selected threshold 下成功接受的本人解锁记录。",
            "unlock_match_ms": "查询模板已存在时，与一个手指的注册模板集合完成匹配的时间。",
            "unlock_end_to_end_ms": "query_template_build_ms + unlock_match_ms；不包含模型/SIFT初始化和传感器采集。",
            "unlock_end_to_end_core_ms": (
                "unlock_end_to_end_ms 扣除 descriptor_prepare_ms、postprocess_ms、"
                "identity_match_overhead_ms、matching_wrapper_overhead_ms。"
            ),
            "end_to_end_summary": (
                "end_to_end 汇总使用 unlock_end_to_end_core_ms；"
                "end_to_end_wall_clock 为完整 unlock_end_to_end_ms。"
            ),
        },
        "total_genuine_attempts": total_genuine_attempts,
        "successful_unlocks_timed": successful_unlocks_timed,
        "failed_unlocks_excluded": failed_genuine_attempts,
        "match_only": summarize_timing(match_values),
        "end_to_end": summarize_timing(end_to_end_core_values),
        "end_to_end_wall_clock": summarize_timing(end_to_end_values),
    }
    write_json(Path(out) / "unlock_timing.json", summary)
    return summary


def run_descriptor_l2_evaluation(
    metadata_rows: list[dict[str, str]],
    identity_templates_path: str | Path,
    template_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    descriptor_source: str = "hardnet",
    max_impostor_identities_per_query: int = -1,
    export_failures: bool = True,
) -> dict[str, Any]:
    """执行完整 identity 级 descriptor 验证评估。

    距离算法由 matching.distance 与模板契约共同约束，可为 L2 或 Hamming。
    输入是已经固定好的 split metadata、identity_templates_20.json 和 image_templates。
    本函数不再重新构建模板，只负责：
        1. 遍历 query；
        2. 生成 genuine/impostor attempts；
        3. 写 verification_scores.csv；
        4. 计算阈值曲线和指标；
        5. 导出配置阈值下的失败样本。

    ``max_impostor_identities_per_query``：
        ``-1`` 全量非本人；``0`` 不比对非本人；``>0`` 每个 query 最多抽这么多个非本人身份。
    """

    out = ensure_dir(output_dir)
    source = str(descriptor_source).lower()
    if source == "rootsift":
        source = "sift"
    if source not in {"hardnet", "sift"}:
        raise ValueError(f"unsupported descriptor_source: {descriptor_source}")
    identities = load_identity_templates(identity_templates_path)
    query_rows = select_query_rows(metadata_rows)
    if not query_rows:
        raise ValueError("No query rows found in metadata.")
    identification_cfg = dict(config.get("identification", {}))
    configured_threshold = float(identification_cfg.get("match_score_threshold", 0.55))
    if not 0.0 <= configured_threshold <= 1.0:
        raise ValueError(
            f"identification.match_score_threshold must be in [0,1], got {configured_threshold}"
        )
    fusion_method = str(identification_cfg.get("fusion_method", "max"))
    management_cfg = dict(config.get("template_management", {}))
    scoring_config = copy.deepcopy(config)
    if bool(management_cfg.get("enabled", False)):
        # 模板库会按 LRU 改变顺序；学习模式必须完整遍历，避免早停导致
        # 同一 query 因模板顺序变化而得到不同的 identity 分数。
        scoring_identification = dict(scoring_config.get("identification", {}))
        scoring_identification["early_stop_on_unlock_threshold"] = False
        scoring_identification["early_stop_threshold"] = None
        scoring_config["identification"] = scoring_identification
    early_stop_threshold = resolve_early_stop_threshold(scoring_config)
    early_stop_enabled = early_stop_threshold is not None
    rng = np.random.default_rng(int(dict(config.get("enrollment", {})).get("random_seed", 42)))
    max_impostors = int(max_impostor_identities_per_query)
    if max_impostors < -1:
        raise ValueError(
            "runtime.max_impostor_identities_per_query 须为 -1（全量）、"
            f"0（不比对非本人）或正整数，收到: {max_impostors}"
        )
    cache: dict[str, dict[str, Any]] = {}
    scores_path = out / "verification_scores.csv"
    management_cfg = dict(config.get("template_management", {}))
    template_manager: TemplateLibraryManager | None = None
    learning_events: list[dict[str, Any]] = []
    if bool(management_cfg.get("enabled", False)):
        # identity_templates_path 在本次运行前刚由 build_identity_templates 重建，
        # 因此每次重跑都会从初始 seed 模板重新开始。
        template_manager = TemplateLibraryManager(
            identities,
            identity_templates_path,
            management_cfg,
            initial_template_count=int(
                dict(config.get("enrollment", {})).get(
                    "enrollment_images_per_identity",
                    20,
                )
            ),
        )

    # 第一遍：每个 query 先完整处理本人匹配和模板学习，再处理非本人匹配。
    score_rows: list[dict[str, str]] = []
    with scores_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        for row in tqdm(
            query_rows,
            desc=f"evaluate {source}_{dict(config.get('matching', {})).get('distance', 'auto')}",
        ):
            query_template_path = Path(template_dir) / template_filename(row["identity_id"], row["image_id"])
            query_template = load_template_cached(query_template_path, cache, require=source)
            true_identity = row["identity_id"]
            genuine_identity = next(
                (item for item in identities if str(item["identity_id"]) == true_identity),
                None,
            )
            if genuine_identity is None:
                raise RuntimeError(f"No registered identity found for query: {true_identity}")

            for path in genuine_identity.get("template_paths", []):
                load_template_cached(path, cache, require=source)
            started = time.perf_counter()
            genuine_result = score_query_against_identity(
                query_template,
                genuine_identity,
                scoring_config,
                cache,
                descriptor_source=source,
                early_stop_threshold=early_stop_threshold,
            )
            genuine_attempt_ms = (time.perf_counter() - started) * 1000.0

            if template_manager is not None:
                learning_events.append(
                    template_manager.learn_after_match(
                        query_template,
                        genuine_identity,
                        scoring_config,
                        genuine_result,
                        template_loader=lambda path: load_template_cached(
                            path,
                            cache,
                            require=source,
                        ),
                        descriptor_source=source,
                    )
                )

            # 学习完成后再读取 impostor 列表；新模板只影响当前 query 后续的匹配，
            # 不回写或重算已经完成的 genuine 结果。
            impostors = [
                item for item in identities
                if str(item["identity_id"]) != true_identity
            ]
            if max_impostors == 0:
                impostors = []
            elif max_impostors > 0 and len(impostors) > max_impostors:
                chosen = rng.choice(len(impostors), size=max_impostors, replace=False)
                impostors = [impostors[int(i)] for i in chosen]

            attempts: list[tuple[dict[str, Any], dict[str, Any], bool, float]] = [
                (genuine_identity, genuine_result, True, genuine_attempt_ms),
            ]
            for identity in impostors:
                for path in identity.get("template_paths", []):
                    load_template_cached(path, cache, require=source)
                started = time.perf_counter()
                result = score_query_against_identity(
                    query_template,
                    identity,
                    scoring_config,
                    cache,
                    descriptor_source=source,
                    early_stop_threshold=early_stop_threshold,
                )
                attempts.append(
                    (
                        identity,
                        result,
                        False,
                        (time.perf_counter() - started) * 1000.0,
                    )
                )

            for identity, result, is_genuine, attempt_ms in attempts:
                owner = str(identity["identity_id"])
                score = float(result["score"])
                query_build_ms = parse_float(row.get("template_build_ms"))
                unlock_end_to_end_ms = (
                    query_build_ms + attempt_ms
                    if is_genuine and query_build_ms is not None
                    else None
                )
                unlock_end_to_end_core_ms = (
                    compute_end_to_end_core_ms(unlock_end_to_end_ms, result)
                    if unlock_end_to_end_ms is not None
                    else None
                )
                csv_row = {
                    "query_id": row["image_id"],
                    "query_identity": true_identity,
                    "owner_identity": owner,
                    "label": int(is_genuine),
                    "score": score,
                    "accepted_at_selected_threshold": int(score >= configured_threshold),
                    "query_image_path": str(query_template.get("image_path", row.get("image_path", ""))),
                    "query_template_path": str(query_template_path),
                    **{
                        key: value
                        for key, value in result.items()
                        if key in SCORE_FIELDNAMES
                    },
                    "attempt_match_ms": f"{attempt_ms:.3f}",
                    "query_template_build_ms": "" if query_build_ms is None else f"{query_build_ms:.3f}",
                    "unlock_match_ms": f"{attempt_ms:.3f}" if is_genuine else "",
                    "unlock_end_to_end_ms": "" if unlock_end_to_end_ms is None else f"{unlock_end_to_end_ms:.3f}",
                    "unlock_end_to_end_core_ms": (
                        ""
                        if unlock_end_to_end_core_ms is None
                        else f"{unlock_end_to_end_core_ms:.3f}"
                    ),
                }
                writer.writerow(csv_row)
                score_rows.append(
                    {
                        field: "" if csv_row.get(field) is None else str(csv_row.get(field, ""))
                        for field in SCORE_FIELDNAMES
                    }
                )

    # 第二遍：使用全部 query 的 identity 级分数生成阈值曲线和统一指标。
    labels = np.asarray([int(row["label"]) for row in score_rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in score_rows], dtype=np.float32)
    curve = build_threshold_curve(labels, scores, config)
    metrics = compute_metrics(
        labels,
        scores,
        configured_threshold,
    )
    # FAR=0 工作点使用 CSV 中的原始 float64 分数，避免 float32 舍入把
    # 最大 impostor 分数压低后产生一个实际仍会误接受的阈值。
    zero_far_scores = np.asarray(
        [float(row["score"]) for row in score_rows], dtype=np.float64
    )
    zero_far_operating_point = select_zero_far_min_frr_threshold(
        labels, zero_far_scores
    )
    metrics["zero_far_min_frr_operating_point"] = zero_far_operating_point
    selected_threshold = configured_threshold
    metrics["configured_threshold"] = configured_threshold
    metrics["selected_threshold"] = selected_threshold
    metrics["fixed_threshold"] = rate_at_threshold(labels, scores, selected_threshold)

    for row in score_rows:
        score = float(row["score"])
        row["accepted_at_selected_threshold"] = int(score >= selected_threshold)
    write_csv_rows(out / "verification_scores.csv", score_rows)
    unlock_timing = write_unlock_timing_report(score_rows, out)
    failure_summary = (
        export_failure_cases(
            score_rows,
            {"threshold": selected_threshold},
            out,
            config,
        )
        if export_failures
        else {"enabled": False, "reason": "export_failures_false"}
    )
    effective_config = build_effective_config_snapshot(scoring_config)
    template_management_summary = (
        template_manager.summarize_events(learning_events)
        if template_manager is not None
        else {
            "enabled": False,
            "requested_in_config": bool(management_cfg.get("enabled", False)),
            "reason": "template_management_disabled",
        }
    )
    num_queries = len(
        {
            (str(row.get("query_identity", "")), str(row.get("query_id", "")))
            for row in score_rows
        }
    )
    metrics.update(
        {
            "descriptor_source": source,
            "matching_backend": matching_backend_name(source, scoring_config),
            "num_queries": num_queries,
            "num_identities": len(identities),
            "num_match_attempts": len(score_rows),
            "num_genuine_attempts": int(np.sum(labels == 1)),
            "num_impostor_attempts": int(np.sum(labels == 0)),
            "fusion_method": fusion_method,
            "scoring_mode": (
                "early_stop_query_pool_evaluation"
                if early_stop_enabled
                else "full_query_pool_evaluation"
            ),
            "offline_full_template_scoring": not early_stop_enabled,
            "early_stop_on_unlock_threshold": early_stop_enabled,
            "early_stop_threshold": early_stop_threshold,
            "max_impostor_identities_per_query": max_impostors,
            "matching_config": dict(scoring_config.get("matching", {})),
            "texture_verification_config": dict(scoring_config.get("texture_verification", {})),
            "template_management": template_management_summary,
            "score_component_summary": summarize_score_components(
                score_rows,
                selected_threshold,
            ),
            "matching_stage_summary": summarize_matching_stages(
                score_rows,
                scoring_config,
            ),
            "effective_config": effective_config,
            "unlock_timing": unlock_timing,
            "failure_export": failure_summary,
        }
    )
    write_csv_rows(out / "match_score_threshold_curve.csv", curve)
    far_frr_table_cfg = resolve_far_frr_table_config(config)
    far_frr_table_path = out / far_frr_table_filename(
        far_frr_table_cfg["min_threshold"],
        far_frr_table_cfg["max_threshold"],
    )
    far_frr_table_rows = build_far_frr_table(
        labels,
        scores,
        min_threshold=far_frr_table_cfg["min_threshold"],
        max_threshold=far_frr_table_cfg["max_threshold"],
        step=far_frr_table_cfg["step"],
    )
    write_csv_rows(far_frr_table_path, far_frr_table_rows)
    metrics["far_frr_table"] = {
        **far_frr_table_cfg,
        "path": str(far_frr_table_path),
        "num_rows": len(far_frr_table_rows),
    }
    per_finger_far_frr_path = out / "per_finger_far_frr_at_global_zero_far.csv"
    write_csv_rows(
        per_finger_far_frr_path,
        build_per_finger_far_frr_table(score_rows, zero_far_operating_point),
    )
    write_json(out / "metrics.json", metrics)
    write_yaml(out / "effective_config.yaml", effective_config)
    write_plots(labels, scores, curve, selected_threshold, metrics, out)
    return {
        "metrics": metrics,
        "scores_path": str(scores_path),
        "far_frr_table_path": str(far_frr_table_path),
        "per_finger_far_frr_path": str(per_finger_far_frr_path),
    }


def run_hardnet_evaluation(
    metadata_rows: list[dict[str, str]],
    identity_templates_path: str | Path,
    template_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    max_impostor_identities_per_query: int = -1,
    export_failures: bool = True,
) -> dict[str, Any]:
    """兼容旧入口：按 HardNet 模板契约评估 L2 或 Hamming。"""

    return run_descriptor_l2_evaluation(
        metadata_rows=metadata_rows,
        identity_templates_path=identity_templates_path,
        template_dir=template_dir,
        config=config,
        output_dir=output_dir,
        descriptor_source="hardnet",
        max_impostor_identities_per_query=max_impostor_identities_per_query,
        export_failures=export_failures,
    )
