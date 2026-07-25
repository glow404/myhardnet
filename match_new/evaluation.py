"""HardNet L2 身份验证评估与失败样本导出。

作用：
    这个脚本负责把“图像级匹配器”扩展成“手机解锁式 identity 级验证实验”：

    1. 每个 query 对本人 identity 的 20 张注册模板打分，得到 genuine attempt；
    2. 同一个 query 对所有非本人 identity 的 20 张注册模板打分，得到 impostor attempts；
    3. identity 级分数默认取 20 张模板中的最大图像级 score；
    4. 根据 identity 级 score 计算 FAR、FRR、EER、AUC、TAR@FAR；
    5. 按目标约束 FAR < 1/50000、FRR < 2% 选择 target_operating_point； todo：这个可以去掉了
    6. 在该阈值下导出 false reject / false accept 原图和拼接预览图。

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

from match_new.identity_matcher import load_template_cached, score_query_against_identity
from match_new.template_builder import load_identity_templates
from match_new.utils import ensure_dir, read_csv_rows, resolve_path, safe_id, template_filename, write_csv_rows, write_json, write_yaml


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
    "num_candidates",
    "num_inliers",
    "raw_inliers",
    "unique_inliers",
    "unique_query_inliers",
    "unique_gallery_inliers",
    "inlier_ratio",
    "mean_l2_distance",
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
]


def parse_float(value: Any) -> float | None:
    """解析 CSV 文本形式的耗时值；空值或非法值返回 ``None``。"""

    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    thresholds = sorted({round(value, 10) for value in thresholds if 0.0 <= value <= 1.0})
    return [rate_at_threshold(labels, scores, threshold) for threshold in thresholds]


def recommend_unlock_threshold(curve: list[dict[str, Any]], far_points: list[float]) -> dict[str, Any] | None:
    """旧版推荐阈值逻辑。

    优先找 FAR=0 且 FRR 最低的阈值；如果没有 FAR=0，则使用 far_points 中最严格点。
    保留它是为了和旧 `match/` 实验结果对照。
    """

    if not curve:
        return None

    def rate(item: dict[str, Any], key: str, default: float) -> float:
        value = item.get(key)
        return default if value is None else float(value)

    zero_far = [item for item in curve if rate(item, "far", 1.0) == 0.0]
    if zero_far:
        best = min(zero_far, key=lambda item: (rate(item, "frr", 1.0), float(item["threshold"])))
        reason = "lowest_frr_at_far_0"
    else:
        strict_far = min([float(point) for point in far_points], default=1.0)
        valid = [item for item in curve if rate(item, "far", 1.0) <= strict_far]
        if valid:
            best = min(valid, key=lambda item: (rate(item, "frr", 1.0), float(item["threshold"])))
            reason = f"lowest_frr_at_far_le_{strict_far}"
        else:
            best = min(curve, key=lambda item: (rate(item, "far", 1.0), rate(item, "frr", 1.0), float(item["threshold"])))
            reason = "lowest_available_far_then_frr"
    frr = rate(best, "frr", 1.0)
    return {"threshold": float(best["threshold"]), "far": rate(best, "far", 1.0), "frr": frr, "tar": 1.0 - frr, "reason": reason}


def select_target_operating_threshold(curve: list[dict[str, Any]], target_far: float, target_frr: float) -> dict[str, Any] | None:
    """按正式目标选择操作阈值。

    当前目标为：
        FAR < 1/50000
        FRR < 2%

    选择策略：
        1. 若存在同时满足两项约束的阈值，取其中 FRR 最低者；
        2. 若只能满足 FAR，则报告“FRR 未达标”并取 FAR 约束内 FRR 最低者；
        3. 若只能满足 FRR，则报告“FAR 未达标”并取 FRR 约束内 FAR 最低者；
        4. 若两项都不能满足，取相对违反程度最小的阈值。
    """

    if not curve:
        return None

    def rate(item: dict[str, Any], key: str, default: float) -> float:
        value = item.get(key)
        return default if value is None else float(value)

    target_far = float(target_far)
    target_frr = float(target_frr)
    both_valid = [item for item in curve if rate(item, "far", 1.0) < target_far and rate(item, "frr", 1.0) < target_frr]
    if both_valid:
        best = min(both_valid, key=lambda item: (rate(item, "frr", 1.0), float(item["threshold"])))
        reason = "meets_far_and_frr_targets"
        satisfied = True
    else:
        far_valid = [item for item in curve if rate(item, "far", 1.0) < target_far]
        if far_valid:
            best = min(far_valid, key=lambda item: (rate(item, "frr", 1.0), float(item["threshold"])))
            reason = "frr_target_not_met_under_far_target"
        else:
            frr_valid = [item for item in curve if rate(item, "frr", 1.0) < target_frr]
            if frr_valid:
                best = min(frr_valid, key=lambda item: (rate(item, "far", 1.0), rate(item, "frr", 1.0), float(item["threshold"])))
                reason = "far_target_not_met_under_frr_target"
            else:
                def violation(item: dict[str, Any]) -> tuple[float, float, float]:
                    far = rate(item, "far", 1.0)
                    frr = rate(item, "frr", 1.0)
                    far_over = max(far - target_far, 0.0) / max(target_far, 1e-12)
                    frr_over = max(frr - target_frr, 0.0) / max(target_frr, 1e-12)
                    return far_over + frr_over, frr, float(item["threshold"])

                best = min(curve, key=violation)
                reason = "no_threshold_meets_either_target"
        satisfied = False

    far = rate(best, "far", 1.0)
    frr = rate(best, "frr", 1.0)
    return {
        "threshold": float(best["threshold"]),
        "far": far,
        "frr": frr,
        "tar": 1.0 - frr,
        "target_far": target_far,
        "target_frr": target_frr,
        "satisfied": bool(satisfied),
        "reason": reason,
        "genuine_accept": best.get("genuine_accept"),
        "genuine_reject": best.get("genuine_reject"),
        "genuine_total": best.get("genuine_total"),
        "impostor_accept": best.get("impostor_accept"),
        "impostor_reject": best.get("impostor_reject"),
        "impostor_total": best.get("impostor_total"),
    }


def compute_metrics(labels: np.ndarray, scores: np.ndarray, selected_threshold: float, far_points: list[float], curve: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """汇总全局指标。

    输出包括：
        - fixed_threshold：历史固定阈值下的 FAR/FRR；
        - recommended_threshold：旧推荐逻辑；
        - target_operating_point：按 FAR<1/50000、FRR<2% 找到的正式阈值；
        - EER / AUC / TAR@FAR。
    """

    eval_cfg = dict(config.get("evaluation", {}))
    target_far = float(eval_cfg.get("target_far", 1.0 / 50000.0))
    target_frr = float(eval_cfg.get("target_frr", 0.02))
    metrics: dict[str, Any] = {
        "selected_threshold": float(selected_threshold),
        "fixed_threshold": rate_at_threshold(labels, scores, selected_threshold),
        "auc": None,
        "eer": None,
        "eer_threshold": None,
        "tar_at_far": {str(point): None for point in far_points},
        "recommended_threshold": recommend_unlock_threshold(curve, far_points),
        "target_operating_point": select_target_operating_threshold(curve, target_far, target_frr),
    }
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return metrics
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    metrics["auc"] = float(auc(fpr, tpr))
    metrics["eer"] = float((fpr[idx] + fnr[idx]) / 2.0)
    metrics["eer_threshold"] = float(thresholds[idx])
    for point in far_points:
        valid = np.where(fpr <= float(point))[0]
        metrics["tar_at_far"][str(point)] = float(np.max(tpr[valid])) if valid.size else 0.0
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
        "online_unlock",
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
    """在 target threshold 下导出失败样本。

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
    root = ensure_dir(Path(output_dir) / "failure_cases" / f"threshold_{safe_id(str(threshold))}")

    # 先扫描 verification_scores，找出所有目标阈值下的失败行。
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
        failures.append(
            {
                "case_index": index,
                "failure_type": failure_type,
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
                "num_candidates": row.get("num_candidates", ""),
                "raw_inliers": row.get("raw_inliers", ""),
                "unique_inliers": row.get("unique_inliers", ""),
                "mean_l2_distance": row.get("mean_l2_distance", ""),
                "mean_reproj_error": row.get("mean_reproj_error", ""),
                "orientation_consistency": row.get("orientation_consistency", ""),
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
            case_id = f"{failure_type}_{local_idx:05d}__score_{safe_id(str(item['score']))}__q_{safe_id(item['query_id'])}__owner_{safe_id(item['owner_identity'])}"
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
                    score_text=f"score={item['score']} threshold={threshold} unique={item['unique_inliers']} raw={item['raw_inliers']}",
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
    summary = {
        "enabled": True,
        "threshold": threshold,
        "target_satisfied": bool(threshold_info.get("satisfied", False)),
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


def write_plots(labels: np.ndarray, scores: np.ndarray, curve: list[dict[str, Any]], selected_threshold: float, output_dir: Path) -> None:
    """写出分数分布图和 FAR/FRR 阈值曲线图。"""

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
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far, label="FAR")
    plt.plot(thresholds, frr, label="FRR")
    plt.axvline(selected_threshold, color="gray", linestyle="--", linewidth=1)
    plt.legend()
    plt.xlabel("match score threshold")
    plt.tight_layout()
    plt.savefig(plots / "far_frr_vs_threshold.png", dpi=150)
    plt.close()


def write_unlock_timing_report(score_rows: list[dict[str, str]], output_dir: str | Path) -> dict[str, Any]:
    """从本人验证记录中生成手机解锁耗时明细和汇总报告。

    离线评估中的本人尝试可用于估算查询模板构建和完整模板匹配耗时，但该流程
    不启用在线早停，因此它不是最终在线延迟数据；准确的早停延迟应由
    ``run_online_unlock.py --benchmark`` 测量。
    """

    rows: list[dict[str, Any]] = []
    total_genuine_attempts = 0
    failed_genuine_attempts = 0
    match_values: list[float] = []
    end_to_end_values: list[float] = []
    for row in score_rows:
        if int(row.get("label", 0)) != 1:
            continue
        total_genuine_attempts += 1
        if int(row.get("accepted_at_selected_threshold", 0)) != 1:
            failed_genuine_attempts += 1
            continue
        match_ms = parse_float(row.get("unlock_match_ms"))
        end_to_end_ms = parse_float(row.get("unlock_end_to_end_ms"))
        query_build_ms = parse_float(row.get("query_template_build_ms"))
        if match_ms is not None:
            match_values.append(match_ms)
        if end_to_end_ms is not None:
            end_to_end_values.append(end_to_end_ms)
        rows.append(
            {
                "query_id": row.get("query_id", ""),
                "query_identity": row.get("query_identity", ""),
                "owner_identity": row.get("owner_identity", ""),
                "score": row.get("score", ""),
                "accepted_at_selected_threshold": row.get("accepted_at_selected_threshold", ""),
                "accepted_at_target_threshold": row.get("accepted_at_target_threshold", ""),
                "query_template_build_ms": "" if query_build_ms is None else query_build_ms,
                "unlock_match_ms": "" if match_ms is None else match_ms,
                "unlock_end_to_end_ms": "" if end_to_end_ms is None else end_to_end_ms,
                "best_template_image_id": row.get("best_template_image_id", ""),
                "num_templates": row.get("num_templates", ""),
                "num_templates_evaluated": row.get("num_templates_evaluated", ""),
                "early_stopped": row.get("early_stopped", ""),
                "early_stop_threshold": row.get("early_stop_threshold", ""),
                "unique_inliers": row.get("unique_inliers", ""),
                "num_candidates": row.get("num_candidates", ""),
                "texture_similarity": row.get("texture_similarity", ""),
                "texture_overlap_fraction": row.get("texture_overlap_fraction", ""),
                "texture_valid_blocks": row.get("texture_valid_blocks", ""),
                "geometry_similarity": row.get("geometry_similarity", ""),
                "geometry_weight": row.get("geometry_weight", ""),
                "texture_weight": row.get("texture_weight", ""),
                "texture_decision": row.get("texture_decision", ""),
                "registered_template_load_ms": row.get("registered_template_load_ms", ""),
                "descriptor_prepare_ms": row.get("descriptor_prepare_ms", ""),
                "candidate_generation_ms": row.get("candidate_generation_ms", ""),
                "candidate_filter_ms": row.get("candidate_filter_ms", ""),
                "ransac_ms": row.get("ransac_ms", ""),
                "inlier_refinement_ms": row.get("inlier_refinement_ms", ""),
                "texture_similarity_ms": row.get("texture_similarity_ms", ""),
                "score_fusion_ms": row.get("score_fusion_ms", ""),
                "postprocess_ms": row.get("postprocess_ms", ""),
                "image_match_total_ms": row.get("image_match_total_ms", ""),
                "identity_fusion_ms": row.get("identity_fusion_ms", ""),
                "identity_match_overhead_ms": row.get("identity_match_overhead_ms", ""),
                "identity_match_total_ms": row.get("identity_match_total_ms", ""),
            }
        )
    out = ensure_dir(output_dir)
    write_csv_rows(Path(out) / "unlock_timing.csv", rows)
    summary = {
        "definition": {
            "timing_scope": "只统计在 selected threshold 下成功接受的本人解锁记录。",
            "unlock_match_ms": "查询模板已存在时，与一个手指的注册模板集合完成匹配的时间。",
            "unlock_end_to_end_ms": "query_template_build_ms + unlock_match_ms；不包含模型/SIFT初始化和传感器采集。",
        },
        "total_genuine_attempts": total_genuine_attempts,
        "successful_unlocks_timed": len(rows),
        "failed_unlocks_excluded": failed_genuine_attempts,
        "match_only": summarize_timing(match_values),
        "end_to_end": summarize_timing(end_to_end_values),
    }
    write_json(Path(out) / "unlock_timing.json", summary)
    return summary


def run_descriptor_l2_evaluation(
    metadata_path: str | Path,
    identity_templates_path: str | Path,
    template_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    descriptor_source: str = "hardnet",
    max_impostor_identities_per_query: int = 0,
    export_failures: bool = True,
) -> dict[str, Any]:
    """执行完整 identity 级 L2 descriptor 验证评估。

    输入是已经固定好的 split metadata、identity_templates_20.json 和 image_templates。
    本函数不再重新构建模板，只负责：
        1. 遍历 query；
        2. 生成 genuine/impostor attempts；
        3. 写 verification_scores.csv；
        4. 计算阈值曲线和指标；
        5. 导出目标阈值下失败样本。
    """

    out = ensure_dir(output_dir)
    source = str(descriptor_source).lower()
    if source == "rootsift":
        source = "sift"
    if source not in {"hardnet", "sift"}:
        raise ValueError(f"unsupported descriptor_source: {descriptor_source}")
    identities = load_identity_templates(identity_templates_path)
    rows = read_csv_rows(metadata_path)
    query_rows = select_query_rows(rows)
    identification_cfg = dict(config.get("identification", {}))
    evaluation_cfg = dict(config.get("evaluation", {}))
    selected_threshold = float(identification_cfg.get("match_score_threshold", 0.55))
    if not 0.0 <= selected_threshold <= 1.0:
        raise ValueError(
            f"identification.match_score_threshold must be in [0,1], got {selected_threshold}"
        )
    fusion_method = str(identification_cfg.get("fusion_method", "max"))
    # 离线标定必须遍历全部注册模板并保留真实 identity 最大分数，否则阈值曲线、
    # FAR 和 FRR 会受模板顺序影响。固定阈值早停仅由独立的在线解锁引擎处理。
    scoring_config = copy.deepcopy(config)
    scoring_config.setdefault("identification", {})["early_stop_on_unlock_threshold"] = False
    far_points = [float(point) for point in evaluation_cfg.get("far_points", [0.001, 0.0001])]
    rng = np.random.default_rng(int(dict(config.get("enrollment", {})).get("random_seed", 42)))
    max_impostors = max(0, int(max_impostor_identities_per_query))
    cache: dict[str, dict[str, Any]] = {}
    scores_path = out / "verification_scores.csv"
    management_cfg = dict(config.get("template_management", {}))

    # 第一遍：逐 query 逐 identity 打分，并把每次验证尝试写入 CSV。
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDNAMES)
        writer.writeheader()
        for row in tqdm(query_rows, desc=f"evaluate {source}_l2"):
            query_template_path = Path(template_dir) / template_filename(row["identity_id"], row["image_id"])
            query_template = load_template_cached(query_template_path, cache, require=source)
            true_identity = row["identity_id"]
            genuine = [item for item in identities if item["identity_id"] == true_identity]
            impostors = [item for item in identities if item["identity_id"] != true_identity]
            if max_impostors > 0 and len(impostors) > max_impostors:
                chosen = rng.choice(len(impostors), size=max_impostors, replace=False)
                impostors = [impostors[int(i)] for i in chosen]
            # 每个 query 一定包含 1 个 genuine owner，外加若干 impostor owners。
            for identity in genuine + impostors:
                owner = str(identity["identity_id"])
                for path in identity.get("template_paths", []):
                    load_template_cached(path, cache, require=source)
                started = time.perf_counter()
                result = score_query_against_identity(
                    query_template,
                    identity,
                    scoring_config,
                    cache,
                    descriptor_source=source,
                    early_stop_threshold=None,
                )
                attempt_ms = (time.perf_counter() - started) * 1000.0
                score = float(result["score"])
                is_genuine = owner == true_identity
                query_build_ms = parse_float(row.get("template_build_ms"))
                unlock_end_to_end_ms = query_build_ms + attempt_ms if is_genuine and query_build_ms is not None else None
                writer.writerow(
                    {
                        "query_id": row["image_id"],
                        "query_identity": true_identity,
                        "owner_identity": owner,
                        "label": int(is_genuine),
                        "score": score,
                        "accepted_at_selected_threshold": int(score >= selected_threshold),
                        "query_image_path": str(query_template.get("image_path", row.get("image_path", ""))),
                        "query_template_path": str(query_template_path),
                        **result,
                        "attempt_match_ms": f"{attempt_ms:.3f}",
                        "query_template_build_ms": "" if query_build_ms is None else f"{query_build_ms:.3f}",
                        "unlock_match_ms": f"{attempt_ms:.3f}" if is_genuine else "",
                        "unlock_end_to_end_ms": "" if unlock_end_to_end_ms is None else f"{unlock_end_to_end_ms:.3f}",
                    }
                )

    # 第二遍：读取分数 CSV，统一计算指标和目标阈值。
    score_rows = read_csv_rows(scores_path)
    labels = np.asarray([int(row["label"]) for row in score_rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in score_rows], dtype=np.float32)
    curve = build_threshold_curve(labels, scores, config)
    metrics = compute_metrics(labels, scores, selected_threshold, far_points, curve, config)
    target = metrics.get("target_operating_point") or {}
    target_threshold = float(target.get("threshold", selected_threshold))
    # 额外写一份带 target threshold 判定结果的 CSV，方便后续人工筛选。
    for row in score_rows:
        score = float(row["score"])
        row["accepted_at_target_threshold"] = int(score >= target_threshold)
        row["target_failure_type"] = "false_reject" if int(row["label"]) == 1 and score < target_threshold else ("false_accept" if int(row["label"]) == 0 and score >= target_threshold else "")
    write_csv_rows(out / "verification_scores_target_threshold.csv", score_rows)
    unlock_timing = write_unlock_timing_report(score_rows, out)
    failure_summary = export_failure_cases(score_rows, metrics.get("target_operating_point"), out, config) if export_failures else {"enabled": False, "reason": "export_failures_false"}
    effective_config = build_effective_config_snapshot(scoring_config)
    template_management_summary = {
        "enabled": False,
        "requested_in_config": bool(management_cfg.get("enabled", False)),
        "reason": "offline_calibration_never_updates_templates",
    }
    metrics.update(
        {
            "descriptor_source": source,
            "matching_backend": matching_backend_name(source, scoring_config),
            "num_queries": len(query_rows),
            "num_identities": len(identities),
            "num_match_attempts": len(score_rows),
            "num_genuine_attempts": int(np.sum(labels == 1)),
            "num_impostor_attempts": int(np.sum(labels == 0)),
            "fusion_method": fusion_method,
            "scoring_mode": "offline_full_threshold_calibration",
            "offline_full_template_scoring": True,
            "early_stop_on_unlock_threshold": False,
            "early_stop_threshold": None,
            "max_impostor_identities_per_query": max_impostors,
            "matching_config": dict(scoring_config.get("matching", {})),
            "texture_verification_config": dict(scoring_config.get("texture_verification", {})),
            "template_management": template_management_summary,
            "score_component_summary": summarize_score_components(score_rows, selected_threshold),
            "effective_config": effective_config,
            "unlock_timing": unlock_timing,
            "failure_export": failure_summary,
        }
    )
    write_csv_rows(out / "match_score_threshold_curve.csv", curve)
    write_json(out / "metrics.json", metrics)
    write_yaml(out / "effective_config.yaml", effective_config)
    write_plots(labels, scores, curve, selected_threshold, out)
    return {"metrics": metrics, "scores_path": str(scores_path)}


def run_hardnet_evaluation(
    metadata_path: str | Path,
    identity_templates_path: str | Path,
    template_dir: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    max_impostor_identities_per_query: int = 0,
    export_failures: bool = True,
) -> dict[str, Any]:
    """兼容旧入口：只评估 HardNet L2。"""

    return run_descriptor_l2_evaluation(
        metadata_path=metadata_path,
        identity_templates_path=identity_templates_path,
        template_dir=template_dir,
        config=config,
        output_dir=output_dir,
        descriptor_source="hardnet",
        max_impostor_identities_per_query=max_impostor_identities_per_query,
        export_failures=export_failures,
    )


def summary_row(metrics: dict[str, Any], far_points: list[float]) -> dict[str, Any]:
    """把 metrics.json 压平成一行 summary CSV。"""

    fixed = metrics.get("fixed_threshold") or {}
    rec = metrics.get("recommended_threshold") or {}
    target = metrics.get("target_operating_point") or {}
    failure_export = metrics.get("failure_export") or {}
    texture_config = metrics.get("texture_verification_config") or {}
    row: dict[str, Any] = {
        "descriptor_source": metrics.get("descriptor_source", "hardnet"),
        "matching_backend": metrics.get("matching_backend", "hardnet_l2_unknown_ransac"),
        "num_queries": metrics.get("num_queries", ""),
        "num_identities": metrics.get("num_identities", ""),
        "num_match_attempts": metrics.get("num_match_attempts", ""),
        "num_genuine_attempts": metrics.get("num_genuine_attempts", ""),
        "num_impostor_attempts": metrics.get("num_impostor_attempts", ""),
        "scoring_mode": metrics.get("scoring_mode", ""),
        "offline_full_template_scoring": metrics.get("offline_full_template_scoring", ""),
        "early_stop_on_unlock_threshold": metrics.get("early_stop_on_unlock_threshold", ""),
        "early_stop_threshold": metrics.get("early_stop_threshold", ""),
        "texture_verification_enabled": texture_config.get("enabled", ""),
        "geometry_weight": texture_config.get("geometry_weight", ""),
        "texture_weight": texture_config.get("texture_weight", ""),
        "geometry_saturation_inliers": texture_config.get("geometry_saturation_inliers", ""),
        "selected_threshold": metrics.get("selected_threshold", ""),
        "eer": metrics.get("eer", ""),
        "eer_threshold": metrics.get("eer_threshold", ""),
        "auc": metrics.get("auc", ""),
        "far_at_selected_threshold": fixed.get("far", ""),
        "frr_at_selected_threshold": fixed.get("frr", ""),
        "tar_at_selected_threshold": fixed.get("tar", ""),
        "recommended_threshold": rec.get("threshold", ""),
        "recommended_far": rec.get("far", ""),
        "recommended_frr": rec.get("frr", ""),
        "recommended_tar": rec.get("tar", ""),
        "recommended_reason": rec.get("reason", ""),
        "target_threshold": target.get("threshold", ""),
        "target_far": target.get("far", ""),
        "target_frr": target.get("frr", ""),
        "target_tar": target.get("tar", ""),
        "target_satisfied": target.get("satisfied", ""),
        "target_reason": target.get("reason", ""),
        "target_far_requirement": target.get("target_far", ""),
        "target_frr_requirement": target.get("target_frr", ""),
        "num_false_rejects_at_target": failure_export.get("num_false_rejects", ""),
        "num_false_accepts_at_target": failure_export.get("num_false_accepts", ""),
        "failure_dir": failure_export.get("failure_dir", ""),
    }
    tar_at_far = metrics.get("tar_at_far") or {}
    for point in far_points:
        tar = tar_at_far.get(str(point), "")
        row[f"tar_at_far_{point}"] = tar
        row[f"frr_at_far_{point}"] = "" if tar in {"", None} else 1.0 - float(tar)
    return row
