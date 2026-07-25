"""用于测量优化后单次指纹匹配耗时的在线引擎。

进程启动时加载模型、SIFT 检测器和注册模板；每次请求仅从一张原始指纹图像
在内存中构建查询模板，与指定手指的注册模板匹配并记录各阶段时间。注册模板
和 query 划分路径直接从在线配置解析，不依赖额外部署档案。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from match_new.identity_matcher import load_template_cached, score_query_against_identity
from match_new.template_builder import (
    build_hardnet_template_from_image,
    load_identity_templates,
)
from match_new.runtime import HardNetDescriptor, build_sift


def percentile(values: list[float], q: float) -> float | None:
    """使用 NumPy 计算耗时样本的指定百分位数；空样本返回 ``None``。"""

    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_timings(values: list[float], percentiles: list[int]) -> dict[str, Any]:
    """汇总一组毫秒耗时，包括最快、平均、最慢及配置指定的百分位数。"""

    if not values:
        return {"count": 0}
    summary: dict[str, Any] = {
        "count": len(values),
        "fastest_ms": float(min(values)),
        "average_ms": float(np.mean(values)),
        "slowest_ms": float(max(values)),
    }
    for point in percentiles:
        summary[f"p{int(point)}_ms"] = percentile(values, int(point))
    return summary


def summarize_timing_section(
    attempts: list[dict[str, Any]],
    total_field: str,
    stage_fields: tuple[str, ...],
    percentiles: list[int],
) -> dict[str, Any]:
    """汇总总耗时，并为每个内部阶段生成相同口径的统计。"""

    section = summarize_timings(
        [float(row[total_field]) for row in attempts],
        percentiles,
    )
    section["stages"] = {
        field: summarize_timings(
            [float(row.get(field, 0.0)) for row in attempts],
            percentiles,
        )
        for field in stage_fields
    }
    return section


def resolve_artifact_path(
    artifacts_dir: Path,
    configured_path: str | Path | None,
    default_filename: str,
) -> Path:
    """解析在线测试所需的离线产物路径。

    绝对路径按原值使用；相对路径相对于 ``artifacts_dir`` 解析。配置未填写时
    使用按注册模板数量推导出的默认文件名。
    """

    raw_path = configured_path or default_filename
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (artifacts_dir / path).resolve()


class OnlineUnlockEngine:
    """在多个解锁请求之间常驻模型、检测器、模板缓存和固定阈值。

    构造实例代表一次应用进程启动。初始化和预热耗时单独记录，不计入单次解锁
    延迟；同一个实例可连续处理多次请求，以模拟手机系统服务常驻运行的场景。
    """

    def __init__(
        self,
        config: dict[str, Any],
        artifacts_dir: str | Path,
    ) -> None:
        """解析离线产物路径，并按当前在线配置初始化耗时测试组件。"""

        self.config = config
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        online_cfg = dict(config.get("online_unlock", {}))
        enrollment_count = int(
            dict(config.get("enrollment", {})).get(
                "enrollment_images_per_identity",
                20,
            )
        )
        self.identity_templates_path = resolve_artifact_path(
            self.artifacts_dir,
            online_cfg.get("identity_templates"),
            f"identity_templates_{enrollment_count}.json",
        )
        self.split_metadata_path = resolve_artifact_path(
            self.artifacts_dir,
            online_cfg.get("split_metadata"),
            f"metadata_with_split_{enrollment_count}.csv",
        )
        self.image_templates_dir = resolve_artifact_path(
            self.artifacts_dir,
            online_cfg.get("image_templates_dir"),
            "image_templates",
        )
        if not self.identity_templates_path.exists():
            raise FileNotFoundError(
                f"注册模板索引不存在: {self.identity_templates_path}"
            )

        # 在线参数完全来自当前配置，可以独立切换 GPU 精度、patch 和匹配参数。
        identification_cfg = dict(config.get("identification", {}))
        configured_early_stop = identification_cfg.get("early_stop_threshold")
        threshold_value = (
            configured_early_stop
            if configured_early_stop not in {"", None}
            else identification_cfg.get("match_score_threshold", 0.55)
        )
        self.threshold = float(threshold_value)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"在线匹配阈值必须位于 [0,1]，当前值为 {self.threshold}")
        self.descriptor_source = "hardnet"
        self.fusion_method = str(
            identification_cfg.get("fusion_method", "max")
        ).lower()
        self.early_stop_enabled = bool(
            identification_cfg.get("early_stop_on_unlock_threshold", True)
        )
        if (
            self.early_stop_enabled
            and self.fusion_method not in {"max", "max_quality_tiebreak"}
        ):
            raise ValueError(
                "在线早停只支持 max 或 max_quality_tiebreak 融合；"
                f"当前融合方式为 {self.fusion_method}"
            )

        self.identities = load_identity_templates(self.identity_templates_path)
        self.identity_by_id = {
            str(identity["identity_id"]): identity
            for identity in self.identities
        }
        if not self.identity_by_id:
            raise RuntimeError(
                f"注册模板索引中没有 identity: {self.identity_templates_path}"
            )

        self.template_cache: dict[str, dict[str, Any]] = {}
        if bool(online_cfg.get("persist_query_template", False)):
            raise ValueError(
                "online unlock builds query templates in memory only; "
                "set online_unlock.persist_query_template=false"
            )
        if bool(online_cfg.get("template_learning_after_decision", False)):
            raise ValueError(
                "online template learning is not part of the fixed-threshold latency path yet; "
                "set online_unlock.template_learning_after_decision=false"
            )

        started = time.perf_counter()
        self.hardnet = HardNetDescriptor(config)
        self.sift = build_sift(config)
        self.model_initialization_ms = (time.perf_counter() - started) * 1000.0

        warmup_runs = max(0, int(online_cfg.get("model_warmup_runs", 0)))
        warmup_started = time.perf_counter()
        for _ in range(warmup_runs):
            self.hardnet.describe(np.zeros((1, 32, 32), dtype=np.float32))
        self.model_warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

        self.preload_templates = bool(online_cfg.get("preload_templates", True))
        self.preload_ms = 0.0
        if self.preload_templates:
            started = time.perf_counter()
            for identity_id in self.identity_by_id:
                self.preload_identity(identity_id)
            self.preload_ms = (time.perf_counter() - started) * 1000.0

    def preload_identity(self, identity_id: str) -> None:
        """把指定手指的全部注册模板载入内存缓存。"""

        identity = self.get_identity(identity_id)
        for path in identity.get("template_paths", []):
            load_template_cached(path, self.template_cache, require="hardnet")

    def get_identity(self, identity_id: str) -> dict[str, Any]:
        """返回指定已注册手指的模板集合；未注册时给出可用 ID 示例。"""

        key = str(identity_id)
        if key not in self.identity_by_id:
            examples = ", ".join(sorted(self.identity_by_id)[:10])
            raise KeyError(f"identity is not registered: {key}. Examples: {examples}")
        return self.identity_by_id[key]

    def unlock(
        self,
        image_path: str | Path,
        identity_id: str,
        *,
        image_id: str | None = None,
    ) -> dict[str, Any]:
        """从原始图像在内存中构建查询模板并返回一次固定阈值判定。

        ``end_to_end_ms`` 从读取查询图像开始，到 identity 级匹配完成为止；
        模型初始化、模型预热、注册模板预加载以及传感器采集图像的时间不包含在内。
        ``match_ms`` 仅统计已有查询模板与注册模板集合之间的匹配阶段。
        """

        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"query image does not exist: {path}")
        identity = self.get_identity(identity_id)
        if not self.preload_templates:
            self.preload_identity(identity_id)

        total_started = time.perf_counter()
        row = {
            "identity_id": str(identity_id),
            "image_id": str(image_id or path.stem),
            "image_path": str(path),
            "split": "query",
        }
        query_template, template_timings = build_hardnet_template_from_image(
            row,
            self.hardnet,
            self.sift,
            self.config,
        )

        started = time.perf_counter()
        match_result = score_query_against_identity(
            query_template,
            identity,
            self.config,
            self.template_cache,
            descriptor_source=self.descriptor_source,
            early_stop_threshold=self.threshold if self.early_stop_enabled else None,
        )
        match_ms = (time.perf_counter() - started) * 1000.0
        matching_wrapper_overhead_ms = max(
            0.0,
            match_ms - float(match_result.get("identity_match_total_ms", 0.0)),
        )
        decision_score = float(match_result["score"])
        result = {
            "identity_id": str(identity_id),
            "image_id": row["image_id"],
            "image_path": str(path),
            "accepted": bool(decision_score >= self.threshold),
            "threshold": self.threshold,
            "decision_score": decision_score,
            "decision_mode": (
                "fixed_threshold_early_stop"
                if self.early_stop_enabled
                else "fixed_threshold_full_template"
            ),
            "score_is_full_identity_max": not bool(match_result["early_stopped"]),
            **template_timings,
            "match_ms": match_ms,
            "matching_wrapper_overhead_ms": matching_wrapper_overhead_ms,
            "end_to_end_ms": (time.perf_counter() - total_started) * 1000.0,
            **match_result,
        }
        return result

    def benchmark(self, query_rows: list[dict[str, str]]) -> dict[str, Any]:
        """依次执行多次本人查询，并汇总解锁延迟、拒真率和早停比例。

        该基准复用同一个引擎实例，因此模型、SIFT 检测器和注册模板不会在每次
        查询时重复初始化或加载，更接近真实在线服务的工作方式。
        """

        attempts: list[dict[str, Any]] = []
        for row in tqdm(query_rows, desc="online unlock benchmark"):
            attempt = self.unlock(
                row["image_path"],
                row["identity_id"],
                image_id=row.get("image_id"),
            )
            attempt["included_in_timing_statistics"] = bool(attempt["accepted"])
            attempts.append(attempt)

        online_cfg = dict(self.config.get("online_unlock", {}))
        timing_cfg = dict(online_cfg.get("timing", {}))
        points = [int(value) for value in timing_cfg.get("percentiles", [50, 90, 95, 99])]
        accepted = sum(bool(row["accepted"]) for row in attempts)
        successful_attempts = [
            row for row in attempts if bool(row["accepted"])
        ]
        template_stage_fields = (
            "image_read_ms",
            "sift_keypoint_detection_ms",
            "keypoint_filter_ms",
            "patch_crop_rotate_ms",
            "hardnet_inference_ms",
            "template_assembly_ms",
            "template_pipeline_overhead_ms",
        )
        matching_stage_fields = (
            "registered_template_load_ms",
            "descriptor_prepare_ms",
            "candidate_generation_ms",
            "candidate_filter_ms",
            "ransac_ms",
            "inlier_refinement_ms",
            "texture_similarity_ms",
            "score_fusion_ms",
            "postprocess_ms",
            "identity_fusion_ms",
            "identity_match_overhead_ms",
            "matching_wrapper_overhead_ms",
        )
        summary = {
            "mode": "fixed_threshold_online_unlock",
            "timing_scope": "accepted_unlocks_only",
            "threshold": self.threshold,
            "fusion_method": self.fusion_method,
            "early_stop_enabled": self.early_stop_enabled,
            "num_attempts": len(attempts),
            "num_accepted": accepted,
            "num_rejected": len(attempts) - accepted,
            "num_timed_successful_unlocks": len(successful_attempts),
            "num_failed_excluded_from_timing": (
                len(attempts) - len(successful_attempts)
            ),
            "frr_on_genuine_benchmark": (
                float((len(attempts) - accepted) / len(attempts))
                if attempts
                else None
            ),
            "early_stop_rate_on_successful_unlocks": (
                float(
                    sum(bool(row["early_stopped"]) for row in successful_attempts)
                    / len(successful_attempts)
                )
                if successful_attempts
                else None
            ),
            "average_templates_evaluated_on_successful_unlocks": (
                float(
                    np.mean(
                        [
                            int(row["num_templates_evaluated"])
                            for row in successful_attempts
                        ]
                    )
                )
                if successful_attempts
                else None
            ),
            "template_build": summarize_timing_section(
                successful_attempts,
                "template_total_ms",
                template_stage_fields,
                points,
            ),
            "matching": summarize_timing_section(
                successful_attempts,
                "match_ms",
                matching_stage_fields,
                points,
            ),
            "end_to_end": summarize_timings(
                [float(row["end_to_end_ms"]) for row in successful_attempts],
                points,
            ),
            "startup": {
                "device": str(self.hardnet.device),
                "inference_precision": self.hardnet.inference_precision,
                "channels_last": self.hardnet.channels_last,
                "model_initialization_ms": self.model_initialization_ms,
                "model_warmup_ms": self.model_warmup_ms,
                "template_preload_ms": self.preload_ms,
            },
        }
        return {"attempts": attempts, "summary": summary}
