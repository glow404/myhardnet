"""HardNet L2 手机解锁式指纹验证主入口。

作用：
    1. 扫描原始指纹图像目录，构建每张图像的 `.npz` 模板；
    2. 每个 identity 随机选择注册模板；
    3. 用其余图像作为 query，分别对本人和非本人 identity 模板库打分；
    4. 输出 FAR/FRR/EER/AUC、满足目标 FAR/FRR 的推荐阈值；
    5. 在目标阈值下导出 false reject / false accept 的原图和拼接预览。

运行参数默认写在 `config_match_new.yaml` 的 output/runtime/data/model/evaluation 等段；
命令行仅作覆盖，优先级：命令行 > 配置文件 > 程序默认值。

典型命令：
    python match_new\\run_hardnet_matching.py

调试：在配置里设置 runtime.limit_identities / limit_images_per_identity，
或临时用命令行覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.evaluation import run_hardnet_evaluation, summary_row
from match_new.input_loader import load_raw_image_metadata, validate_identity_image_counts
from match_new.template_builder import build_hardnet_templates, build_identity_templates, load_image_template
from match_new.utils import ensure_dir, load_config, read_csv_rows, resolve_path, template_filename, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"
DEFAULT_OUTPUT_DIR = "outputs"

# 单张注册模板总时长的完整组成。这里不包含 template_total_ms、sift_ms 等
# 聚合/兼容字段，避免在按手指求和时重复计算同一阶段。
ENROLLMENT_STAGE_FIELDS = (
    "image_read_ms",
    "sift_keypoint_detection_ms",
    "keypoint_filter_ms",
    "patch_crop_rotate_ms",
    "hardnet_inference_ms",
    "template_assembly_ms",
    "template_pipeline_overhead_ms",
    "template_persist_ms",
    "template_registration_overhead_ms",
)


def numeric(value: Any) -> float | None:
    """尽可能把 CSV/JSON 中的值转换为浮点数；空值或非法值返回 ``None``。"""

    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    """用最近秩方式计算小规模耗时样本的百分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(q)))
    index = max(0, min(len(ordered) - 1, index))
    return float(ordered[index])


def summarize_values(values: list[float]) -> dict[str, Any]:
    """汇总一组毫秒耗时，输出总计、均值、范围及 P50/P95。"""

    if not values:
        return {"count": 0}
    total = float(sum(values))
    return {
        "count": len(values),
        "total_ms": total,
        "avg_ms": total / len(values),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def build_enrollment_timing_report(success_rows: list[dict[str, str]], identity_payload: dict[str, Any]) -> dict[str, Any]:
    """按手指汇总被选中注册图像的模板构建时间。

    单个手指的注册总时长等于其全部注册图像模板构建耗时之和；缺失的耗时会单独
    计数，避免把不完整数据误当作真实注册时长。除总时长外，还会累计图像读取、
    SIFT 检测、关键点筛选、局部块裁剪旋转、HardNet 推理和模板持久化等阶段。
    """

    row_by_key = {(row["identity_id"], row["image_id"]): row for row in success_rows}
    per_identity: list[dict[str, Any]] = []
    for identity in identity_payload.get("identities", []):
        identity_id = str(identity.get("identity_id", ""))
        image_ids = [str(image_id) for image_id in identity.get("template_image_ids", [])]
        values: list[float] = []
        missing = 0
        stage_values = {
            field: []
            for field in ENROLLMENT_STAGE_FIELDS
        }
        missing_stage_timing_values = 0
        for image_id in image_ids:
            row = row_by_key.get((identity_id, image_id))
            elapsed = numeric(row.get("template_build_ms") if row else None)
            if elapsed is None:
                missing += 1
            else:
                values.append(elapsed)
            for field in ENROLLMENT_STAGE_FIELDS:
                stage_elapsed = numeric(row.get(field) if row else None)
                if stage_elapsed is None:
                    missing_stage_timing_values += 1
                else:
                    stage_values[field].append(stage_elapsed)
        summary = summarize_values(values)
        stage_complete = bool(image_ids) and all(
            len(stage_values[field]) == len(image_ids)
            for field in ENROLLMENT_STAGE_FIELDS
        )
        stage_totals = {
            field: (
                float(sum(stage_values[field]))
                if len(stage_values[field]) == len(image_ids) and image_ids
                else ""
            )
            for field in ENROLLMENT_STAGE_FIELDS
        }
        accounted_stage_ms = (
            float(sum(float(value) for value in stage_totals.values()))
            if stage_complete
            else ""
        )
        registration_total_ms = summary.get("total_ms", "")
        per_identity.append(
            {
                "identity_id": identity_id,
                "num_enrollment_templates": len(image_ids),
                "num_timed_templates": len(values),
                "missing_timing_count": missing,
                "stage_timing_complete": int(stage_complete),
                "missing_stage_timing_values": missing_stage_timing_values,
                "registration_total_ms": registration_total_ms,
                "registration_avg_template_ms": summary.get("avg_ms", ""),
                "registration_min_template_ms": summary.get("min_ms", ""),
                "registration_max_template_ms": summary.get("max_ms", ""),
                "registration_p50_template_ms": summary.get("p50_ms", ""),
                "registration_p95_template_ms": summary.get("p95_ms", ""),
                **{
                    f"registration_{field}": value
                    for field, value in stage_totals.items()
                },
                "registration_accounted_stage_ms": accounted_stage_ms,
                "registration_accounting_error_ms": (
                    float(registration_total_ms) - float(accounted_stage_ms)
                    if registration_total_ms not in {"", None}
                    and accounted_stage_ms not in {"", None}
                    else ""
                ),
            }
        )
    totals = [float(row["registration_total_ms"]) for row in per_identity if row.get("registration_total_ms") not in {"", None}]
    return {"per_identity": per_identity, "summary": summarize_values(totals)}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认值都在配置文件中；这里只提供覆盖项。"""

    parser = argparse.ArgumentParser(
        description="运行 HardNet L2 手机指纹解锁离线评估；默认参数来自 --config 指定的 YAML。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径。")
    parser.add_argument("--image-root", "--image_root", dest="image_root", default=None, help="覆盖 data.image_root。")
    parser.add_argument(
        "--identity-depth",
        "--identity_depth",
        dest="identity_depth",
        type=int,
        default=None,
        help="覆盖 data.identity_depth。",
    )
    parser.add_argument("--model_path", default=None, help="覆盖 model.checkpoint。")
    parser.add_argument("--output_dir", default=None, help="覆盖 output.output_dir。")
    parser.add_argument(
        "--skip-template-build",
        "--skip_template_build",
        dest="skip_template_build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 runtime.skip_template_build。可用 --skip-template-build / --no-skip-template-build。",
    )
    parser.add_argument("--max_impostor_identities_per_query", type=int, default=None, help="覆盖 runtime.max_impostor_identities_per_query。")
    parser.add_argument("--random_seed", type=int, default=None, help="覆盖 enrollment.random_seed。")
    parser.add_argument("--target_far", type=float, default=None, help="覆盖 evaluation.target_far。")
    parser.add_argument("--target_frr", type=float, default=None, help="覆盖 evaluation.target_frr。")
    parser.add_argument("--max_failure_cases_per_type", type=int, default=None, help="覆盖 evaluation.failure_export.max_cases_per_type。")
    parser.add_argument(
        "--failure-export",
        "--failure_export",
        dest="failure_export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 evaluation.failure_export.enabled。可用 --failure-export / --no-failure-export。",
    )
    parser.add_argument("--limit_identities", type=int, default=None, help="覆盖 runtime.limit_identities。")
    parser.add_argument("--limit_images_per_identity", type=int, default=None, help="覆盖 runtime.limit_images_per_identity。")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """把命令行覆盖项写回 config。仅当命令行显式传入时覆盖。"""

    if args.image_root is not None:
        config.setdefault("data", {})["image_root"] = str(Path(args.image_root).expanduser().resolve())
    if args.identity_depth is not None:
        config.setdefault("data", {})["identity_depth"] = int(args.identity_depth)
    if args.model_path:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.model_path).expanduser().resolve())
    if args.output_dir is not None:
        # 命令行路径按当前工作目录理解，写成绝对路径避免歧义。
        config.setdefault("output", {})["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        config.setdefault("output", {})["_output_dir_from_cli"] = True
    if args.skip_template_build is not None:
        config.setdefault("runtime", {})["skip_template_build"] = bool(args.skip_template_build)
    if args.max_impostor_identities_per_query is not None:
        config.setdefault("runtime", {})["max_impostor_identities_per_query"] = int(args.max_impostor_identities_per_query)
    if args.random_seed is not None:
        config.setdefault("enrollment", {})["random_seed"] = int(args.random_seed)
    if args.target_far is not None:
        config.setdefault("evaluation", {})["target_far"] = float(args.target_far)
    if args.target_frr is not None:
        config.setdefault("evaluation", {})["target_frr"] = float(args.target_frr)
    if args.max_failure_cases_per_type is not None:
        config.setdefault("evaluation", {}).setdefault("failure_export", {})["max_cases_per_type"] = int(args.max_failure_cases_per_type)
    if args.failure_export is not None:
        config.setdefault("evaluation", {}).setdefault("failure_export", {})["enabled"] = bool(args.failure_export)
    if args.limit_identities is not None:
        config.setdefault("runtime", {})["limit_identities"] = int(args.limit_identities)
    if args.limit_images_per_identity is not None:
        config.setdefault("runtime", {})["limit_images_per_identity"] = int(args.limit_images_per_identity)


def resolve_run_settings(config: dict[str, Any]) -> dict[str, Any]:
    """从配置解析本次运行的路径与开关。"""

    output_cfg = dict(config.get("output", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    failure_cfg = dict(dict(config.get("evaluation", {})).get("failure_export", {}))
    raw_output = output_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    if bool(output_cfg.get("_output_dir_from_cli")):
        output_dir = Path(str(raw_output)).expanduser()
    else:
        output_dir = resolve_path(config, raw_output)
    return {
        "output_dir": output_dir,
        "skip_template_build": bool(runtime_cfg.get("skip_template_build", False)),
        "max_impostor_identities_per_query": int(runtime_cfg.get("max_impostor_identities_per_query", 0)),
        "limit_identities": int(runtime_cfg.get("limit_identities", 0)),
        "limit_images_per_identity": int(runtime_cfg.get("limit_images_per_identity", 0)),
        "export_failures": bool(failure_cfg.get("enabled", True)),
    }


def limit_rows_for_debug(rows: list[dict[str, str]], limit_identities: int, limit_images_per_identity: int) -> list[dict[str, str]]:
    """调试用数据裁剪。

    正式实验不要设置这两个参数；它们只用于快速 smoke test，
    例如 2 个 identity、每个 identity 22 张图，刚好能形成 20 张注册图和 2 张 query。
    """

    if limit_identities <= 0 and limit_images_per_identity <= 0:
        return rows
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["identity_id"]].append(row)
    selected: list[dict[str, str]] = []
    for identity_id in sorted(groups)[: limit_identities or None]:
        identity_rows = sorted(groups[identity_id], key=lambda item: item["image_id"])
        if limit_images_per_identity > 0:
            identity_rows = identity_rows[:limit_images_per_identity]
        selected.extend(identity_rows)
    return selected


def validate_templates(
    template_dir: str | Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    max_checks: int = 10,
) -> dict[str, Any]:
    """抽样检查模板字段是否和 keypoints 对齐。

    这里主要检查 HardNet descriptor 行数是否等于 keypoint 数，并确认维度为 128。
    如果模板构建阶段出现 keypoint/descriptor 错位，后续匹配指标会完全不可信。
    """

    require_overlap_image = bool(dict(config.get("texture_verification", {})).get("enabled", False))
    checked = 0
    for row in rows[:max_checks]:
        template = load_image_template(Path(template_dir) / template_filename(row["identity_id"], row["image_id"]), require="hardnet")
        n = int(template["keypoints_xy"].shape[0])
        hardnet = template["hardnet_descriptors"]
        if hardnet.shape != (n, 128):
            raise ValueError(f"HardNet shape mismatch for {row['image_id']}: {hardnet.shape} vs {(n, 128)}")
        if require_overlap_image and not bool(template.get("has_overlap_image", False)):
            raise ValueError(
                f"Template {row['image_id']} has no overlap_image required by texture verification. "
                "Rebuild templates without --skip-template-build."
            )
        checked += 1
    return {
        "checked_templates": checked,
        "patch_crop_size": 32,
        "patch_out_size": 32,
        "descriptor_type": "hardnet",
        "overlap_image_required": require_overlap_image,
    }


def main() -> None:
    """串起完整实验流程。"""

    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    settings = resolve_run_settings(config)
    checkpoint = resolve_path(config, dict(config.get("model", {}))["checkpoint"])
    if not checkpoint.exists() and not settings["skip_template_build"]:
        raise FileNotFoundError(f"HardNet checkpoint does not exist: {checkpoint}")

    # 1. 直接扫描原始图像；调试时可按 identity 数和每个 identity 的图像数裁剪数据。
    output_dir = ensure_dir(settings["output_dir"])
    rows = load_raw_image_metadata(config)
    rows = limit_rows_for_debug(rows, int(settings["limit_identities"]), int(settings["limit_images_per_identity"]))
    if not rows:
        raise RuntimeError("No raw fingerprint images found.")
    enrollment = dict(config.get("enrollment", {}))
    enrollment_count = int(enrollment.get("enrollment_images_per_identity", 20))
    validate_identity_image_counts(rows, enrollment_count + 1, context="raw image input")
    write_csv_rows(output_dir / "metadata_all.csv", rows)

    # 2. 构建图像级模板；如果 skip_template_build，则复用已有模板。
    template_dir = output_dir / "image_templates"
    metadata_success_path = output_dir / "metadata_success.csv"
    if settings["skip_template_build"]:
        candidate_rows = read_csv_rows(metadata_success_path) if metadata_success_path.exists() else rows
        raw_keys = {(row["identity_id"], row["image_id"]) for row in rows}
        success_rows = [
            row
            for row in candidate_rows
            if (row["identity_id"], row["image_id"]) in raw_keys
            and (template_dir / template_filename(row["identity_id"], row["image_id"])).exists()
        ]
        if not success_rows:
            raise RuntimeError("skip_template_build was set but no image templates were found.")
    else:
        report = build_hardnet_templates(rows, template_dir, config)
        write_json(output_dir / "build_report.json", {k: v for k, v in report.items() if k != "success_rows"})
        write_csv_rows(output_dir / "template_build_timings.csv", report.get("template_timings", []))
        success_rows = report["success_rows"]
        if not success_rows:
            raise RuntimeError("No templates were built successfully.")
    write_csv_rows(metadata_success_path, success_rows)
    validate_identity_image_counts(success_rows, enrollment_count + 1, context="successfully built templates")
    write_json(output_dir / "template_validation.json", validate_templates(template_dir, success_rows, config))

    # 3. 固定随机种子，为每个 identity 选择注册模板，其余作为 query。
    identity_templates_path = output_dir / f"identity_templates_{enrollment_count}.json"
    split_metadata_path = output_dir / f"metadata_with_split_{enrollment_count}.csv"
    identity_payload = build_identity_templates(
        success_rows,
        identity_templates_path,
        split_metadata_path,
        enrollment_count=enrollment_count,
        seed=int(enrollment.get("random_seed", 42)),
    )
    write_json(output_dir / f"identity_templates_{enrollment_count}_summary.json", {k: v for k, v in identity_payload.items() if k != "identities"})
    enrollment_timing = build_enrollment_timing_report(success_rows, identity_payload)
    write_csv_rows(output_dir / "enrollment_timing.csv", enrollment_timing["per_identity"])
    write_json(output_dir / "enrollment_timing.json", enrollment_timing)

    # 4. 执行身份验证评估，并在目标阈值下导出失败样本。
    result = run_hardnet_evaluation(
        metadata_path=split_metadata_path,
        identity_templates_path=identity_templates_path,
        template_dir=template_dir,
        config=config,
        output_dir=output_dir / "eval_hardnet_l2",
        max_impostor_identities_per_query=int(settings["max_impostor_identities_per_query"]),
        export_failures=bool(settings["export_failures"]),
    )
    # 5. 汇总一行 CSV/JSON，方便和其他实验横向比较。
    far_points = [float(point) for point in dict(config.get("evaluation", {})).get("far_points", [0.001, 0.0001])]
    summary = [summary_row(result["metrics"], far_points)]
    summary_csv = output_dir / "hardnet_l2_summary.csv"
    summary_json = output_dir / "hardnet_l2_summary.json"
    write_csv_rows(summary_csv, summary)
    write_json(
        summary_json,
        {
            "summary": summary,
            "metadata": str(split_metadata_path),
            "identity_templates": str(identity_templates_path),
            "image_templates": str(template_dir),
            "eval_dir": str(output_dir / "eval_hardnet_l2"),
            "effective_config": result["metrics"].get("effective_config"),
            "run_settings": {
                "output_dir": str(output_dir),
                "skip_template_build": settings["skip_template_build"],
                "max_impostor_identities_per_query": settings["max_impostor_identities_per_query"],
                "limit_identities": settings["limit_identities"],
                "limit_images_per_identity": settings["limit_images_per_identity"],
                "export_failures": settings["export_failures"],
            },
        },
    )
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "outputs": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
