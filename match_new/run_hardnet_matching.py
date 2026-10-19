"""HardNet float/L2 与 binary/Hamming 指纹离线验证主入口。

作用：
    1. 扫描原始指纹图像目录，构建每张图像的 `.npz` 模板；
    2. 每个 identity 随机选择注册模板；
    3. 用其余图像作为 query，分别对本人和非本人 identity 模板库打分；
    4. 输出 FAR/FRR 阈值曲线、配置阈值下的结果、EER 和 AUC；
    5. 在配置阈值下导出 false reject / false accept 的原图和拼接预览。

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardnet_train.model import (
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
)
from match_new.descriptor_contract import (
    BINARY_DESCRIPTOR_KIND,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    HAMMING_DISTANCE_METRIC,
    L2_DISTANCE_METRIC,
    PACKED_UINT8_STORAGE,
    descriptor_columns,
    require_supported_contract,
    resolve_descriptor_contract,
)
from match_new.evaluation import run_hardnet_evaluation
from match_new.hadamard_binarization import (
    HadamardBinarizer,
    hadamard_binarization_enabled,
)
from match_new.input_loader import load_raw_image_metadata, validate_identity_image_counts
from match_new.runtime import HARDNET_INFERENCE_STAGE_FIELDS
from match_new.template_builder import build_hardnet_templates, build_identity_templates, load_image_template
from match_new.utils import ensure_dir, load_config, resolve_path, template_filename, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"
DEFAULT_OUTPUT_DIR = "outputs"
OBSOLETE_ROOT_OUTPUT_FILENAMES = (
    "build_report.json",
    "enrollment_timing.csv",
    "enrollment_timing.json",
    "hardnet_l2_summary.csv",
    "hardnet_l2_summary.json",
    "hardnet_hamming_summary.csv",
    "hardnet_hamming_summary.json",
)

# 单张模板构建总时长的完整组成。这里不包含 template_total_ms、sift_ms 等
# 聚合/兼容字段，避免重复计算同一阶段。
TEMPLATE_BUILD_STAGE_FIELDS = (
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


def build_template_timing_average(
    template_timings: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总全部成功模板的构建阶段平均耗时，避免输出逐图明细。"""

    success_rows = [
        row
        for row in template_timings
        if str(row.get("status", "")).lower() == "success"
    ]
    summary: dict[str, Any] = {
        "num_input_images": len(template_timings),
        "num_successful_templates": len(success_rows),
        "num_failed_templates": len(template_timings) - len(success_rows),
    }
    average_fields = (
        "template_build_ms",
        *TEMPLATE_BUILD_STAGE_FIELDS,
        *HARDNET_INFERENCE_STAGE_FIELDS,
        "num_keypoints",
    )
    for field in average_fields:
        values = [
            value
            for row in success_rows
            if (value := numeric(row.get(field))) is not None
        ]
        output_field = (
            "avg_num_keypoints"
            if field == "num_keypoints"
            else f"avg_{field}"
        )
        summary[output_field] = (
            float(sum(values) / len(values))
            if values
            else ""
        )
    return summary


def remove_obsolete_root_outputs(output_dir: Path) -> None:
    """删除旧版本生成但当前已取消的重复输出文件。"""

    for filename in OBSOLETE_ROOT_OUTPUT_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)
    # 跳过模板构建时没有本次耗时，不能保留旧版逐图明细冒充当前结果。
    (output_dir / "template_build_timings.csv").unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认值都在配置文件中；这里只提供覆盖项。"""

    parser = argparse.ArgumentParser(
        description="运行 HardNet 浮点/L2 或二值/Hamming 手机指纹解锁离线评估；默认参数来自 --config 指定的 YAML。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径。")
    parser.add_argument("--image-root", "--image_root", dest="image_root", default=None, help="覆盖 data.image_root。")
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        default=None,
        help="覆盖 model.checkpoint。",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=None,
        help="覆盖 output.output_dir。",
    )
    parser.add_argument(
        "--hadamard-binarization",
        "--hadamard_binarization",
        dest="hadamard_binarization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "覆盖 model.hadamard_binarization.enabled。可用 "
            "--hadamard-binarization / --no-hadamard-binarization。"
        ),
    )
    parser.add_argument(
        "--ratio-threshold",
        "--ratio_threshold",
        dest="ratio_threshold",
        type=float,
        default=None,
        help="覆盖 matching.ratio_threshold（浮点/L2）。",
    )
    parser.add_argument(
        "--hamming-ratio-threshold",
        "--hamming_ratio_threshold",
        dest="hamming_ratio_threshold",
        type=float,
        default=None,
        help="覆盖 matching.hamming.ratio_threshold（二值/Hamming）。",
    )
    parser.add_argument(
        "--skip-template-build",
        "--skip_template_build",
        dest="skip_template_build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 runtime.skip_template_build。可用 --skip-template-build / --no-skip-template-build。",
    )
    parser.add_argument(
        "--max_impostor_identities_per_query",
        type=int,
        default=None,
        help="覆盖 runtime.max_impostor_identities_per_query（-1=全量，0=不比对非本人）。",
    )
    parser.add_argument("--random_seed", type=int, default=None, help="覆盖 enrollment.random_seed。")
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
    if args.model_path:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.model_path).expanduser().resolve())
    if args.output_dir is not None:
        # 命令行路径按当前工作目录理解，写成绝对路径避免歧义。
        config.setdefault("output", {})["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        config.setdefault("output", {})["_output_dir_from_cli"] = True
    if args.hadamard_binarization is not None:
        config.setdefault("model", {}).setdefault("hadamard_binarization", {})[
            "enabled"
        ] = bool(args.hadamard_binarization)
    if args.ratio_threshold is not None:
        config.setdefault("matching", {})["ratio_threshold"] = float(
            args.ratio_threshold
        )
    if args.hamming_ratio_threshold is not None:
        config.setdefault("matching", {}).setdefault("hamming", {})[
            "ratio_threshold"
        ] = float(args.hamming_ratio_threshold)
    if args.skip_template_build is not None:
        config.setdefault("runtime", {})["skip_template_build"] = bool(args.skip_template_build)
    if args.max_impostor_identities_per_query is not None:
        config.setdefault("runtime", {})["max_impostor_identities_per_query"] = int(args.max_impostor_identities_per_query)
    if args.random_seed is not None:
        config.setdefault("enrollment", {})["random_seed"] = int(args.random_seed)
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
        "max_impostor_identities_per_query": int(runtime_cfg.get("max_impostor_identities_per_query", -1)),
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


def checkpoint_descriptor_metadata(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """解析 checkpoint 及可选 Hadamard 后处理的输出描述子元数据。"""

    kind = str(checkpoint.get("descriptor_kind", FLOAT_DESCRIPTOR_KIND)).strip().lower()
    metric = str(checkpoint.get("descriptor_metric", L2_DISTANCE_METRIC)).strip().lower()
    if (kind, metric) not in {
        (FLOAT_DESCRIPTOR_KIND, L2_DISTANCE_METRIC),
        (BINARY_DESCRIPTOR_KIND, HAMMING_DISTANCE_METRIC),
    }:
        raise ValueError(
            f"Unsupported checkpoint descriptor contract: kind={kind}, metric={metric}."
        )
    dimension = checkpoint_descriptor_dim(checkpoint)
    if kind == BINARY_DESCRIPTOR_KIND:
        architecture = str(
            checkpoint.get("model_architecture", "residual_binary_hash_v1")
        )
        storage = str(
            checkpoint.get("binary_storage", PACKED_UINT8_STORAGE)
        ).strip().lower()
        bitorder = str(checkpoint.get("binary_bitorder", "little")).strip().lower()
    else:
        architecture = checkpoint_model_architecture(checkpoint)
        storage = FLOAT32_STORAGE
        bitorder = ""
    metadata = {
        "model_architecture": architecture,
        "descriptor_kind": kind,
        "descriptor_metric": metric,
        "descriptor_dim": dimension,
        "descriptor_storage": storage,
        "descriptor_bitorder": bitorder,
        "descriptor_transform_name": "",
        "descriptor_transform_id": "",
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_parameter_count": checkpoint.get("model_parameter_count"),
        "model_macs_per_patch": checkpoint.get("model_macs_per_patch"),
    }
    if hadamard_binarization_enabled(config):
        if (kind, metric) != (FLOAT_DESCRIPTOR_KIND, L2_DISTANCE_METRIC):
            raise ValueError(
                "Hadamard post-binarization requires a float/L2 checkpoint; "
                f"got kind={kind}, metric={metric}."
            )
        binarizer = HadamardBinarizer.from_config(config)
        if dimension != binarizer.order:
            raise ValueError(
                "Hadamard order must equal the checkpoint descriptor dimension: "
                f"order={binarizer.order}, descriptor_dim={dimension}."
            )
        metadata.update(
            {
                "source_descriptor_kind": kind,
                "source_descriptor_metric": metric,
                "source_descriptor_dim": dimension,
                "descriptor_kind": BINARY_DESCRIPTOR_KIND,
                "descriptor_metric": HAMMING_DISTANCE_METRIC,
                "descriptor_dim": binarizer.order,
                "descriptor_storage": PACKED_UINT8_STORAGE,
                "descriptor_bitorder": binarizer.bitorder,
                **binarizer.metadata(),
            }
        )
    return metadata


def validate_configured_descriptor(
    config: dict[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """将配置中的显式描述子选项作为 checkpoint 契约断言。"""

    model_cfg = dict(config.get("model", {}))
    matching_cfg = dict(config.get("matching", {}))
    checks = {
        "model.descriptor_kind": (
            model_cfg.get("descriptor_kind", "auto"),
            metadata["descriptor_kind"],
        ),
        "matching.distance": (
            matching_cfg.get("distance", "auto"),
            metadata["descriptor_metric"],
        ),
    }
    for field, (configured, actual) in checks.items():
        value = str(configured if configured is not None else "").strip().lower()
        if value == "euclidean":
            value = L2_DISTANCE_METRIC
        if value not in {"", "auto", str(actual).lower()}:
            raise ValueError(
                f"{field} does not match checkpoint: config={value}, checkpoint={actual}."
            )
    configured_dim = str(
        model_cfg.get("descriptor_dim", "auto")
    ).strip().lower()
    if configured_dim not in {"", "auto"} and int(configured_dim) != int(
        metadata["descriptor_dim"]
    ):
        raise ValueError(
            "model.descriptor_dim does not match checkpoint: "
            f"config={configured_dim}, checkpoint={metadata['descriptor_dim']}."
        )
    if metadata["descriptor_kind"] == BINARY_DESCRIPTOR_KIND:
        configured_storage = str(
            model_cfg.get("binary_storage", "auto")
        ).strip().lower()
        configured_bitorder = str(
            model_cfg.get("binary_bitorder", "auto")
        ).strip().lower()
        if configured_storage not in {
            "",
            "auto",
            str(metadata["descriptor_storage"]).lower(),
        }:
            raise ValueError(
                "model.binary_storage does not match checkpoint: "
                f"config={configured_storage}, checkpoint={metadata['descriptor_storage']}."
            )
        if configured_bitorder not in {
            "",
            "auto",
            str(metadata["descriptor_bitorder"]).lower(),
        }:
            raise ValueError(
                "model.binary_bitorder does not match checkpoint: "
                f"config={configured_bitorder}, checkpoint={metadata['descriptor_bitorder']}."
            )


def resolve_expected_descriptor_dim(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> int:
    """从 checkpoint 解析描述子维度并校验显式配置。"""

    target = Path(checkpoint_path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Descriptor checkpoint does not exist: {target}")
    checkpoint = torch.load(target, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {target}")
    metadata = checkpoint_descriptor_metadata(checkpoint, config)
    validate_configured_descriptor(config, metadata)
    return int(metadata["descriptor_dim"])


def resolve_checkpoint_metadata(
    checkpoint_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """读取 checkpoint 和可选后处理对应的统一输出元数据。"""

    target = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(target, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {target}")
    return checkpoint_descriptor_metadata(checkpoint, config)


def validate_templates(
    template_dir: str | Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    descriptor_metadata: Mapping[str, Any],
    max_checks: int = 10,
) -> dict[str, Any]:
    """抽样检查模板字段是否与 checkpoint 描述子契约一致。"""

    require_overlap_image = bool(dict(config.get("texture_verification", {})).get("enabled", False))
    checked = 0
    for row in rows[:max_checks]:
        template = load_image_template(Path(template_dir) / template_filename(row["identity_id"], row["image_id"]), require="hardnet")
        n = int(template["keypoints_xy"].shape[0])
        hardnet = template["hardnet_descriptors"]
        contract = resolve_descriptor_contract(template, "hardnet")
        require_supported_contract(
            contract,
            label=str(template.get("template_path", row["image_id"])),
        )
        actual = {
            "descriptor_kind": contract.kind,
            "descriptor_metric": contract.metric,
            "descriptor_dim": contract.dimension,
            "descriptor_storage": contract.storage,
            "descriptor_bitorder": contract.bitorder,
            "descriptor_transform_name": contract.transform_name,
            "descriptor_transform_id": contract.transform_id,
        }
        expected = {
            field: descriptor_metadata[field]
            for field in actual
        }
        if actual != expected:
            raise ValueError(
                f"Template/checkpoint descriptor mismatch for {row['image_id']}: "
                f"template={actual}, checkpoint={expected}. Rebuild templates."
            )
        expected_shape = (n, descriptor_columns(contract))
        if hardnet.shape != expected_shape:
            raise ValueError(
                f"HardNet shape mismatch for {row['image_id']}: "
                f"{hardnet.shape} vs {expected_shape}"
            )
        if require_overlap_image and not bool(template.get("has_overlap_image", False)):
            raise ValueError(
                f"Template {row['image_id']} has no overlap_image required by texture verification. "
                "Rebuild templates without --skip-template-build."
            )
        checked += 1
    patch_cfg = dict(config.get("patch", {}))
    data_cfg = dict(config.get("data", {}))
    image_root = resolve_path(config, data_cfg.get("image_root", ""))
    return {
        "checked_templates": checked,
        "patch_crop_size": int(patch_cfg.get("crop_size", 32)),
        "patch_out_size": int(patch_cfg.get("out_size", 32)),
        "descriptor_type": "hardnet",
        **{
            field: descriptor_metadata[field]
            for field in (
                "descriptor_kind",
                "descriptor_metric",
                "descriptor_dim",
                "descriptor_storage",
                "descriptor_bitorder",
                "descriptor_transform_name",
                "descriptor_transform_id",
            )
        },
        "evaluation_dataset": image_root.name,
        "image_root": str(image_root),
        "overlap_image_required": require_overlap_image,
    }


def main() -> None:
    """串起完整实验流程。"""

    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    settings = resolve_run_settings(config)
    checkpoint = resolve_path(config, dict(config.get("model", {}))["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Descriptor checkpoint does not exist: {checkpoint}")
    resolve_expected_descriptor_dim(config, checkpoint)
    checkpoint_metadata = resolve_checkpoint_metadata(checkpoint, config)
    descriptor_metric = str(checkpoint_metadata["descriptor_metric"])
    config.setdefault("model", {})["descriptor_kind"] = str(
        checkpoint_metadata["descriptor_kind"]
    )
    config.setdefault("matching", {})["distance"] = descriptor_metric
    eval_dir_name = f"eval_hardnet_{descriptor_metric}"

    # 1. 直接扫描原始图像；调试时可按 identity 数和每个 identity 的图像数裁剪数据。
    output_dir = ensure_dir(settings["output_dir"])
    remove_obsolete_root_outputs(output_dir)
    rows = load_raw_image_metadata(config)
    rows = limit_rows_for_debug(rows, int(settings["limit_identities"]), int(settings["limit_images_per_identity"]))
    if not rows:
        raise RuntimeError("No raw fingerprint images found.")
    enrollment = dict(config.get("enrollment", {}))
    enrollment_count = int(enrollment.get("enrollment_images_per_identity", 20))
    rows = validate_identity_image_counts(
        rows, enrollment_count + 1, context="原始图像扫描"
    )
    write_csv_rows(output_dir / "metadata_all.csv", rows)

    # 2. 构建图像级模板；如果 skip_template_build，则按原始 metadata 复用已有模板。
    template_dir = output_dir / "image_templates"
    if settings["skip_template_build"]:
        # No inference was initialized in this run; do not retain stale startup timing.
        (output_dir / "inference_startup.json").unlink(missing_ok=True)
        success_rows = []
        for candidate in rows:
            key = (candidate["identity_id"], candidate["image_id"])
            template_path = template_dir / template_filename(*key)
            if not template_path.exists():
                continue
            row = dict(candidate)
            row["template_path"] = str(template_path)
            row["status"] = "success"
            success_rows.append(row)
        if not success_rows:
            raise RuntimeError("skip_template_build was set but no image templates were found.")
    else:
        report = build_hardnet_templates(rows, template_dir, config)
        write_json(output_dir / "inference_startup.json", report["inference_settings"])
        timing_average = build_template_timing_average(
            report.get("template_timings", [])
        )
        write_csv_rows(
            output_dir / "template_build_timings.csv",
            [timing_average],
        )
        success_rows = report["success_rows"]
        if not success_rows:
            raise RuntimeError("No templates were built successfully.")
    validate_templates(
        template_dir,
        success_rows,
        config,
        descriptor_metadata=checkpoint_metadata,
    )

    # 3. 按策略为每个 identity 选择注册模板，其余作为 query。
    identity_templates_path = output_dir / f"identity_templates_{enrollment_count}.json"
    _identity_payload, split_rows = build_identity_templates(
        success_rows,
        identity_templates_path,
        enrollment_count=enrollment_count,
        seed=int(enrollment.get("random_seed", 42)),
        selection_strategy=str(enrollment.get("selection_strategy", "first")),
    )

    # 4. 执行身份验证评估，并在配置阈值下导出失败样本。
    eval_dir = output_dir / eval_dir_name
    result = run_hardnet_evaluation(
        metadata_rows=split_rows,
        identity_templates_path=identity_templates_path,
        template_dir=template_dir,
        config=config,
        output_dir=eval_dir,
        max_impostor_identities_per_query=int(settings["max_impostor_identities_per_query"]),
        export_failures=bool(settings["export_failures"]),
    )
    print(
        json.dumps(
            {
                "verification_scores_csv": result["scores_path"],
                "metrics_json": str(eval_dir / "metrics.json"),
                "effective_config_yaml": str(
                    eval_dir / "effective_config.yaml"
                ),
                "far_frr_table_csv": result["far_frr_table_path"],
                "per_finger_far_frr_csv": result["per_finger_far_frr_path"],
                "outputs": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
