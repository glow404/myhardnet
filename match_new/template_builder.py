"""match_new HardNet 图像模板构建与注册模板划分。

主实验只保存：
    - SIFT 关键点位置/尺度/方向/响应
    - HardNet 描述子
"""

from __future__ import annotations

import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.descriptor_contract import (
    BINARY_DESCRIPTOR_KIND,
    DESCRIPTOR_SCHEMA_VERSION,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    L2_DISTANCE_METRIC,
    descriptor_columns,
    require_supported_contract,
    resolve_descriptor_contract,
)
from match_new.runtime import (
    HARDNET_INFERENCE_STAGE_FIELDS,
    HardNetDescriptor,
    build_sift,
    detect_sift_keypoints,
    imread_grayscale,
    patchable_keypoints,
)

from .utils import ensure_dir, template_filename, write_json


LOGGER = logging.getLogger(__name__)


def detect_keypoints_only(image: np.ndarray, config: dict[str, Any], sift: cv2.SIFT) -> list[cv2.KeyPoint]:
    """HardNet 模板构建用：只检测关键点，不计算 SIFT 描述子。"""

    return detect_sift_keypoints(image, sift, config)


def extract_keypoint_patches(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    crop_size: int,
    out_size: int,
    min_overlap_ratio: float,
    normalize: bool,
) -> tuple[list[cv2.KeyPoint], np.ndarray, list[int]]:
    """兼容旧模板构建代码的 patch 裁剪入口。"""

    return patchable_keypoints(
        image,
        keypoints,
        {
            "patch": {
                "crop_size": crop_size,
                "out_size": out_size,
                "min_overlap_ratio": min_overlap_ratio,
                "normalize": normalize,
            }
        },
    )


def hardnet_template_payload(
    row: dict[str, str],
    keypoints: list[cv2.KeyPoint],
    hardnet_descriptors: np.ndarray,
    overlap_image: np.ndarray,
    *,
    descriptor_kind: str = FLOAT_DESCRIPTOR_KIND,
    descriptor_metric: str = L2_DISTANCE_METRIC,
    descriptor_dim: int | None = None,
    descriptor_storage: str = FLOAT32_STORAGE,
    descriptor_bitorder: str = "",
    descriptor_transform_name: str = "",
    descriptor_transform_id: str = "",
) -> dict[str, Any]:
    """构造离线注册/评估共用的内存模板结构。"""

    xy = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    gray = np.asarray(overlap_image)
    if gray.ndim != 2:
        raise ValueError(f"overlap_image must be a 2D grayscale image, got {gray.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    kind = str(descriptor_kind).strip().lower()
    descriptors = np.asarray(
        hardnet_descriptors,
        dtype=np.uint8 if kind == BINARY_DESCRIPTOR_KIND else np.float32,
    )
    if descriptors.ndim != 2:
        raise ValueError(
            f"hardnet_descriptors must be [N,D], got {descriptors.shape}"
        )
    if descriptors.shape[0] != len(keypoints):
        raise ValueError(
            "HardNet descriptor/keypoint row mismatch: "
            f"descriptors={descriptors.shape[0]}, keypoints={len(keypoints)}"
        )
    dimension = int(
        descriptor_dim if descriptor_dim is not None else descriptors.shape[1]
    )
    if dimension <= 0:
        raise ValueError(f"HardNet descriptor dimension must be positive, got {dimension}")
    template = {
        "template_format_version": 6,
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "identity_id": str(row["identity_id"]),
        "image_id": str(row["image_id"]),
        "image_path": str(row["image_path"]),
        "keypoints_xy": xy,
        "keypoints_size": np.asarray([kp.size for kp in keypoints], dtype=np.float32),
        "keypoints_angle": np.asarray(
            [kp.angle if kp.angle >= 0 else 0.0 for kp in keypoints],
            dtype=np.float32,
        ),
        "keypoints_response": np.asarray([kp.response for kp in keypoints], dtype=np.float32),
        "hardnet_descriptors": descriptors,
        "hardnet_descriptor_dim": dimension,
        "hardnet_descriptor_kind": kind,
        "hardnet_descriptor_metric": str(descriptor_metric).strip().lower(),
        "hardnet_descriptor_storage": str(descriptor_storage).strip().lower(),
        "hardnet_descriptor_bitorder": str(descriptor_bitorder).strip().lower(),
        "hardnet_descriptor_transform_name": str(descriptor_transform_name).strip(),
        "hardnet_descriptor_transform_id": str(descriptor_transform_id).strip(),
        "sift_descriptors": np.zeros((0, 128), dtype=np.float32),
        "overlap_image": gray,
        "template_path": "",
        "has_hardnet": True,
        "has_sift": False,
        "has_overlap_image": True,
    }
    contract = resolve_descriptor_contract(template, "hardnet")
    require_supported_contract(contract, label=str(row.get("image_id", "<memory>")))
    expected_shape = (len(keypoints), descriptor_columns(contract))
    if descriptors.shape != expected_shape:
        raise ValueError(
            "HardNet descriptor shape does not match its contract: "
            f"descriptors={descriptors.shape}, expected={expected_shape}."
        )
    return template


def save_hardnet_template_payload(output_path: str | Path, template: dict[str, Any]) -> Path:
    """把一份内存 HardNet 模板压缩保存为 ``.npz`` 注册产物。"""

    target = Path(output_path).expanduser()
    ensure_dir(target.parent)
    contract = resolve_descriptor_contract(template, "hardnet")
    require_supported_contract(contract, label=str(target))
    descriptors = np.asarray(
        template["hardnet_descriptors"],
        dtype=np.uint8 if contract.kind == BINARY_DESCRIPTOR_KIND else np.float32,
    )
    expected_columns = descriptor_columns(contract)
    if descriptors.ndim != 2 or descriptors.shape[1] != expected_columns:
        raise ValueError(
            "HardNet template descriptor contract mismatch before save: "
            f"path={target}, descriptors={descriptors.shape}, "
            f"dimension={contract.dimension}, storage={contract.storage}."
        )
    np.savez_compressed(
        target,
        template_format_version=np.asarray(
            int(template.get("template_format_version", 4)),
            dtype=np.uint8,
        ),
        descriptor_schema_version=np.asarray(
            int(template.get("descriptor_schema_version", DESCRIPTOR_SCHEMA_VERSION)),
            dtype=np.uint8,
        ),
        identity_id=np.asarray(template["identity_id"]),
        image_id=np.asarray(template["image_id"]),
        image_path=np.asarray(template["image_path"]),
        keypoints_xy=np.asarray(template["keypoints_xy"], dtype=np.float32),
        keypoints_size=np.asarray(template["keypoints_size"], dtype=np.float32),
        keypoints_angle=np.asarray(template["keypoints_angle"], dtype=np.float32),
        keypoints_response=np.asarray(template["keypoints_response"], dtype=np.float32),
        hardnet_descriptors=descriptors,
        hardnet_descriptor_dim=np.asarray(
            int(template["hardnet_descriptor_dim"]),
            dtype=np.int32,
        ),
        hardnet_descriptor_kind=np.asarray(
            str(template.get("hardnet_descriptor_kind", FLOAT_DESCRIPTOR_KIND))
        ),
        hardnet_descriptor_metric=np.asarray(
            str(template.get("hardnet_descriptor_metric", L2_DISTANCE_METRIC))
        ),
        hardnet_descriptor_storage=np.asarray(
            str(template.get("hardnet_descriptor_storage", FLOAT32_STORAGE))
        ),
        hardnet_descriptor_bitorder=np.asarray(
            str(template.get("hardnet_descriptor_bitorder", ""))
        ),
        hardnet_descriptor_transform_name=np.asarray(
            str(template.get("hardnet_descriptor_transform_name", ""))
        ),
        hardnet_descriptor_transform_id=np.asarray(
            str(template.get("hardnet_descriptor_transform_id", ""))
        ),
        overlap_image=np.asarray(template["overlap_image"], dtype=np.uint8),
    )
    template["template_path"] = str(target)
    return target


def save_hardnet_template(
    output_path: str | Path,
    row: dict[str, str],
    keypoints: list[cv2.KeyPoint],
    hardnet_descriptors: np.ndarray,
    overlap_image: np.ndarray,
) -> Path:
    """兼容原有调用方式：先构造统一模板结构，再将其保存到磁盘。"""

    return save_hardnet_template_payload(
        output_path,
        hardnet_template_payload(row, keypoints, hardnet_descriptors, overlap_image),
    )


def build_hardnet_template_from_image(
    row: dict[str, str],
    hardnet: HardNetDescriptor,
    sift: cv2.SIFT,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """从一张原始图像构建查询或注册模板，并记录各阶段耗时。

    返回值中的第一项是可直接参与匹配的内存模板；第二项包含图像读取、SIFT
    关键点检测、局部块提取、HardNet 描述子推理及总耗时。注册流程会进一步
    持久化模板，内存中的 query 模板可供离线评估流程直接匹配。
    """

    total_started = time.perf_counter()
    started = time.perf_counter()
    image = imread_grayscale(row["image_path"])
    image_read_ms = (time.perf_counter() - started) * 1000.0
    if image is None:
        raise FileNotFoundError(row["image_path"])

    started = time.perf_counter()
    keypoints = detect_keypoints_only(image, config, sift=sift)
    sift_keypoint_detection_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    max_keypoints = int(dict(config.get("keypoint_filter", {})).get("max_keypoints", 400))
    if max_keypoints > 0 and len(keypoints) > max_keypoints:
        order = np.argsort([kp.response for kp in keypoints])[::-1][:max_keypoints]
        keypoints = [keypoints[int(index)] for index in order]
    keypoint_filter_ms = (time.perf_counter() - started) * 1000.0
    sift_ms = sift_keypoint_detection_ms + keypoint_filter_ms

    patch_cfg = dict(config.get("patch", {}))
    started = time.perf_counter()
    selected_keypoints, patches, _selected_indices = extract_keypoint_patches(
        image,
        keypoints,
        crop_size=int(patch_cfg.get("crop_size", 32)),
        out_size=int(patch_cfg.get("out_size", 32)),
        min_overlap_ratio=float(patch_cfg.get("min_overlap_ratio", 0.55)),
        normalize=bool(patch_cfg.get("normalize", True)),
    )
    patch_extract_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    hardnet_descriptors = hardnet.describe(patches)
    hardnet_ms = (time.perf_counter() - started) * 1000.0
    inference_details = getattr(hardnet, "last_timings", {})
    inference_stages = {
        field: float(inference_details.get(field, 0.0))
        for field in HARDNET_INFERENCE_STAGE_FIELDS
    }
    if inference_details:
        # Include the describe() wrapper in the same total, without double counting.
        inference_stages["hardnet_inference_overhead_ms"] = max(
            0.0,
            hardnet_ms - sum(
                value for field, value in inference_stages.items()
                if field != "hardnet_inference_overhead_ms"
            ),
        )
    if hardnet_descriptors.shape[0] != len(selected_keypoints):
        raise ValueError("descriptor/keypoint alignment failed")

    started = time.perf_counter()
    template = hardnet_template_payload(
        row,
        selected_keypoints,
        hardnet_descriptors,
        image,
        descriptor_kind=hardnet.descriptor_kind,
        descriptor_metric=hardnet.descriptor_metric,
        descriptor_dim=hardnet.descriptor_dim,
        descriptor_storage=hardnet.descriptor_storage,
        descriptor_bitorder=hardnet.descriptor_bitorder,
        descriptor_transform_name=hardnet.descriptor_transform_name,
        descriptor_transform_id=hardnet.descriptor_transform_id,
    )
    template_assembly_ms = (time.perf_counter() - started) * 1000.0
    template_total_ms = (time.perf_counter() - total_started) * 1000.0
    accounted_ms = (
        image_read_ms
        + sift_keypoint_detection_ms
        + keypoint_filter_ms
        + patch_extract_ms
        + hardnet_ms
        + template_assembly_ms
    )
    timings = {
        "image_read_ms": image_read_ms,
        "sift_keypoint_detection_ms": sift_keypoint_detection_ms,
        "keypoint_filter_ms": keypoint_filter_ms,
        "sift_ms": sift_ms,
        "patch_crop_rotate_ms": patch_extract_ms,
        "patch_extract_ms": patch_extract_ms,
        "hardnet_inference_ms": hardnet_ms,
        "hardnet_ms": hardnet_ms,
        **inference_stages,
        "hardnet_inference_profiled": bool(inference_details),
        "template_assembly_ms": template_assembly_ms,
        "template_pipeline_overhead_ms": max(
            0.0,
            template_total_ms - accounted_ms,
        ),
        "template_total_ms": template_total_ms,
        "num_keypoints": len(selected_keypoints),
    }
    return template, timings


def load_image_template(path: str | Path, *, require: str | None = None) -> dict[str, Any]:
    """读取单描述子图像模板。

    `require`:
        - ``"hardnet"``：必须含 `hardnet_descriptors`
        - ``"sift"``：必须含 `sift_descriptors`
        - ``None``：按文件中实际存在的描述子字段读取，但两者不能同时存在
    """

    target = Path(path).expanduser()
    with np.load(target, allow_pickle=True) as data:
        keys = set(data.files)
        has_hardnet = "hardnet_descriptors" in keys
        has_sift = "sift_descriptors" in keys
        if has_hardnet and has_sift:
            raise ValueError(
                f"拒绝读取双描述子模板: {target}. "
                "请用 build_hardnet_templates 重新生成 HardNet 单描述子模板。"
            )
        if require == "hardnet" and not has_hardnet:
            raise ValueError(f"HardNet 模板缺少 hardnet_descriptors: {target}")
        if require == "sift" and not has_sift:
            raise ValueError(f"RootSIFT 模板缺少 sift_descriptors: {target}")
        if require is None and not has_hardnet and not has_sift:
            raise ValueError(f"模板缺少描述子字段: {target}")

        n_xy = int(np.asarray(data["keypoints_xy"]).reshape(-1, 2).shape[0]) if "keypoints_xy" in keys else 0
        saved_kind = (
            str(np.asarray(data["hardnet_descriptor_kind"]).item()).strip().lower()
            if "hardnet_descriptor_kind" in keys
            else FLOAT_DESCRIPTOR_KIND
        )
        hardnet = (
            np.asarray(
                data["hardnet_descriptors"],
                dtype=(
                    np.uint8
                    if saved_kind == BINARY_DESCRIPTOR_KIND
                    else np.float32
                ),
            )
            if has_hardnet
            else np.zeros((0, 0), dtype=np.float32)
        )
        hardnet_contract_source: dict[str, Any] = {
            "hardnet_descriptors": hardnet,
            "template_path": str(target),
        }
        for field in (
            "hardnet_descriptor_dim",
            "hardnet_descriptor_kind",
            "hardnet_descriptor_metric",
            "hardnet_descriptor_storage",
            "hardnet_descriptor_bitorder",
            "hardnet_descriptor_transform_name",
            "hardnet_descriptor_transform_id",
        ):
            if field in keys:
                hardnet_contract_source[field] = np.asarray(data[field]).item()
        if has_hardnet:
            hardnet_contract = resolve_descriptor_contract(
                hardnet_contract_source,
                "hardnet",
            )
            require_supported_contract(hardnet_contract, label=str(target))
            hardnet_descriptor_dim = hardnet_contract.dimension
            hardnet_descriptor_kind = hardnet_contract.kind
            hardnet_descriptor_metric = hardnet_contract.metric
            hardnet_descriptor_storage = hardnet_contract.storage
            hardnet_descriptor_bitorder = hardnet_contract.bitorder
            hardnet_descriptor_transform_name = hardnet_contract.transform_name
            hardnet_descriptor_transform_id = hardnet_contract.transform_id
            expected_descriptor_columns = descriptor_columns(hardnet_contract)
        else:
            hardnet_descriptor_dim = 0
            hardnet_descriptor_kind = FLOAT_DESCRIPTOR_KIND
            hardnet_descriptor_metric = L2_DISTANCE_METRIC
            hardnet_descriptor_storage = FLOAT32_STORAGE
            hardnet_descriptor_bitorder = ""
            hardnet_descriptor_transform_name = ""
            hardnet_descriptor_transform_id = ""
            expected_descriptor_columns = 0
        if has_hardnet and (
            hardnet.ndim != 2
            or hardnet.shape[0] != n_xy
            or hardnet.shape[1] != expected_descriptor_columns
        ):
            raise ValueError(
                "HardNet 模板描述子/关键点/元数据不一致: "
                f"{target}: descriptors={hardnet.shape}, keypoints={n_xy}, "
                f"descriptor_dim={hardnet_descriptor_dim}, "
                f"storage={hardnet_descriptor_storage}"
            )
        sift = (
            np.asarray(data["sift_descriptors"], dtype=np.float32)
            if has_sift
            else np.zeros((0, 128), dtype=np.float32)
        )
        return {
            "template_format_version": int(np.asarray(data["template_format_version"]).item()) if "template_format_version" in keys else 1,
            "descriptor_schema_version": int(np.asarray(data["descriptor_schema_version"]).item()) if "descriptor_schema_version" in keys else 0,
            "identity_id": str(np.asarray(data["identity_id"]).item()),
            "image_id": str(np.asarray(data["image_id"]).item()),
            "image_path": str(np.asarray(data["image_path"]).item()),
            "keypoints_xy": np.asarray(data["keypoints_xy"], dtype=np.float32) if "keypoints_xy" in keys else np.zeros((0, 2), dtype=np.float32),
            "keypoints_size": np.asarray(data["keypoints_size"], dtype=np.float32) if "keypoints_size" in keys else np.zeros((n_xy,), dtype=np.float32),
            "keypoints_angle": np.asarray(data["keypoints_angle"], dtype=np.float32) if "keypoints_angle" in keys else np.zeros((n_xy,), dtype=np.float32),
            "keypoints_response": np.asarray(data["keypoints_response"], dtype=np.float32) if "keypoints_response" in keys else np.zeros((n_xy,), dtype=np.float32),
            "hardnet_descriptors": hardnet,
            "hardnet_descriptor_dim": hardnet_descriptor_dim,
            "hardnet_descriptor_kind": hardnet_descriptor_kind,
            "hardnet_descriptor_metric": hardnet_descriptor_metric,
            "hardnet_descriptor_storage": hardnet_descriptor_storage,
            "hardnet_descriptor_bitorder": hardnet_descriptor_bitorder,
            "hardnet_descriptor_transform_name": hardnet_descriptor_transform_name,
            "hardnet_descriptor_transform_id": hardnet_descriptor_transform_id,
            "sift_descriptors": sift,
            "overlap_image": np.asarray(data["overlap_image"], dtype=np.uint8) if "overlap_image" in keys else np.zeros((0, 0), dtype=np.uint8),
            "template_path": str(target),
            "has_hardnet": has_hardnet,
            "has_sift": has_sift,
            "has_overlap_image": "overlap_image" in keys,
        }


def build_hardnet_templates(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """批量构建 HardNet 图像模板。"""

    out = ensure_dir(output_dir)
    hardnet = HardNetDescriptor(config)
    sift = build_sift(config)
    successes: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    timings: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="build hardnet templates"):
        started = time.perf_counter()
        try:
            template, stage_timings = build_hardnet_template_from_image(row, hardnet, sift, config)
            path = out / template_filename(row["identity_id"], row["image_id"])
            persist_started = time.perf_counter()
            save_hardnet_template_payload(path, template)
            template_persist_ms = (
                time.perf_counter() - persist_started
            ) * 1000.0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            # 注册模板的外层总耗时还包括函数调用、路径构造等少量开销。将它单独
            # 列出后，各阶段之和可以与 template_build_ms 完整对账。
            template_registration_overhead_ms = max(
                0.0,
                elapsed_ms
                - float(stage_timings["template_total_ms"])
                - template_persist_ms,
            )
            timing = {
                "identity_id": row["identity_id"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "template_path": str(path),
                "template_build_ms": elapsed_ms,
                **stage_timings,
                "template_persist_ms": template_persist_ms,
                "template_registration_overhead_ms": (
                    template_registration_overhead_ms
                ),
                "status": "success",
            }
            timings.append(timing)
            successes.append(
                {
                    **row,
                    "template_path": str(path),
                    "template_build_ms": f"{elapsed_ms:.3f}",
                    **{
                        field: f"{float(value):.3f}"
                        for field, value in stage_timings.items()
                    },
                    "template_persist_ms": f"{template_persist_ms:.3f}",
                    "template_registration_overhead_ms": (
                        f"{template_registration_overhead_ms:.3f}"
                    ),
                    "num_keypoints": str(stage_timings["num_keypoints"]),
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.warning("skip %s/%s: %s", row.get("identity_id"), row.get("image_id"), exc)
            timings.append(
                {
                    "identity_id": row.get("identity_id", ""),
                    "image_id": row.get("image_id", ""),
                    "image_path": row.get("image_path", ""),
                    "template_path": "",
                    "template_build_ms": elapsed_ms,
                    "num_keypoints": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )
            errors.append({**row, "error": str(exc)})

    return {
        "num_input_rows": len(rows),
        "num_templates": len(successes),
        "num_errors": len(errors),
        "success_rows": successes,
        "errors": errors,
        "template_timings": timings,
        "descriptor_type": "hardnet",
        "descriptor_kind": hardnet.descriptor_kind,
        "descriptor_metric": hardnet.descriptor_metric,
        "descriptor_dim": hardnet.descriptor_dim,
        "descriptor_storage": hardnet.descriptor_storage,
        "descriptor_bitorder": hardnet.descriptor_bitorder,
        "descriptor_transform_name": hardnet.descriptor_transform_name,
        "descriptor_transform_id": hardnet.descriptor_transform_id,
        "inference_settings": hardnet.inference_settings(),
    }


def apply_identity_template_split(
    rows: list[dict[str, str]],
    identities: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """根据注册模板索引在内存中标记 enroll/query，避免持久化中间 split CSV。"""

    enrolled = {
        (str(identity["identity_id"]), str(image_id))
        for identity in identities
        for image_id in identity.get("template_image_ids", [])
    }
    return [
        {
            **row,
            "split": (
                "enroll"
                if (str(row["identity_id"]), str(row["image_id"])) in enrolled
                else "query"
            ),
        }
        for row in rows
    ]


def select_enrollment_rows(
    candidates: list[dict[str, str]],
    enrollment_count: int,
    selection_strategy: str,
    *,
    seed: int = 42,
) -> list[dict[str, str]]:
    """按目录扫描顺序从候选图像中选择注册模板。"""

    ordered = list(candidates)
    count = max(0, int(enrollment_count))
    if count <= 0 or not ordered:
        return []
    strategy = str(selection_strategy or "first").strip().lower()
    if strategy == "first":
        return ordered[: min(count, len(ordered))]
    if strategy == "stride":
        step = max(1, len(ordered) // count)
        return ordered[::step][:count]
    if strategy == "random":
        return random.Random(int(seed)).sample(ordered, min(count, len(ordered)))
    raise ValueError(
        "enrollment.selection_strategy 仅支持 first、stride 或 random，"
        f"收到: {selection_strategy!r}"
    )


def build_identity_templates(
    rows: list[dict[str, str]],
    output_path: str | Path,
    enrollment_count: int,
    seed: int = 42,
    selection_strategy: str = "first",
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """为每个 identity 选择注册模板，并返回内存中的 query 划分。"""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["identity_id"]].append(row)

    strategy = str(selection_strategy or "first").strip().lower()
    identities: list[dict[str, Any]] = []
    warnings: list[str] = []
    for identity_id in groups:
        chosen = select_enrollment_rows(
            groups[identity_id],
            enrollment_count,
            strategy,
            seed=int(seed),
        )
        if len(chosen) < int(enrollment_count):
            warnings.append(f"{identity_id} has only {len(chosen)} templates")
        identities.append(
            {
                "identity_id": identity_id,
                "template_paths": [row["template_path"] for row in chosen],
                "template_image_ids": [row["image_id"] for row in chosen],
                "num_templates": len(chosen),
            }
        )

    payload = {
        "enrollment_images_per_identity": int(enrollment_count),
        "selection_strategy": strategy,
        "random_seed": int(seed),
        "num_identities": len(identities),
        "identities": identities,
        "warnings": warnings,
    }
    split_rows = apply_identity_template_split(rows, identities)
    write_json(output_path, payload)
    return payload, split_rows


def load_identity_templates(path: str | Path) -> list[dict[str, Any]]:
    """读取 identity_templates JSON。"""

    import json

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return payload["identities"] if isinstance(payload, dict) else payload
