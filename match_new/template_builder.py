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

from val_new.core import HardNetDescriptor, build_sift, detect_sift_keypoints, imread_grayscale, patchable_keypoints

from .utils import ensure_dir, template_filename, write_csv_rows, write_json


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


def save_hardnet_template(
    output_path: str | Path,
    row: dict[str, str],
    keypoints: list[cv2.KeyPoint],
    hardnet_descriptors: np.ndarray,
    overlap_image: np.ndarray,
) -> Path:
    """保存 HardNet 专用图像模板及纹理验证使用的无损灰度图。"""

    target = Path(output_path).expanduser()
    ensure_dir(target.parent)
    xy = np.asarray([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    gray = np.asarray(overlap_image)
    if gray.ndim != 2:
        raise ValueError(f"overlap_image must be a 2D grayscale image, got {gray.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    np.savez_compressed(
        target,
        template_format_version=np.asarray(2, dtype=np.uint8),
        identity_id=np.asarray(row["identity_id"]),
        image_id=np.asarray(row["image_id"]),
        image_path=np.asarray(row["image_path"]),
        keypoints_xy=xy,
        keypoints_size=np.asarray([kp.size for kp in keypoints], dtype=np.float32),
        keypoints_angle=np.asarray([kp.angle if kp.angle >= 0 else 0.0 for kp in keypoints], dtype=np.float32),
        keypoints_response=np.asarray([kp.response for kp in keypoints], dtype=np.float32),
        hardnet_descriptors=np.asarray(hardnet_descriptors, dtype=np.float32),
        overlap_image=gray,
    )
    return target


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
        hardnet = (
            np.asarray(data["hardnet_descriptors"], dtype=np.float32)
            if has_hardnet
            else np.zeros((0, 128), dtype=np.float32)
        )
        sift = (
            np.asarray(data["sift_descriptors"], dtype=np.float32)
            if has_sift
            else np.zeros((0, 128), dtype=np.float32)
        )
        return {
            "template_format_version": int(np.asarray(data["template_format_version"]).item()) if "template_format_version" in keys else 1,
            "identity_id": str(np.asarray(data["identity_id"]).item()),
            "image_id": str(np.asarray(data["image_id"]).item()),
            "image_path": str(np.asarray(data["image_path"]).item()),
            "keypoints_xy": np.asarray(data["keypoints_xy"], dtype=np.float32) if "keypoints_xy" in keys else np.zeros((0, 2), dtype=np.float32),
            "keypoints_size": np.asarray(data["keypoints_size"], dtype=np.float32) if "keypoints_size" in keys else np.zeros((n_xy,), dtype=np.float32),
            "keypoints_angle": np.asarray(data["keypoints_angle"], dtype=np.float32) if "keypoints_angle" in keys else np.zeros((n_xy,), dtype=np.float32),
            "keypoints_response": np.asarray(data["keypoints_response"], dtype=np.float32) if "keypoints_response" in keys else np.zeros((n_xy,), dtype=np.float32),
            "hardnet_descriptors": hardnet,
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
    patch_cfg = dict(config.get("patch", {}))
    filter_cfg = dict(config.get("keypoint_filter", {}))
    max_keypoints = int(filter_cfg.get("max_keypoints", 400))
    successes: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    timings: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="build hardnet templates"):
        started = time.perf_counter()
        try:
            image = imread_grayscale(row["image_path"])
            if image is None:
                raise FileNotFoundError(row["image_path"])
            keypoints = detect_keypoints_only(image, config, sift=sift)
            if max_keypoints > 0 and len(keypoints) > max_keypoints:
                order = np.argsort([kp.response for kp in keypoints])[::-1][:max_keypoints]
                keypoints = [keypoints[int(i)] for i in order]
            selected_keypoints, patches, _selected_indices = extract_keypoint_patches(
                image,
                keypoints,
                crop_size=int(patch_cfg.get("crop_size", 64)),
                out_size=int(patch_cfg.get("out_size", 32)),
                min_overlap_ratio=float(patch_cfg.get("min_overlap_ratio", 0.55)),
                normalize=bool(patch_cfg.get("normalize", True)),
            )
            hardnet_desc = hardnet.describe(patches)
            if hardnet_desc.shape[0] != len(selected_keypoints):
                raise ValueError("descriptor/keypoint alignment failed")
            path = out / template_filename(row["identity_id"], row["image_id"])
            save_hardnet_template(path, row, selected_keypoints, hardnet_desc, image)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            timing = {
                "identity_id": row["identity_id"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "template_path": str(path),
                "template_build_ms": elapsed_ms,
                "num_keypoints": len(selected_keypoints),
                "status": "success",
            }
            timings.append(timing)
            successes.append({**row, "template_path": str(path), "template_build_ms": f"{elapsed_ms:.3f}", "num_keypoints": str(len(selected_keypoints))})
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
    }


def build_identity_templates(
    rows: list[dict[str, str]],
    output_path: str | Path,
    split_metadata_output: str | Path,
    enrollment_count: int,
    seed: int,
) -> dict[str, Any]:
    """为每个 identity 随机选择注册模板，并写出 split metadata。"""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["identity_id"]].append(row)

    rng = random.Random(int(seed))
    selected: set[tuple[str, str]] = set()
    identities: list[dict[str, Any]] = []
    warnings: list[str] = []
    for identity_id in sorted(groups):
        candidates = sorted(groups[identity_id], key=lambda item: item["image_id"])
        rng.shuffle(candidates)
        chosen = candidates[: min(int(enrollment_count), len(candidates))]
        if len(chosen) < int(enrollment_count):
            warnings.append(f"{identity_id} has only {len(chosen)} templates")
        for row in chosen:
            selected.add((row["identity_id"], row["image_id"]))
        identities.append(
            {
                "identity_id": identity_id,
                "template_paths": [row["template_path"] for row in chosen],
                "template_image_ids": [row["image_id"] for row in chosen],
                "num_templates": len(chosen),
            }
        )

    split_rows = []
    for row in rows:
        split_rows.append({**row, "split": "enroll" if (row["identity_id"], row["image_id"]) in selected else "query"})
    write_csv_rows(split_metadata_output, split_rows)
    payload = {
        "enrollment_images_per_identity": int(enrollment_count),
        "selection_strategy": "random",
        "random_seed": int(seed),
        "num_identities": len(identities),
        "identities": identities,
        "warnings": warnings,
    }
    write_json(output_path, payload)
    return payload


def load_identity_templates(path: str | Path) -> list[dict[str, Any]]:
    """读取 identity_templates JSON。"""

    import json

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return payload["identities"] if isinstance(payload, dict) else payload
