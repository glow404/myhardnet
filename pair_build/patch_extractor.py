"""正样本 patch 提取阶段。

作用：
- 读取 build_positive 阶段生成的 all_positive_pairs.csv。
- 根据 CSV 中的 keypoint 坐标和方向，从原始小图中裁剪正样本 patch 对。
- 按配置裁剪并输出 HardNet 需要的单通道 PNG；当前默认配置为直接裁 32x32。
- 将成功落盘的 patch 路径写回 all_positive_pairs.csv，并拆分出
  train_pairs.csv / val_pairs.csv / test_pairs.csv。

输入：
- all_positive_pairs.csv：包含几何验证后的正样本 correspondence。
- 原始图像：CSV 中的 image_a_path / image_b_path。

输出：
- patches/：按 split/finger_id 保存的 PNG patch。
- all_positive_pairs.csv：更新 patch_a_path / patch_p_path 后的可训练样本表。
- patch_diagnostics.csv / patch_summary.json：patch 质量与失败原因统计。
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from positive_builder import POSITIVE_FIELDNAMES
from utils import ensure_dir, get_nested, imread_grayscale, imwrite_image, resolve_path, sanitize_token, stable_id, write_json


PATCH_DIAGNOSTIC_FIELDNAMES = [
    "pair_id",
    "split",
    "finger_id",
    "image_pair_id",
    "source",
    "ok",
    "reason",
    "patch_a_path",
    "patch_p_path",
    "patch_a_std",
    "patch_p_std",
    "patch_a_blank_ratio",
    "patch_p_blank_ratio",
    "patch_a_valid_ratio",
    "patch_p_valid_ratio",
    "patch_a_overlap_ratio",
    "patch_p_overlap_ratio",
]


def _to_float(row: dict[str, str], key: str) -> float:
    """从 CSV 行中读取浮点数。"""

    return float(row[key])


def _load_image(path: str, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    """懒加载原图，并用路径做缓存，避免同一张图重复解码。"""

    if path not in cache:
        cache[path] = imread_grayscale(path)
    return cache[path]


def _overlap_ratio(image_shape: tuple[int, int], x: float, y: float, crop_size: int) -> float:
    """计算裁剪框与原图的有效重叠比例。

    patch 中央点靠近边缘时，旋转裁剪会产生黑边。这个比例用于提前过滤严重越界样本。
    """

    height, width = image_shape
    half = crop_size / 2.0
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / max(float(crop_size * crop_size), 1.0)


def _extract_aligned_patch(image: np.ndarray, x: float, y: float, angle: float, crop_size: int, out_size: int) -> np.ndarray:
    """按 keypoint 方向旋转对齐后裁剪 patch。

    先围绕关键点旋转整张小图，再用 getRectSubPix 取局部窗口。
    这样 patch 内的主方向更一致，HardNet 不必把大量容量浪费在旋转变化上。
    """

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(x), float(y)), float(angle), 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    patch = cv2.getRectSubPix(rotated, patchSize=(int(crop_size), int(crop_size)), center=(float(x), float(y)))
    return cv2.resize(patch, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)


def _patch_metrics(patch: np.ndarray, blank_threshold: int) -> dict[str, float]:
    """计算 patch 质量指标：标准差、空白比例、有效像素比例。"""

    patch_float = patch.astype(np.float32)
    return {
        "std": float(np.std(patch_float)),
        "blank_ratio": float(np.mean(patch_float <= blank_threshold)),
        "valid_ratio": float(np.mean(patch_float > blank_threshold)),
    }


def _validate_and_extract(image: np.ndarray | None, x: float, y: float, angle: float, config: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """提取单个 patch 并做质量过滤。

    过滤顺序：
    1. 图像能否读取
    2. 裁剪框是否严重越界
    3. 空白比例是否过高
    4. 灰度标准差是否过低
    5. 有效像素比例是否足够
    """

    if image is None:
        return None, {"ok": False, "reason": "image_read_failed"}
    crop_size = int(get_nested(config, "patch", "patch_crop_size", default=64))
    out_size = int(get_nested(config, "patch", "patch_out_size", default=32))
    min_overlap = float(get_nested(config, "patch", "min_overlap_ratio", default=0.55))
    blank_threshold = int(get_nested(config, "patch", "blank_threshold", default=8))
    max_blank = float(get_nested(config, "patch", "max_blank_ratio", default=0.55))
    min_std = float(get_nested(config, "patch", "min_patch_std", default=5.0))
    min_valid = float(get_nested(config, "patch", "min_valid_ratio", default=0.35))

    overlap = _overlap_ratio(image.shape[:2], x, y, crop_size)
    if overlap < min_overlap:
        return None, {"ok": False, "reason": "overlap_too_low", "overlap_ratio": overlap}
    patch = _extract_aligned_patch(image, x, y, angle, crop_size, out_size)
    metrics = _patch_metrics(patch, blank_threshold)
    metrics["overlap_ratio"] = overlap
    if metrics["blank_ratio"] > max_blank:
        return None, {"ok": False, "reason": "blank_ratio_too_high", **metrics}
    if metrics["std"] < min_std:
        return None, {"ok": False, "reason": "patch_std_too_low", **metrics}
    if metrics["valid_ratio"] < min_valid:
        return None, {"ok": False, "reason": "valid_ratio_too_low", **metrics}
    return patch, {"ok": True, "reason": "", **metrics}


def _patch_path(output_root: Path, row: dict[str, str], side: str) -> Path:
    """构造 patch 输出路径。

    文件名使用短 hash，避免 Windows 下完整 image_pair_id 过长导致路径写入失败；
    完整可读信息仍保存在 CSV 元数据里。
    """

    split = sanitize_token(row["split"])
    finger = sanitize_token(row["finger_id"])
    source = sanitize_token(row["source"])
    kp_key = "kp_a_idx" if side == "a" else "kp_b_idx"
    short_id = stable_id(row["pair_id"], side, readable_parts=0, digest_size=14).strip("_")
    filename = f"{source}_kp{sanitize_token(row[kp_key])}_{short_id}_{side}.png"
    return output_root / "patches" / split / finger / filename


def extract_patches(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """执行 patch 提取阶段。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    input_path = output_root / "all_positive_pairs.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"缺少 all_positive_pairs.csv，请先运行 build_positive: {input_path}")

    temp_path = output_root / "all_positive_pairs.tmp.csv"
    diagnostics_path = output_root / "patch_diagnostics.csv"
    split_paths = {split: output_root / f"{split}_pairs.csv" for split in ["train", "val", "test"]}
    image_cache: dict[str, np.ndarray | None] = {}
    reason_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    total_rows = 0
    kept_rows = 0

    # 使用临时 CSV 写出成功样本，最后原子替换 all_positive_pairs.csv。
    # 这样即使中途失败，也尽量不破坏原始正样本 CSV。
    with input_path.open("r", encoding="utf-8-sig", newline="") as in_handle, temp_path.open("w", encoding="utf-8", newline="") as all_handle, diagnostics_path.open("w", encoding="utf-8", newline="") as diag_handle:
        reader = csv.DictReader(in_handle)
        all_writer = csv.DictWriter(all_handle, fieldnames=POSITIVE_FIELDNAMES)
        diag_writer = csv.DictWriter(diag_handle, fieldnames=PATCH_DIAGNOSTIC_FIELDNAMES)
        all_writer.writeheader()
        diag_writer.writeheader()

        split_handles = {name: path.open("w", encoding="utf-8", newline="") for name, path in split_paths.items()}
        try:
            split_writers = {name: csv.DictWriter(handle, fieldnames=POSITIVE_FIELDNAMES) for name, handle in split_handles.items()}
            for writer in split_writers.values():
                writer.writeheader()

            for row in reader:
                total_rows += 1
                image_a = _load_image(row["image_a_path"], image_cache)
                image_b = _load_image(row["image_b_path"], image_cache)
                patch_a, metrics_a = _validate_and_extract(image_a, _to_float(row, "x_a"), _to_float(row, "y_a"), _to_float(row, "angle_a"), config)
                patch_p, metrics_p = _validate_and_extract(image_b, _to_float(row, "x_b"), _to_float(row, "y_b"), _to_float(row, "angle_b"), config)

                patch_a_path = _patch_path(output_root, row, "a")
                patch_p_path = _patch_path(output_root, row, "p")
                ok = patch_a is not None and patch_p is not None
                reason = ""
                # 任意一侧 patch 不合格，该正样本对都不能用于 HardNet 训练。
                if not ok:
                    reason = str(metrics_a.get("reason") or metrics_p.get("reason") or "unknown")
                    reason_counts[reason] += 1
                else:
                    if not imwrite_image(patch_a_path, patch_a) or not imwrite_image(patch_p_path, patch_p):
                        ok = False
                        reason = "patch_write_failed"
                        reason_counts[reason] += 1

                diag_writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "split": row["split"],
                        "finger_id": row["finger_id"],
                        "image_pair_id": row["image_pair_id"],
                        "source": row["source"],
                        "ok": int(ok),
                        "reason": reason,
                        "patch_a_path": str(patch_a_path) if ok else "",
                        "patch_p_path": str(patch_p_path) if ok else "",
                        "patch_a_std": metrics_a.get("std", ""),
                        "patch_p_std": metrics_p.get("std", ""),
                        "patch_a_blank_ratio": metrics_a.get("blank_ratio", ""),
                        "patch_p_blank_ratio": metrics_p.get("blank_ratio", ""),
                        "patch_a_valid_ratio": metrics_a.get("valid_ratio", ""),
                        "patch_p_valid_ratio": metrics_p.get("valid_ratio", ""),
                        "patch_a_overlap_ratio": metrics_a.get("overlap_ratio", ""),
                        "patch_p_overlap_ratio": metrics_p.get("overlap_ratio", ""),
                    }
                )
                if not ok:
                    continue

                row["patch_a_path"] = str(patch_a_path)
                row["patch_p_path"] = str(patch_p_path)
                all_writer.writerow({key: row.get(key, "") for key in POSITIVE_FIELDNAMES})
                split_writers[row["split"]].writerow({key: row.get(key, "") for key in POSITIVE_FIELDNAMES})
                kept_rows += 1
                split_counts[row["split"]] += 1
        finally:
            for handle in split_handles.values():
                handle.close()

    os.replace(temp_path, input_path)
    summary_path = output_root / "patch_summary.json"
    write_json(
        summary_path,
        {
            "input_pair_count": total_rows,
            "kept_pair_count": kept_rows,
            "dropped_pair_count": total_rows - kept_rows,
            "split_counts": dict(split_counts),
            "drop_reason_counts": dict(reason_counts),
            "image_cache_count": len(image_cache),
        },
    )
    logger.info("patch 提取完成 | input=%d | kept=%d | dropped=%d", total_rows, kept_rows, total_rows - kept_rows)
    return {"all_positive_pairs_path": str(input_path), "patch_diagnostics_path": str(diagnostics_path), "summary_path": str(summary_path), "kept_pair_count": kept_rows}
