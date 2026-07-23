"""检查 HardNet 训练 batch 与 hardest negative。

作用：
    直接复用 `hardnet_train.data.FingerprintPairDataset` 和
    `hardnet_train.data.FingerImagePairBatchSampler`，按训练代码的 batch 构造逻辑
    生成若干 batch；然后用指定 HardNet checkpoint 计算描述子，并复现
    `hardnet_train.loss.HardNetLoss` 中的 hardest-in-batch 负样本选择。

输入：
    - pair CSV，例如 `outputs/hardnet_dataset/train_pairs.csv`。
    - HardNet checkpoint，例如 `outputs/hardnet_train_post_nips_cosine5.26/best.pt`。
    - batch_size、fingers_per_batch、batches 等采样参数。

输出：
    - `batch_manifest.csv`：所有 batch、所有样本的 hardest negative 明细。
    - `batch_000/batch_records.csv`：单个 batch 的明细。
    - `batch_000/sample_000/...`：
      - `anchor.png`
      - `positive.png`
      - `hard_negative.png`
      - `triplet_panel.png`

用途：
    用于人工检查训练时 HardNet 看到的 anchor/positive/hardest-negative
    到底是什么局部 patch，排查伪负样本、过难负样本、batch 构造是否合理等问题。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardnet_train.data import FingerImagePairBatchSampler, FingerprintPairDataset
from val_new.core import (
    HardNetDescriptor,
    copy_file,
    ensure_dir,
    get_nested,
    imread_grayscale,
    imwrite_image,
    load_config,
    resolve_path,
    seed_everything,
    write_csv,
    write_json,
)


MANIFEST_FIELDS = [
    "batch_index",
    "row_in_batch",
    "dataset_index",
    "pair_id",
    "finger_id",
    "image_pair_id",
    "point_group",
    "anchor_patch_path",
    "positive_patch_path",
    "positive_dist",
    "hard_negative_dist",
    "triplet_loss",
    "hard_negative_dataset_index",
    "hard_negative_row_in_batch",
    "hard_negative_pair_id",
    "hard_negative_finger_id",
    "hard_negative_image_pair_id",
    "hard_negative_point_group",
    "hard_negative_side",
    "hard_negative_patch_path",
    "hard_negative_mode",
    "sample_dir",
    "triplet_panel_path",
]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Inspect HardNet batch sampler and hardest negative patches.")
    parser.add_argument("--config", default="val_new/config.yaml", help="Config path.")
    parser.add_argument("--pairs-csv", default=None, help="Pair CSV. Default uses outputs/hardnet_dataset/train_pairs.csv.")
    parser.add_argument("--checkpoint", default=None, help="Override HardNet checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--batches", type=int, default=2, help="How many batches to inspect.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used by the sampler.")
    parser.add_argument("--fingers-per-batch", type=int, default=4, help="fingers_per_batch used by the sampler.")
    parser.add_argument("--margin", type=float, default=0.7, help="Triplet margin used only for reporting loss.")
    parser.add_argument("--seed", type=int, default=None, help="Sampler seed. Defaults to config seed.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows loaded from CSV for quick inspection.")
    parser.add_argument("--max-rows-per-finger", type=int, default=None, help="Limit rows per finger for quick inspection.")
    parser.add_argument("--panel-scale", type=int, default=5, help="Patch visualization scale.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint is not None:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = resolve_path(config, get_nested(config, "output", "output_dir")) / "batch_inspect"
    output_dir = ensure_dir(output_dir)

    seed = int(args.seed if args.seed is not None else get_nested(config, "seed", default=42))
    seed_everything(seed)
    pairs_csv = Path(args.pairs_csv).expanduser().resolve() if args.pairs_csv else (ROOT / "outputs" / "hardnet_dataset" / "train_pairs.csv").resolve()

    dataset = FingerprintPairDataset(
        pairs_csv,
        max_rows=args.max_rows,
        max_rows_per_finger=args.max_rows_per_finger,
        normalize=bool(get_nested(config, "patch", "normalize", default=True)),
    )
    sampler = FingerImagePairBatchSampler(
        dataset.records,
        batch_size=int(args.batch_size),
        fingers_per_batch=int(args.fingers_per_batch),
        batches_per_epoch=int(args.batches),
        seed=seed,
        drop_incomplete=True,
    )
    hardnet = HardNetDescriptor(config)

    all_rows: list[dict[str, Any]] = []
    for batch_index, batch_indices in enumerate(sampler):
        if batch_index >= int(args.batches):
            break
        batch_dir = ensure_dir(output_dir / f"batch_{batch_index:03d}")
        batch_rows = inspect_batch(
            batch_index=batch_index,
            batch_indices=batch_indices,
            dataset=dataset,
            hardnet=hardnet,
            margin=float(args.margin),
            batch_dir=batch_dir,
            panel_scale=int(args.panel_scale),
        )
        write_csv(batch_dir / "batch_records.csv", batch_rows, MANIFEST_FIELDS)
        all_rows.extend(batch_rows)
        print(f"wrote batch {batch_index}: {batch_dir}", flush=True)

    manifest_path = write_csv(output_dir / "batch_manifest.csv", all_rows, MANIFEST_FIELDS)
    summary_path = write_json(
        output_dir / "batch_inspect_summary.json",
        {
            "pairs_csv": str(pairs_csv),
            "checkpoint": str(resolve_path(config, get_nested(config, "model", "checkpoint"))),
            "loaded_records": len(dataset.records),
            "requested_batches": int(args.batches),
            "written_rows": len(all_rows),
            "batch_size": int(args.batch_size),
            "fingers_per_batch": int(args.fingers_per_batch),
            "margin": float(args.margin),
            "seed": seed,
            "manifest_path": str(manifest_path),
        },
    )
    print(f"wrote: {manifest_path}")
    print(f"wrote: {summary_path}")


def inspect_batch(
    batch_index: int,
    batch_indices: list[int],
    dataset: FingerprintPairDataset,
    hardnet: HardNetDescriptor,
    margin: float,
    batch_dir: Path,
    panel_scale: int,
) -> list[dict[str, Any]]:
    """检查单个 batch，并输出每个样本的三联 patch 可视化。"""

    anchors, positives, point_groups = load_batch_tensors(dataset, batch_indices)
    anchor_desc = hardnet.describe(anchors)
    positive_desc = hardnet.describe(positives)
    distances = pairwise_l2(anchor_desc, positive_desc)
    hard_negative_info = find_hardest_negatives(distances, point_groups)

    rows: list[dict[str, Any]] = []
    for row_in_batch, dataset_index in enumerate(batch_indices):
        record = dataset.records[dataset_index]
        negative = hard_negative_info[row_in_batch]
        sample_dir = ensure_dir(batch_dir / f"sample_{row_in_batch:03d}_{safe_token(record.finger_id)}")
        copied_anchor = sample_dir / "anchor.png"
        copied_positive = sample_dir / "positive.png"
        copy_file(record.patch_a_path, copied_anchor)
        copy_file(record.patch_p_path, copied_positive)

        neg_record = dataset.records[batch_indices[negative["row_in_batch"]]] if negative["row_in_batch"] is not None else None
        hard_negative_path = hard_negative_patch_path(neg_record, str(negative["side"]))
        copied_negative = sample_dir / "hard_negative.png"
        if hard_negative_path:
            copy_file(hard_negative_path, copied_negative)

        panel_path = sample_dir / "triplet_panel.png"
        make_triplet_panel(
            anchor_path=record.patch_a_path,
            positive_path=record.patch_p_path,
            hard_negative_path=hard_negative_path,
            output_path=panel_path,
            title_lines=[
                f"batch={batch_index} row={row_in_batch}",
                f"pos={distances[row_in_batch, row_in_batch]:.4f} hard_neg={negative['distance']:.4f}",
                f"loss={max(0.0, margin + distances[row_in_batch, row_in_batch] - negative['distance']):.4f}",
            ],
            scale=panel_scale,
        )

        rows.append(
            {
                "batch_index": batch_index,
                "row_in_batch": row_in_batch,
                "dataset_index": dataset_index,
                "pair_id": record.pair_id,
                "finger_id": record.finger_id,
                "image_pair_id": record.image_pair_id,
                "point_group": int(record.point_group),
                "anchor_patch_path": record.patch_a_path,
                "positive_patch_path": record.patch_p_path,
                "positive_dist": format_float(distances[row_in_batch, row_in_batch]),
                "hard_negative_dist": format_float(negative["distance"]),
                "triplet_loss": format_float(max(0.0, margin + distances[row_in_batch, row_in_batch] - negative["distance"])),
                "hard_negative_dataset_index": batch_indices[negative["row_in_batch"]] if negative["row_in_batch"] is not None else "",
                "hard_negative_row_in_batch": negative["row_in_batch"] if negative["row_in_batch"] is not None else "",
                "hard_negative_pair_id": neg_record.pair_id if neg_record is not None else "",
                "hard_negative_finger_id": neg_record.finger_id if neg_record is not None else "",
                "hard_negative_image_pair_id": neg_record.image_pair_id if neg_record is not None else "",
                "hard_negative_point_group": int(neg_record.point_group) if neg_record is not None else "",
                "hard_negative_side": negative["side"],
                "hard_negative_patch_path": hard_negative_path,
                "hard_negative_mode": negative["mode"],
                "sample_dir": str(sample_dir),
                "triplet_panel_path": str(panel_path),
            }
        )
    return rows


def load_batch_tensors(
    dataset: FingerprintPairDataset,
    batch_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 dataset index 读取一个 batch 的 anchor/positive/point_group。"""

    anchors: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    point_groups: list[int] = []
    for index in batch_indices:
        item = dataset[index]
        anchors.append(item["anchor"].numpy())
        positives.append(item["positive"].numpy())
        point_groups.append(int(item["point_group"]))
    return (
        np.stack(anchors).astype(np.float32),
        np.stack(positives).astype(np.float32),
        np.asarray(point_groups, dtype=np.int64),
    )


def pairwise_l2(anchor_desc: np.ndarray, positive_desc: np.ndarray) -> np.ndarray:
    """计算 distances[i, j] = d(anchor_i, positive_j)。"""

    diff = anchor_desc[:, None, :] - positive_desc[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def find_hardest_negatives(distances: np.ndarray, point_groups: np.ndarray) -> list[dict[str, Any]]:
    """复现 HardNetLoss 中 hardest negative 的选择逻辑。

    对第 i 个正样本对，loss 同时考虑：
    - anchor_i 到其他 positive_j 的最近距离。
    - positive_i 到其他 anchor_j 的最近距离，也就是 distances[j, i]。

    二者取更小者作为 hardest negative。
    """

    batch_size = distances.shape[0]
    invalid = np.eye(batch_size, dtype=bool)
    invalid |= point_groups[:, None] == point_groups[None, :]
    large = np.finfo(np.float32).max / 16.0
    infos: list[dict[str, Any]] = []
    for i in range(batch_size):
        row_distances = distances[i].copy()
        row_distances[invalid[i]] = large
        col_distances = distances[:, i].copy()
        col_distances[invalid[:, i]] = large
        row_j = int(np.argmin(row_distances))
        col_j = int(np.argmin(col_distances))
        row_value = float(row_distances[row_j])
        col_value = float(col_distances[col_j])
        if row_value >= large / 2.0 and col_value >= large / 2.0:
            infos.append({"row_in_batch": None, "side": "", "mode": "none", "distance": float("nan"), "patch_path": ""})
        elif row_value <= col_value:
            infos.append(
                {
                    "row_in_batch": row_j,
                    "side": "positive",
                    "mode": "anchor_i_vs_positive_j",
                    "distance": row_value,
                    "patch_path": None,
                }
            )
        else:
            infos.append(
                {
                    "row_in_batch": col_j,
                    "side": "anchor",
                    "mode": "positive_i_vs_anchor_j",
                    "distance": col_value,
                    "patch_path": None,
                }
            )
    return infos


def hard_negative_patch_path(record: Any, side: str) -> str:
    """根据 hardest negative 来源侧返回实际 patch 路径。"""

    if record is None:
        return ""
    if side == "positive":
        return str(record.patch_p_path)
    if side == "anchor":
        return str(record.patch_a_path)
    return ""


def make_triplet_panel(
    anchor_path: str,
    positive_path: str,
    hard_negative_path: str,
    output_path: Path,
    title_lines: list[str],
    scale: int,
) -> bool:
    """生成 anchor/positive/hard-negative 三联图。"""

    patches = [
        ("anchor", imread_grayscale(anchor_path)),
        ("positive", imread_grayscale(positive_path)),
        ("hard negative", imread_grayscale(hard_negative_path) if hard_negative_path else None),
    ]
    panels: list[np.ndarray] = []
    for label, patch in patches:
        if patch is None:
            patch = np.zeros((32, 32), dtype=np.uint8)
        patch = normalize_to_uint8(patch)
        patch = cv2.resize(patch, (patch.shape[1] * scale, patch.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        canvas = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        canvas = cv2.copyMakeBorder(canvas, 34, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    body = np.concatenate(panels, axis=1)
    header_height = 24 * max(1, len(title_lines)) + 8
    header = np.full((header_height, body.shape[1], 3), 255, dtype=np.uint8)
    for line_index, line in enumerate(title_lines):
        cv2.putText(header, line, (8, 24 + 24 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    image = np.concatenate([header, body], axis=0)
    return imwrite_image(output_path, image)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """把 patch 转成可显示的 uint8。"""

    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    values = values - float(values.min())
    denom = max(float(values.max()), 1e-6)
    return np.clip(values / denom * 255.0, 0, 255).astype(np.uint8)


def safe_token(text: str) -> str:
    """生成适合文件名的 token。"""

    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return token[:60] or "sample"


def format_float(value: float) -> str:
    """CSV 友好的浮点数字符串。"""

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
