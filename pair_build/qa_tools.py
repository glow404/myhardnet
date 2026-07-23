"""数据集构建 QA 报告与可视化。

作用：
- 汇总 index/split/build_positive/extract_patches 生成的 JSON/CSV 统计。
- 生成 Markdown 报告，方便快速检查数据规模、source 分布、失败原因。
- 生成两类可视化：
  1. 原图上的 keypoint 匹配连线预览
  2. 已裁剪 patch pair 的网格预览

说明：
- QA 阶段只读已有产物，不重新做 SIFT 或 patch 裁剪。
- 这些预览用于人工抽查几何扩展是否靠谱，不作为训练输入。
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import ensure_dir, get_nested, imread_grayscale, imwrite_image, read_json, resolve_path


def _read_rows_limit(path: Path, limit: int) -> list[dict[str, str]]:
    """从 CSV 中最多读取 limit 行，用于生成预览。"""

    if not path.exists() or limit <= 0:
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= limit:
                break
    return rows


def _count_csv(path: Path, key: str | None = None) -> Counter[str]:
    """统计 CSV 行数或某一字段的取值分布。"""

    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            counter[str(row.get(key, "")) if key else "rows"] += 1
    return counter


def _draw_match_preview(row: dict[str, str], output_path: Path) -> bool:
    """绘制一对原图上的 keypoint 对应关系。

    绿色表示 SIFT/RANSAC 内点，橙色表示 geometry_expanded。
    """

    image_a = imread_grayscale(row["image_a_path"])
    image_b = imread_grayscale(row["image_b_path"])
    if image_a is None or image_b is None:
        return False
    canvas_h = max(image_a.shape[0], image_b.shape[0])
    canvas_w = image_a.shape[1] + image_b.shape[1]
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[: image_a.shape[0], : image_a.shape[1]] = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    canvas[: image_b.shape[0], image_a.shape[1] :] = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR)
    pt_a = (int(round(float(row["x_a"]))), int(round(float(row["y_a"]))))
    pt_b = (int(round(float(row["x_b"]) + image_a.shape[1])), int(round(float(row["y_b"]))))
    color = (0, 180, 0) if row.get("source") == "sift_inlier" else (0, 120, 255)
    cv2.circle(canvas, pt_a, 4, color, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, pt_b, 4, color, 1, lineType=cv2.LINE_AA)
    cv2.line(canvas, pt_a, pt_b, color, 1, lineType=cv2.LINE_AA)
    return imwrite_image(output_path, canvas)


def _build_patch_preview(rows: list[dict[str, str]], output_path: Path) -> bool:
    """把若干 patch pair 拼成一个网格图。

    每个 tile 左侧是 anchor patch，右侧是 positive patch。
    """

    tiles: list[np.ndarray] = []
    for row in rows:
        patch_a = imread_grayscale(row.get("patch_a_path", ""))
        patch_p = imread_grayscale(row.get("patch_p_path", ""))
        if patch_a is None or patch_p is None:
            continue
        pair = np.concatenate([patch_a, patch_p], axis=1)
        pair = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
        tiles.append(pair)
    if not tiles:
        return False

    tile_h, tile_w = tiles[0].shape[:2]
    cols = min(4, len(tiles))
    rows_n = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows_n * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // cols) * tile_h
        x = (index % cols) * tile_w
        canvas[y : y + tile_h, x : x + tile_w] = tile
    return imwrite_image(output_path, canvas)


def generate_qa_report(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """生成 QA Markdown 报告和预览图。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    qa_root = ensure_dir(output_root / "qa")

    index_summary = read_json(output_root / "index_summary.json") if (output_root / "index_summary.json").exists() else {}
    split_summary = read_json(output_root / "split_summary.json") if (output_root / "split_summary.json").exists() else {}
    positive_summary = read_json(output_root / "positive_build_summary.json") if (output_root / "positive_build_summary.json").exists() else {}
    patch_summary = read_json(output_root / "patch_summary.json") if (output_root / "patch_summary.json").exists() else {}

    all_pairs_path = output_root / "all_positive_pairs.csv"
    diagnostics_path = output_root / "match_diagnostics.csv"
    patch_diagnostics_path = output_root / "patch_diagnostics.csv"

    source_counts = _count_csv(all_pairs_path, "source")
    split_counts = _count_csv(all_pairs_path, "split")
    skip_counts = _count_csv(diagnostics_path, "skip_reason")
    patch_drop_counts = _count_csv(patch_diagnostics_path, "reason")

    match_preview_count = int(get_nested(config, "qa", "match_preview_count", default=10))
    patch_preview_count = int(get_nested(config, "qa", "patch_preview_count", default=16))
    preview_rows = _read_rows_limit(all_pairs_path, max(match_preview_count, patch_preview_count))

    match_previews: list[str] = []
    for index, row in enumerate(preview_rows[:match_preview_count]):
        preview_path = qa_root / f"match_preview_{index:03d}_{row.get('source', 'sample')}.png"
        if _draw_match_preview(row, preview_path):
            match_previews.append(str(preview_path))

    patch_preview_path = qa_root / "patch_preview_grid.png"
    patch_preview_ok = _build_patch_preview(preview_rows[:patch_preview_count], patch_preview_path)

    report_path = qa_root / "report.md"
    # Markdown 报告尽量只放高信号统计；更细的逐样本信息保留在各 CSV/JSON 里。
    report_lines = [
        "# HardNet Fingerprint Dataset QA",
        "",
        "## Summary",
        f"- Images: {index_summary.get('image_count', 0)}",
        f"- Fingers: {index_summary.get('finger_count', 0)}",
        f"- Image pairs processed: {positive_summary.get('image_pair_count', 0)}",
        f"- Positive pairs after matching: {positive_summary.get('positive_pair_count', 0)}",
        f"- Positive pairs after patch extraction: {patch_summary.get('kept_pair_count', 0)}",
        "",
        "## Split Counts",
    ]
    for key, value in sorted(split_counts.items()):
        report_lines.append(f"- {key or 'unknown'}: {value}")
    report_lines.extend(["", "## Source Counts"])
    for key, value in sorted(source_counts.items()):
        report_lines.append(f"- {key or 'unknown'}: {value}")
    report_lines.extend(["", "## Match Skip Reasons"])
    for key, value in sorted(skip_counts.items()):
        report_lines.append(f"- {key or 'none'}: {value}")
    report_lines.extend(["", "## Patch Drop Reasons"])
    for key, value in sorted(patch_drop_counts.items()):
        report_lines.append(f"- {key or 'kept'}: {value}")
    report_lines.extend(["", "## Visualizations"])
    for path in match_previews[:10]:
        report_lines.append(f"- Match preview: `{path}`")
    if patch_preview_ok:
        report_lines.append(f"- Patch preview grid: `{patch_preview_path}`")

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("QA 报告完成: %s", report_path)
    return {
        "report_path": str(report_path),
        "match_preview_count": len(match_previews),
        "patch_preview_path": str(patch_preview_path) if patch_preview_ok else "",
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
    }
