"""原始指纹小图索引阶段。

作用：
- 扫描配置中的 `paths.data_root`，收集所有允许后缀的图像文件。
- 按相对路径前 `index.finger_id_depth` 级目录生成 `finger_id`。
- 读取每张图确认可用，并记录宽高、绝对路径、相对路径等元数据。

输入：
- pair_build/data 下的原始小局部指纹图。

输出：
- sample_index.csv：后续 split / build_positive 的主索引。
- index_summary.json：图像数、finger 数、坏图、重复 stem 等 QA 信息。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utils import ensure_dir, get_nested, image_extensions, imread_grayscale, resolve_path, write_csv_rows, write_json


def _relative_stem(path: Path, root: Path) -> str:
    """生成相对 root 的无后缀 key，用于稳定标识一张图像。"""

    return path.relative_to(root).with_suffix("").as_posix()


def _extract_finger_id(relative_stem: str, depth: int) -> str:
    """从相对路径中提取 finger_id。

    当前数据约定前三层目录共同定义一个 finger_id，例如：
    `butieping/fys_R0/Rgd1237`。
    """

    parts = Path(relative_stem).parts[:-1]
    if not parts:
        return "unknown_finger"
    if depth <= 0:
        return "/".join(parts)
    return "/".join(parts[:depth])


def build_sample_index(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """执行索引阶段并写出产物。"""

    data_root = resolve_path(config, get_nested(config, "paths", "data_root", default="data"))
    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    ensure_dir(output_root)

    if not data_root.exists():
        raise FileNotFoundError(f"原图目录不存在: {data_root}")

    allowed = set(image_extensions(config))
    priority = {ext: index for index, ext in enumerate(image_extensions(config))}
    finger_id_depth = int(get_nested(config, "index", "finger_id_depth", default=3))

    # 用相对 stem 做主键。若同一个 stem 同时存在 bmp/png 等多个版本，
    # 则按 config 中 image_extensions 的顺序选择优先级更高的文件。
    items: dict[str, Path] = {}
    duplicates: list[dict[str, str]] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        key = _relative_stem(path, data_root)
        if key in items:
            old = items[key]
            selected = path if priority.get(path.suffix.lower(), 999) < priority.get(old.suffix.lower(), 999) else old
            discarded = old if selected == path else path
            items[key] = selected
            duplicates.append({"sample_key": key, "kept_path": str(selected), "discarded_path": str(discarded)})
        else:
            items[key] = path

    # 读取图像不仅是为了拿宽高，也可以提前发现损坏或 OpenCV 无法解码的文件。
    rows: list[dict[str, Any]] = []
    bad_images: list[dict[str, str]] = []
    for key, image_path in sorted(items.items()):
        image = imread_grayscale(image_path)
        if image is None:
            bad_images.append({"sample_key": key, "image_path": str(image_path.resolve()), "reason": "image_read_failed"})
            continue
        rows.append(
            {
                "sample_key": key,
                "finger_id": _extract_finger_id(key, depth=finger_id_depth),
                "image_id": Path(key).name,
                "image_path": str(image_path.resolve()),
                "image_rel_path": image_path.relative_to(data_root).as_posix(),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
        )

    sample_index_path = output_root / "sample_index.csv"
    summary_path = output_root / "index_summary.json"
    write_csv_rows(sample_index_path, rows)
    finger_ids = sorted({row["finger_id"] for row in rows})
    write_json(
        summary_path,
        {
            "data_root": str(data_root),
            "image_count": len(rows),
            "finger_count": len(finger_ids),
            "duplicate_count": len(duplicates),
            "bad_image_count": len(bad_images),
            "finger_examples": finger_ids[:20],
            "duplicates": duplicates[:100],
            "bad_images": bad_images[:100],
        },
    )
    logger.info("索引完成 | images=%d | fingers=%d | bad=%d", len(rows), len(finger_ids), len(bad_images))
    return {"sample_index_path": str(sample_index_path), "summary_path": str(summary_path), "image_count": len(rows), "finger_count": len(finger_ids)}
