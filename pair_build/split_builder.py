"""按 finger_id 划分 train / val / test。

作用：
- 读取 index 阶段生成的 sample_index.csv。
- 在 finger_id 粒度上随机划分数据集，避免同一个手指同时出现在训练和验证/测试中。

输出：
- split_assignments.csv：sample_index 的每一行增加 split 字段。
- split_summary.json：每个 split 的 finger 数、图像数和后续正样本数占位。

注意：
- 这里只做身份隔离，不做任何图像匹配，也不生成 patch。
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

from utils import get_nested, read_csv_rows, resolve_path, write_csv_rows, write_json


def _assign_fingers(finger_ids: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """给每个 finger_id 分配 split。

    比例会先归一化，因此 8/1/1 和 0.8/0.1/0.1 等价。
    当 finger 数较少导致某个 split 为空时，会尽量从样本最多的 split 借一个。
    """

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split 比例之和必须大于 0")
    train_ratio /= total_ratio
    val_ratio /= total_ratio

    # 只打乱 finger_id，不打乱同一 finger 内的图像。
    # 后续 build_positive 会在每个 split/finger 内部做两两图像组合。
    shuffled = list(finger_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(round(total * train_ratio))
    val_end = int(round(total * (train_ratio + val_ratio)))
    split_map = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    if total >= 3:
        for split_name in ["train", "val", "test"]:
            if split_map[split_name]:
                continue
            donor = max(["train", "val", "test"], key=lambda name: len(split_map[name]))
            if len(split_map[donor]) > 1:
                split_map[split_name].append(split_map[donor].pop())

    assignments: dict[str, str] = {}
    for split_name, ids in split_map.items():
        for finger_id in ids:
            assignments[finger_id] = split_name
    return assignments


def build_split(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """执行 split 阶段并写出 split_assignments.csv。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    sample_index_path = output_root / "sample_index.csv"
    if not sample_index_path.exists():
        raise FileNotFoundError(f"缺少 sample_index.csv，请先运行 index: {sample_index_path}")

    rows = read_csv_rows(sample_index_path)
    if not rows:
        raise ValueError("sample_index.csv 为空")

    train_ratio = float(get_nested(config, "split", "train_ratio", default=0.8))
    val_ratio = float(get_nested(config, "split", "val_ratio", default=0.1))
    test_ratio = float(get_nested(config, "split", "test_ratio", default=0.1))
    seed = int(get_nested(config, "split", "random_seed", default=42))
    finger_ids = sorted({row["finger_id"] for row in rows})
    assignments = _assign_fingers(finger_ids, train_ratio, val_ratio, test_ratio, seed)

    split_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"finger_count": 0, "image_count": 0, "positive_pair_count": 0})
    split_fingers: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_name = assignments[row["finger_id"]]
        current = dict(row)
        current["split"] = split_name
        split_rows.append(current)
        summary[split_name]["image_count"] += 1
        split_fingers[split_name].add(row["finger_id"])

    for split_name, ids in split_fingers.items():
        summary[split_name]["finger_count"] = len(ids)

    split_path = output_root / "split_assignments.csv"
    summary_path = output_root / "split_summary.json"
    write_csv_rows(split_path, split_rows)
    write_json(summary_path, {"random_seed": seed, "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio}, "splits": summary})
    logger.info("split 完成 | train=%d | val=%d | test=%d", summary["train"]["image_count"], summary["val"]["image_count"], summary["test"]["image_count"])
    return {"split_assignments_path": str(split_path), "summary_path": str(summary_path), "split_summary": dict(summary)}
