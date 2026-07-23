"""在 SIFT 困难图像对上做公平匹配对比。

作用：
    读取 `select_sift_hard_cases.py` 选出的困难图像对，只在这些图像对上比较：

        RootSIFT descriptor vs HardNet descriptor

    公平性原则：
        同一批图像对、同一批 SIFT keypoints、同一种匹配策略、同一组匹配参数；
        唯一变化是 descriptor 类型。

输入：
    - `outputs/val_new_eval/hard_cases/hard_cases.csv`
    - HardNet checkpoint
    - `config.yaml` 中的 strategies

输出：
    - `outputs/val_new_eval/fair_matching/fair_matching_details.csv`
      每个困难图像对、每个策略的详细结果。
    - `outputs/val_new_eval/fair_matching/fair_matching_summary.csv`
      按策略汇总后的公平对比结果。
    - `outputs/val_new_eval/fair_matching/fair_matching_summary.json`
      JSON 版汇总。
    - `outputs/val_new_eval/fair_matching/fair_matching_summary.png`
      RootSIFT/HardNet 平均一对一内点数柱状图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from val_new.core import (
    HardNetDescriptor,
    ImagePair,
    build_sift,
    ensure_dir,
    evaluate_descriptor_pair,
    format_float,
    get_nested,
    load_config,
    prepare_pair,
    resolve_path,
    safe_float,
    seed_everything,
    strategy_label,
    write_csv,
    write_json,
)


DETAIL_FIELDS = [
    "strategy",
    "matcher",
    "ratio_thresh",
    "top_k",
    "mutual",
    "image_pair_id",
    "finger_id",
    "sift_raw_matches",
    "sift_inliers",
    "sift_unique_query_inliers",
    "sift_one_to_one_inliers",
    "sift_mean_reproj_error",
    "hardnet_raw_matches",
    "hardnet_inliers",
    "hardnet_unique_query_inliers",
    "hardnet_one_to_one_inliers",
    "hardnet_mean_reproj_error",
    "inlier_delta",
    "unique_query_delta",
    "one_to_one_delta",
    "hardnet_better_inliers",
    "hardnet_better_one_to_one",
    "eval_keypoints_a",
    "eval_keypoints_b",
    "skip_reason",
]


SUMMARY_FIELDS = [
    "strategy",
    "matcher",
    "ratio_thresh",
    "top_k",
    "mutual",
    "pair_count",
    "sift_inliers_mean",
    "hardnet_inliers_mean",
    "sift_unique_query_inliers_mean",
    "hardnet_unique_query_inliers_mean",
    "sift_one_to_one_inliers_mean",
    "hardnet_one_to_one_inliers_mean",
    "mean_inlier_delta",
    "mean_unique_query_delta",
    "mean_one_to_one_delta",
    "hardnet_better_inliers_rate",
    "hardnet_better_one_to_one_rate",
]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Fair RootSIFT vs HardNet matching on selected SIFT-hard image pairs.")
    parser.add_argument("--config", default="val_new/config.yaml", help="Config path.")
    parser.add_argument("--hard-cases-csv", default=None, help="Override hard cases CSV path.")
    parser.add_argument("--checkpoint", default=None, help="Override HardNet checkpoint path.")
    parser.add_argument("--output-dir", default=None, help="Override output.output_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint is not None:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
    if args.output_dir is not None:
        config.setdefault("output", {})["output_dir"] = str(Path(args.output_dir).expanduser().resolve())

    seed_everything(int(get_nested(config, "seed", default=42)))
    output_dir = ensure_dir(resolve_path(config, get_nested(config, "output", "output_dir")))
    fair_dir = ensure_dir(output_dir / "fair_matching")
    hard_cases_csv = Path(args.hard_cases_csv).expanduser().resolve() if args.hard_cases_csv else output_dir / "hard_cases" / "hard_cases.csv"
    if not hard_cases_csv.exists():
        raise FileNotFoundError(f"Hard cases CSV not found. Run select_sift_hard_cases first: {hard_cases_csv}")

    pairs = read_hard_case_pairs(hard_cases_csv)
    if not pairs:
        raise RuntimeError(f"No hard cases found in {hard_cases_csv}")

    strategies = [dict(strategy) for strategy in get_nested(config, "strategies", default=[])]
    if not strategies:
        raise RuntimeError("No strategies found in config.")

    sift = build_sift(config)
    hardnet = HardNetDescriptor(config)
    detail_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        prepared = prepare_pair(pair, config, sift, hardnet=hardnet)
        for strategy in strategies:
            detail_rows.append(evaluate_strategy(pair, prepared, strategy, config))
        if index % 10 == 0:
            print(f"evaluated hard cases: {index}", flush=True)

    details_path = write_csv(fair_dir / "fair_matching_details.csv", detail_rows, DETAIL_FIELDS)
    summary_rows = summarize(detail_rows)
    summary_csv_path = write_csv(fair_dir / "fair_matching_summary.csv", summary_rows, SUMMARY_FIELDS)
    summary_json_path = write_json(fair_dir / "fair_matching_summary.json", summary_rows)
    write_summary_plot(summary_rows, fair_dir / "fair_matching_summary.png")

    print(f"hard cases evaluated: {len(pairs)}")
    print(f"strategies: {len(strategies)}")
    print(f"wrote: {details_path}")
    print(f"wrote: {summary_csv_path}")
    print(f"wrote: {summary_json_path}")


def read_hard_case_pairs(path: Path) -> list[ImagePair]:
    """读取困难图像对 CSV。"""

    pairs: list[ImagePair] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pairs.append(
                ImagePair(
                    image_pair_id=row["image_pair_id"],
                    finger_id=row.get("finger_id", ""),
                    image_a_path=row["image_a_path"],
                    image_b_path=row["image_b_path"],
                )
            )
    return pairs


def evaluate_strategy(pair: ImagePair, prepared: Any, strategy: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """在同一个策略下同时评估 RootSIFT 和 HardNet。"""

    sift_result = evaluate_descriptor_pair(prepared, prepared.rootsift_a, prepared.rootsift_b, strategy, config)
    hardnet_result = evaluate_descriptor_pair(prepared, prepared.hardnet_a, prepared.hardnet_b, strategy, config)
    skip_reason = ";".join(sorted({reason for reason in [prepared.skip_reason, sift_result.skip_reason, hardnet_result.skip_reason] if reason}))
    return {
        "strategy": strategy_label(strategy),
        "matcher": strategy.get("matcher", ""),
        "ratio_thresh": strategy.get("ratio_thresh", ""),
        "top_k": strategy.get("top_k", ""),
        "mutual": int(bool(strategy.get("mutual", True))),
        "image_pair_id": pair.image_pair_id,
        "finger_id": pair.finger_id,
        "sift_raw_matches": sift_result.raw_matches,
        "sift_inliers": sift_result.inliers,
        "sift_unique_query_inliers": sift_result.unique_query_inliers,
        "sift_one_to_one_inliers": sift_result.one_to_one_inliers,
        "sift_mean_reproj_error": format_float(sift_result.mean_reproj_error),
        "hardnet_raw_matches": hardnet_result.raw_matches,
        "hardnet_inliers": hardnet_result.inliers,
        "hardnet_unique_query_inliers": hardnet_result.unique_query_inliers,
        "hardnet_one_to_one_inliers": hardnet_result.one_to_one_inliers,
        "hardnet_mean_reproj_error": format_float(hardnet_result.mean_reproj_error),
        "inlier_delta": hardnet_result.inliers - sift_result.inliers,
        "unique_query_delta": hardnet_result.unique_query_inliers - sift_result.unique_query_inliers,
        "one_to_one_delta": hardnet_result.one_to_one_inliers - sift_result.one_to_one_inliers,
        "hardnet_better_inliers": int(hardnet_result.inliers > sift_result.inliers),
        "hardnet_better_one_to_one": int(hardnet_result.one_to_one_inliers > sift_result.one_to_one_inliers),
        "eval_keypoints_a": len(prepared.keypoints_a),
        "eval_keypoints_b": len(prepared.keypoints_b),
        "skip_reason": skip_reason,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按策略汇总公平对比结果。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for strategy_name, group in grouped.items():
        first = group[0]
        sift_inliers = numeric(group, "sift_inliers")
        hardnet_inliers = numeric(group, "hardnet_inliers")
        sift_unique = numeric(group, "sift_unique_query_inliers")
        hardnet_unique = numeric(group, "hardnet_unique_query_inliers")
        sift_one = numeric(group, "sift_one_to_one_inliers")
        hardnet_one = numeric(group, "hardnet_one_to_one_inliers")
        summary_rows.append(
            {
                "strategy": strategy_name,
                "matcher": first["matcher"],
                "ratio_thresh": first["ratio_thresh"],
                "top_k": first["top_k"],
                "mutual": first["mutual"],
                "pair_count": len(group),
                "sift_inliers_mean": safe_float(np.mean(sift_inliers)),
                "hardnet_inliers_mean": safe_float(np.mean(hardnet_inliers)),
                "sift_unique_query_inliers_mean": safe_float(np.mean(sift_unique)),
                "hardnet_unique_query_inliers_mean": safe_float(np.mean(hardnet_unique)),
                "sift_one_to_one_inliers_mean": safe_float(np.mean(sift_one)),
                "hardnet_one_to_one_inliers_mean": safe_float(np.mean(hardnet_one)),
                "mean_inlier_delta": safe_float(np.mean(hardnet_inliers - sift_inliers)),
                "mean_unique_query_delta": safe_float(np.mean(hardnet_unique - sift_unique)),
                "mean_one_to_one_delta": safe_float(np.mean(hardnet_one - sift_one)),
                "hardnet_better_inliers_rate": safe_float(np.mean(hardnet_inliers > sift_inliers)),
                "hardnet_better_one_to_one_rate": safe_float(np.mean(hardnet_one > sift_one)),
            }
        )
    return summary_rows


def numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    """提取数值列。"""

    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def write_summary_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    """绘制 RootSIFT/HardNet 平均一对一内点数对比图。"""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not summary_rows:
        return
    labels = [str(row["strategy"]) for row in summary_rows]
    sift_values = [float(row["sift_one_to_one_inliers_mean"]) for row in summary_rows]
    hardnet_values = [float(row["hardnet_one_to_one_inliers_mean"]) for row in summary_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(9, len(labels) * 1.4), 5.0))
    plt.bar(x - 0.18, sift_values, width=0.36, label="RootSIFT one-to-one inliers")
    plt.bar(x + 0.18, hardnet_values, width=0.36, label="HardNet one-to-one inliers")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("mean one-to-one RANSAC inliers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

