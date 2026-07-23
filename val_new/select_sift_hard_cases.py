"""选择 SIFT/RootSIFT 表现差的困难图像对。

作用：
    1. 从 `pairs_csv` 中按 `image_pair_id` 去重读取图像对。
    2. 使用 RootSIFT 和指定的 SIFT 选择策略做图像级匹配。
    3. 按 RootSIFT RANSAC 内点数从少到多排序，选出 bottom_k 个困难样本。
    4. 把这些困难样本对应的原图复制出来，并生成 RootSIFT 内点连线预览图。

输入：
    - `val_new/config.yaml`
    - `outputs/hardnet_dataset/test_pairs.csv`

输出：
    - `outputs/val_new_eval/hard_cases/all_sift_case_scores.csv`
      所有扫描图像对的 SIFT 表现。
    - `outputs/val_new_eval/hard_cases/hard_cases.csv`
      选出的困难图像对清单。
    - `outputs/val_new_eval/hard_cases/case_xxx/`
      每个困难图像对的原图拷贝和 RootSIFT 匹配预览图。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from val_new.core import (
    ImagePair,
    build_sift,
    copy_file,
    draw_inlier_preview,
    ensure_dir,
    evaluate_descriptor_pair,
    get_nested,
    load_config,
    prepare_pair,
    read_image_pairs,
    resolve_path,
    seed_everything,
    strategy_label,
    write_csv,
    write_json,
)


ALL_SCORE_FIELDS = [
    "rank_by_sift",
    "image_pair_id",
    "finger_id",
    "image_a_path",
    "image_b_path",
    "original_keypoints_a",
    "original_keypoints_b",
    "eval_keypoints_a",
    "eval_keypoints_b",
    "sift_raw_matches",
    "sift_inliers",
    "sift_unique_query_inliers",
    "sift_one_to_one_inliers",
    "sift_mean_reproj_error",
    "skip_reason",
]


HARD_CASE_FIELDS = [
    *ALL_SCORE_FIELDS,
    "case_dir",
    "copied_image_a_path",
    "copied_image_b_path",
    "sift_inlier_preview_path",
]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Select image pairs where RootSIFT/SIFT has low inlier count.")
    parser.add_argument("--config", default="val_new/config.yaml", help="Config path.")
    parser.add_argument("--pairs-csv", default=None, help="Override data.pairs_csv.")
    parser.add_argument("--output-dir", default=None, help="Override output.output_dir.")
    parser.add_argument("--bottom-k", type=int, default=None, help="How many low-SIFT-inlier image pairs to select.")
    parser.add_argument("--max-image-pairs", type=int, default=None, help="Debug limit for scanned unique image pairs.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pairs_csv is not None:
        config.setdefault("data", {})["pairs_csv"] = str(Path(args.pairs_csv).expanduser().resolve())
    if args.output_dir is not None:
        config.setdefault("output", {})["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    if args.bottom_k is not None:
        config.setdefault("hard_case", {})["bottom_k"] = int(args.bottom_k)
    if args.max_image_pairs is not None:
        config.setdefault("hard_case", {})["max_image_pairs"] = int(args.max_image_pairs)

    seed_everything(int(get_nested(config, "seed", default=42)))
    output_dir = ensure_dir(resolve_path(config, get_nested(config, "output", "output_dir")))
    hard_case_dir = ensure_dir(output_dir / "hard_cases")
    pairs_csv = resolve_path(config, get_nested(config, "data", "pairs_csv"))
    bottom_k = int(get_nested(config, "hard_case", "bottom_k", default=50))
    max_image_pairs = get_nested(config, "hard_case", "max_image_pairs", default=None)
    max_image_pairs = int(max_image_pairs) if max_image_pairs not in [None, ""] else None

    strategy = selection_strategy(config)
    sift = build_sift(config)
    image_pairs = read_image_pairs(pairs_csv, max_image_pairs=max_image_pairs)
    if not image_pairs:
        raise RuntimeError(f"No image pairs found in {pairs_csv}")

    score_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(image_pairs, start=1):
        prepared = prepare_pair(pair, config, sift, hardnet=None)
        result = evaluate_descriptor_pair(prepared, prepared.rootsift_a, prepared.rootsift_b, strategy, config)
        score_rows.append(score_row(index, pair, prepared, result))
        if index % 50 == 0:
            print(f"scanned image pairs: {index}", flush=True)

    ranked_rows = sorted(
        score_rows,
        key=lambda row: (
            int(row["sift_inliers"]),
            int(row["sift_raw_matches"]),
            int(row["eval_keypoints_a"]) + int(row["eval_keypoints_b"]),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_by_sift"] = rank

    all_scores_path = write_csv(hard_case_dir / "all_sift_case_scores.csv", ranked_rows, ALL_SCORE_FIELDS)
    selected_rows = ranked_rows[:bottom_k]
    hard_case_rows = materialize_hard_cases(selected_rows, config, sift, strategy, hard_case_dir)
    hard_cases_path = write_csv(hard_case_dir / "hard_cases.csv", hard_case_rows, HARD_CASE_FIELDS)

    summary = {
        "scanned_image_pairs": len(score_rows),
        "selected_hard_cases": len(hard_case_rows),
        "bottom_k": bottom_k,
        "selection_strategy": strategy_label(strategy),
        "all_scores_path": str(all_scores_path),
        "hard_cases_path": str(hard_cases_path),
    }
    summary_path = write_json(hard_case_dir / "hard_case_selection_summary.json", summary)

    print(f"scanned image pairs: {len(score_rows)}")
    print(f"selected hard cases: {len(hard_case_rows)}")
    print(f"wrote: {all_scores_path}")
    print(f"wrote: {hard_cases_path}")
    print(f"wrote: {summary_path}")


def selection_strategy(config: dict[str, Any]) -> dict[str, Any]:
    """读取用于定义“SIFT 表现差”的匹配策略。

    默认使用 ratio=0.85 + mutual=true 的经典 RootSIFT 策略。
    """

    name = str(get_nested(config, "hard_case", "selection_strategy", default="ratio_0.85_two_way"))
    for strategy in get_nested(config, "strategies", default=[]):
        if str(strategy.get("name", "")) == name:
            return dict(strategy)
    return {"name": name, "matcher": "ratio", "ratio_thresh": 0.85, "mutual": True}


def score_row(index: int, pair: ImagePair, prepared: Any, result: Any) -> dict[str, Any]:
    """把单个图像对的 SIFT 结果转换成 CSV 行。"""

    skip_reason = prepared.skip_reason or result.skip_reason
    return {
        "rank_by_sift": index,
        "image_pair_id": pair.image_pair_id,
        "finger_id": pair.finger_id,
        "image_a_path": pair.image_a_path,
        "image_b_path": pair.image_b_path,
        "original_keypoints_a": prepared.original_keypoints_a,
        "original_keypoints_b": prepared.original_keypoints_b,
        "eval_keypoints_a": len(prepared.keypoints_a),
        "eval_keypoints_b": len(prepared.keypoints_b),
        "sift_raw_matches": result.raw_matches,
        "sift_inliers": result.inliers,
        "sift_unique_query_inliers": result.unique_query_inliers,
        "sift_one_to_one_inliers": result.one_to_one_inliers,
        "sift_mean_reproj_error": "" if result.mean_reproj_error != result.mean_reproj_error else f"{result.mean_reproj_error:.6f}",
        "skip_reason": skip_reason,
    }


def materialize_hard_cases(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    sift: Any,
    strategy: dict[str, Any],
    hard_case_dir: Path,
) -> list[dict[str, Any]]:
    """复制困难图像对原图，并生成 RootSIFT 内点连线预览图。"""

    output_rows: list[dict[str, Any]] = []
    max_inliers = int(get_nested(config, "hard_case", "preview_max_inliers", default=80))
    for rank, row in enumerate(rows, start=1):
        pair = ImagePair(
            image_pair_id=str(row["image_pair_id"]),
            finger_id=str(row["finger_id"]),
            image_a_path=str(row["image_a_path"]),
            image_b_path=str(row["image_b_path"]),
        )
        case_dir = ensure_dir(hard_case_dir / f"case_{rank:03d}_sift_inliers_{int(row['sift_inliers']):03d}_{short_token(pair.image_pair_id)}")
        copied_a = case_dir / f"image_a{Path(pair.image_a_path).suffix or '.bmp'}"
        copied_b = case_dir / f"image_b{Path(pair.image_b_path).suffix or '.bmp'}"
        preview = case_dir / "rootsift_inlier_preview.png"
        copy_file(pair.image_a_path, copied_a)
        copy_file(pair.image_b_path, copied_b)

        prepared = prepare_pair(pair, config, sift, hardnet=None)
        result = evaluate_descriptor_pair(prepared, prepared.rootsift_a, prepared.rootsift_b, strategy, config)
        draw_inlier_preview(pair, prepared, result, preview, max_inliers=max_inliers)
        output_rows.append(
            {
                **row,
                "case_dir": str(case_dir),
                "copied_image_a_path": str(copied_a),
                "copied_image_b_path": str(copied_b),
                "sift_inlier_preview_path": str(preview),
            }
        )
    return output_rows


def short_token(text: str) -> str:
    """生成适合目录名的短 token。"""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned[:42] or "pair"


if __name__ == "__main__":
    main()

