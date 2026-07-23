"""单独查看 HardNet 内点匹配可视化。

本脚本用于从一个原始指纹图像目录中构建 HardNet 图像模板，并按 FAR/FRR
实验中的 genuine 匹配方式，导出“内点数最高”和“内点数最低但非 0”的匹配
case 可视化。

匹配参数直接读取 `config_match_new.yaml` 的 `matching` 配置，并复用
`match_new.hardnet_matcher.match_templates_descriptor_l2()`，因此候选生成、
Lowe ratio、RANSAC、unique inlier 去重等逻辑与正式 FAR/FRR 脚本保持一致。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.hardnet_matcher import match_templates_descriptor_l2
from match_new.template_builder import build_hardnet_templates, build_identity_templates, load_image_template
from match_new.utils import ensure_dir, load_config, read_csv_rows, resolve_path, safe_id, template_filename, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"
DEFAULT_IMAGE_ROOT = ROOT / "pair_build" / "select_top500"
DEFAULT_OUTPUT_DIR = MATCH_NEW_DIR / "hardnet_inlier_visuals"
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Visualize top/bottom HardNet unique-inlier genuine matches.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="match_new config YAML.")
    parser.add_argument("--image_root", default=str(DEFAULT_IMAGE_ROOT), help="Input fingerprint image directory.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--model_path", default=None, help="Override model.checkpoint in config.")
    parser.add_argument("--skip_template_build", action="store_true", help="Reuse existing output_dir/image_templates.")
    parser.add_argument("--enrollment_count", type=int, default=None, help="Override enrollment images per identity.")
    parser.add_argument("--random_seed", type=int, default=None, help="Override enrollment random seed.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of highest-unique-inlier cases to export.")
    parser.add_argument("--bottom_k", type=int, default=10, help="Number of lowest unique-inlier cases above the minimum threshold.")
    parser.add_argument(
        "--bottom_min_unique_inliers",
        type=int,
        default=2,
        help="Minimum unique_inliers for bottom cases. Default: 3.",
    )
    parser.add_argument("--max_lines", type=int, default=200, help="Maximum lines to draw per visualization.")
    parser.add_argument("--identity_depth", type=int, default=1, help="How many path levels under image_root form identity_id.")
    parser.add_argument(
        "--include_impostor",
        action="store_true",
        help="Also score query against other identities. Default only exports genuine matches.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """把命令行覆盖项写回 config，保持后续模板构建与正式实验一致。"""

    if args.model_path:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.model_path).expanduser().resolve())
    if args.random_seed is not None:
        config.setdefault("enrollment", {})["random_seed"] = int(args.random_seed)
    if args.enrollment_count is not None:
        config.setdefault("enrollment", {})["enrollment_images_per_identity"] = int(args.enrollment_count)


def scan_image_rows(image_root: str | Path, identity_depth: int) -> list[dict[str, str]]:
    """从原始图像目录扫描 metadata。

    默认 `identity_depth=1`，即 `pair_build/select_top500/zyh_L0/pair_1.bmp`
    会得到：
      identity_id = zyh_L0
      image_id = pair_1
      image_path = 原图绝对路径
    """

    root = Path(image_root).expanduser().resolve()
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) <= identity_depth:
            continue
        identity_id = "/".join(rel.parts[:identity_depth])
        rows.append(
            {
                "identity_id": identity_id,
                "image_id": path.stem,
                "image_path": str(path.resolve()),
                "split": "",
            }
        )
    return rows


def imread_grayscale(path: str | Path) -> np.ndarray | None:
    """读取灰度图，兼容 Windows 中文路径。"""

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_image(path: str | Path, image: np.ndarray) -> bool:
    """写图像，兼容 Windows 中文路径。"""

    target = Path(path)
    ensure_dir(target.parent)
    ok, encoded = cv2.imencode(target.suffix or ".png", image)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def copy_if_exists(src: str | Path, dst: str | Path) -> str:
    """复制原图到 case 目录；源文件不存在时返回空字符串。"""

    source = Path(src)
    if not source.exists():
        return ""
    target = Path(dst)
    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return str(target)


def load_template_cached(path: str | Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """读取 HardNet 模板并做内存缓存，避免同一注册模板被反复从磁盘加载。"""

    key = str(Path(path).expanduser())
    if key not in cache:
        cache[key] = load_image_template(key, require="hardnet")
    return cache[key]


def draw_one_line(canvas: np.ndarray, item: dict[str, Any], xoff: int, header: int, color: tuple[int, int, int], thickness: int) -> None:
    """在左右拼接图上画一条匹配线。"""

    qx, qy = item["query_xy"]
    gx, gy = item["gallery_xy"]
    p1 = (int(round(qx)), int(round(qy + header)))
    p2 = (int(round(gx + xoff)), int(round(gy + header)))
    cv2.circle(canvas, p1, 2, color, -1)
    cv2.circle(canvas, p2, 2, color, -1)
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def draw_match_lines(
    query_img: np.ndarray,
    gallery_img: np.ndarray,
    result: dict[str, Any],
    output_path: Path,
    title: str,
    match_key: str,
    max_lines: int,
) -> str:
    """绘制左右拼接的 HardNet 内点连线图。

    `match_key` 可取 `unique_inliers` 或 `raw_inliers`，二者都来自
    `include_debug=True` 的 matcher 输出。
    """

    q_bgr = cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
    g_bgr = cv2.cvtColor(gallery_img, cv2.COLOR_GRAY2BGR)
    header = 118
    gap = 16
    h = max(q_bgr.shape[0], g_bgr.shape[0])
    image_pair_width = q_bgr.shape[1] + gap + g_bgr.shape[1]
    w = max(image_pair_width, 900)
    canvas = np.full((h + header, w, 3), 255, dtype=np.uint8)
    canvas[header : header + q_bgr.shape[0], : q_bgr.shape[1]] = q_bgr
    xoff = q_bgr.shape[1] + gap
    canvas[header : header + g_bgr.shape[0], xoff : xoff + g_bgr.shape[1]] = g_bgr

    debug = result.get("debug_matches") or {}
    matches = list(debug.get(match_key) or [])[: max(0, int(max_lines))]
    color = (40, 180, 40) if match_key == "unique_inliers" else (180, 120, 40)
    thickness = 2 if match_key == "unique_inliers" else 1
    for item in matches:
        draw_one_line(canvas, item, xoff, header, color=color, thickness=thickness)

    info_lines = [
        title[:220],
        f"{match_key}: drawn={len(matches)} score={result.get('score', 0)} unique={result.get('unique_inliers', 0)} raw={result.get('raw_inliers', 0)}",
        f"mean_l2={float(result.get('mean_l2_distance', 0.0)):.4f} mean_reproj={float(result.get('mean_reproj_error', 0.0)):.4f}",
        "left: query    right: best enrollment template",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (8, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return str(output_path) if imwrite_image(output_path, canvas) else ""


def binary_on_overlap(image: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """只在 overlap 区域内做 Otsu 二值化。"""

    output = np.zeros_like(image, dtype=np.uint8)
    pixels = image[overlap > 0]
    if pixels.size == 0:
        return output
    threshold, _ = cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    output[(overlap > 0) & (image >= threshold)] = 255
    return output


def write_overlap_visuals(query_img: np.ndarray, gallery_img: np.ndarray, result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """根据 matcher 输出的 affine_matrix 写出 warp 和 overlap 图。"""

    matrix = result.get("affine_matrix")
    if matrix is None:
        return {}
    affine = np.asarray(matrix, dtype=np.float64)
    h, w = gallery_img.shape[:2]

    warped_query = cv2.warpAffine(query_img, affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    query_mask = np.ones(query_img.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(query_mask, affine, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gallery_mask = np.ones(gallery_img.shape[:2], dtype=np.uint8) * 255
    overlap = cv2.bitwise_and(warped_mask, gallery_mask)
    overlap = (overlap > 0).astype(np.uint8) * 255

    query_overlap = cv2.bitwise_and(warped_query, warped_query, mask=overlap)
    gallery_overlap = cv2.bitwise_and(gallery_img, gallery_img, mask=overlap)
    query_bin = binary_on_overlap(query_overlap, overlap)
    gallery_bin = binary_on_overlap(gallery_overlap, overlap)
    overlap_and = cv2.bitwise_and(query_bin, gallery_bin)
    overlap_xor = cv2.bitwise_xor(query_bin, gallery_bin)

    overlap_color = np.zeros((h, w, 3), dtype=np.uint8)
    overlap_color[query_bin > 0] = (0, 0, 220)
    overlap_color[gallery_bin > 0] = (0, 180, 0)
    overlap_color[(query_bin > 0) & (gallery_bin > 0)] = (0, 220, 220)

    paths = {
        "query_warped_to_gallery": str(output_dir / "query_warped_to_gallery.png"),
        "overlap_mask": str(output_dir / "overlap_mask.png"),
        "overlap_query_gray": str(output_dir / "overlap_query_gray.png"),
        "overlap_gallery_gray": str(output_dir / "overlap_gallery_gray.png"),
        "overlap_query_binary": str(output_dir / "overlap_query_binary.png"),
        "overlap_gallery_binary": str(output_dir / "overlap_gallery_binary.png"),
        "overlap_and": str(output_dir / "overlap_and.png"),
        "overlap_xor": str(output_dir / "overlap_xor.png"),
        "overlap_color": str(output_dir / "overlap_color.png"),
    }
    imwrite_image(paths["query_warped_to_gallery"], warped_query)
    imwrite_image(paths["overlap_mask"], overlap)
    imwrite_image(paths["overlap_query_gray"], query_overlap)
    imwrite_image(paths["overlap_gallery_gray"], gallery_overlap)
    imwrite_image(paths["overlap_query_binary"], query_bin)
    imwrite_image(paths["overlap_gallery_binary"], gallery_bin)
    imwrite_image(paths["overlap_and"], overlap_and)
    imwrite_image(paths["overlap_xor"], overlap_xor)
    imwrite_image(paths["overlap_color"], overlap_color)
    return paths


def result_without_debug(result: dict[str, Any]) -> dict[str, Any]:
    """JSON 中保存核心指标，避免 debug matches 太大。"""

    return {key: value for key, value in result.items() if key != "debug_matches"}


def score_genuine_pairs(
    split_rows: list[dict[str, str]],
    identity_payload: dict[str, Any],
    template_dir: Path,
    config: dict[str, Any],
    include_impostor: bool,
) -> list[dict[str, Any]]:
    """按 FAR/FRR 的方式匹配 query 与注册模板，并返回每个 query 的最佳命中。"""

    identity_by_id = {str(item["identity_id"]): item for item in identity_payload.get("identities", [])}
    identities = list(identity_by_id.values())
    query_rows = [row for row in split_rows if str(row.get("split", "")).lower() == "query"]
    cache: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    for row in query_rows:
        query_identity = row["identity_id"]
        query_id = row["image_id"]
        query_template_path = template_dir / template_filename(query_identity, query_id)
        if not query_template_path.exists():
            continue
        query_template = load_template_cached(query_template_path, cache)
        owner_candidates = identities if include_impostor else [identity_by_id.get(query_identity)]
        for owner in owner_candidates:
            if not owner:
                continue
            label = "genuine" if str(owner["identity_id"]) == query_identity else "impostor"
            best_result: dict[str, Any] | None = None
            best_template: dict[str, Any] | None = None
            best_template_path = ""
            for template_path in owner.get("template_paths", []):
                gallery_template = load_template_cached(template_path, cache)
                result = match_templates_descriptor_l2(
                    query_template,
                    gallery_template,
                    config,
                    descriptor_source="hardnet",
                    include_debug=True,
                )
                if best_result is None or (
                    float(result.get("score", 0.0)),
                    float(result.get("quality_score", 0.0)),
                    int(result.get("unique_inliers", 0)),
                ) > (
                    float(best_result.get("score", 0.0)),
                    float(best_result.get("quality_score", 0.0)),
                    int(best_result.get("unique_inliers", 0)),
                ):
                    best_result = result
                    best_template = gallery_template
                    best_template_path = str(template_path)
            if best_result is None or best_template is None:
                continue
            cases.append(
                {
                    "label": label,
                    "query_identity": query_identity,
                    "owner_identity": str(owner["identity_id"]),
                    "query_id": query_id,
                    "query_image_path": str(query_template.get("image_path", "")),
                    "query_template_path": str(query_template_path),
                    "best_template_id": str(best_template.get("image_id", "")),
                    "best_template_image_path": str(best_template.get("image_path", "")),
                    "best_template_path": best_template_path,
                    "score": float(best_result.get("score", 0.0)),
                    "quality_score": float(best_result.get("quality_score", 0.0)),
                    "unique_inliers": int(best_result.get("unique_inliers", 0)),
                    "raw_inliers": int(best_result.get("raw_inliers", 0)),
                    "num_candidates": int(best_result.get("num_candidates", 0)),
                    "num_raw_matches": int(best_result.get("num_raw_matches", 0)),
                    "inlier_ratio": float(best_result.get("inlier_ratio", 0.0)),
                    "mean_l2_distance": float(best_result.get("mean_l2_distance", 0.0)),
                    "mean_reproj_error": float(best_result.get("mean_reproj_error", 0.0)),
                    "orientation_consistency": float(best_result.get("orientation_consistency", 0.0)),
                    "dominant_angle_delta": float(best_result.get("dominant_angle_delta", 0.0)),
                    "match_result": best_result,
                }
            )
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    top_k: int,
    bottom_k: int,
    bottom_min_unique_inliers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """选择 unique_inliers 最高和最低的 case。

    bottom 组要求 unique_inliers 不低于 bottom_min_unique_inliers。
    默认阈值为 3，避免导出几乎无法估计可靠几何关系的极弱匹配。
    """

    top_cases = sorted(
        cases,
        key=lambda item: (
            int(item.get("unique_inliers", 0)),
            float(item.get("score", 0.0)),
            float(item.get("quality_score", 0.0)),
            str(item.get("query_identity", "")),
            str(item.get("query_id", "")),
        ),
        reverse=True,
    )[: max(0, int(top_k))]
    min_unique = max(0, int(bottom_min_unique_inliers))
    positive = [case for case in cases if int(case.get("unique_inliers", 0)) >= min_unique]
    bottom_cases = sorted(
        positive,
        key=lambda item: (
            int(item.get("unique_inliers", 0)),
            float(item.get("score", 0.0)),
            float(item.get("quality_score", 0.0)),
            str(item.get("query_identity", "")),
            str(item.get("query_id", "")),
        ),
    )[: max(0, int(bottom_k))]
    return top_cases, bottom_cases


def export_case(case: dict[str, Any], group: str, rank: int, output_root: Path, max_lines: int) -> dict[str, Any] | None:
    """导出单个 case 的连线图、overlap 图和 JSON。"""

    query_img = imread_grayscale(case["query_image_path"])
    gallery_img = imread_grayscale(case["best_template_image_path"])
    if query_img is None or gallery_img is None:
        return None

    case_name = (
        f"{rank:03d}__{safe_id(case['label'])}__{safe_id(case['query_identity'])}"
        f"__q_{safe_id(case['query_id'])}__tmpl_{safe_id(case['best_template_id'])}"
        f"__u{int(case['unique_inliers'])}"
    )
    case_dir = ensure_dir(output_root / group / case_name)
    query_copy = copy_if_exists(case["query_image_path"], case_dir / f"query{Path(case['query_image_path']).suffix or '.png'}")
    gallery_copy = copy_if_exists(case["best_template_image_path"], case_dir / f"gallery{Path(case['best_template_image_path']).suffix or '.png'}")
    result = case["match_result"]
    title = (
        f"{group} | {case['label']} | query={case['query_identity']}/{case['query_id']} "
        f"gallery={case['owner_identity']}/{case['best_template_id']}"
    )
    unique_path = draw_match_lines(query_img, gallery_img, result, case_dir / "match_lines_unique.png", title, "unique_inliers", max_lines)
    raw_path = draw_match_lines(query_img, gallery_img, result, case_dir / "match_lines_raw.png", title, "raw_inliers", max_lines)
    overlap_paths = write_overlap_visuals(query_img, gallery_img, result, case_dir)

    payload = {
        **{key: value for key, value in case.items() if key != "match_result"},
        "case_group": group,
        "rank_in_group": rank,
        "case_dir": str(case_dir),
        "query_copy": query_copy,
        "gallery_copy": gallery_copy,
        "match_lines_unique": unique_path,
        "match_lines_raw": raw_path,
        "overlap_outputs": overlap_paths,
        "match_result": result_without_debug(result),
        "debug_counts": {
            "candidates": len((result.get("debug_matches") or {}).get("candidates") or []),
            "raw_inliers": len((result.get("debug_matches") or {}).get("raw_inliers") or []),
            "unique_inliers": len((result.get("debug_matches") or {}).get("unique_inliers") or []),
        },
    }
    write_json(case_dir / "match_result.json", payload)
    return payload


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    checkpoint = resolve_path(config, dict(config.get("model", {}))["checkpoint"])
    if not checkpoint.exists() and not args.skip_template_build:
        raise FileNotFoundError(f"HardNet checkpoint does not exist: {checkpoint}")

    output_dir = ensure_dir(args.output_dir)
    template_dir = ensure_dir(output_dir / "image_templates")
    rows = scan_image_rows(args.image_root, int(args.identity_depth))
    if not rows:
        raise RuntimeError(f"No images found under {args.image_root}")
    write_csv_rows(output_dir / "metadata_all.csv", rows)

    metadata_success_path = output_dir / "metadata_success.csv"
    if args.skip_template_build:
        if not metadata_success_path.exists():
            raise RuntimeError("--skip_template_build requires output_dir/metadata_success.csv")
        success_rows = read_csv_rows(metadata_success_path)
    else:
        report = build_hardnet_templates(rows, template_dir, config)
        write_json(output_dir / "build_report.json", {key: value for key, value in report.items() if key != "success_rows"})
        write_csv_rows(output_dir / "template_build_timings.csv", report.get("template_timings", []))
        success_rows = report["success_rows"]
        if not success_rows:
            raise RuntimeError("No templates were built successfully.")
        write_csv_rows(metadata_success_path, success_rows)

    enrollment = dict(config.get("enrollment", {}))
    enrollment_count = int(enrollment.get("enrollment_images_per_identity", 20))
    random_seed = int(enrollment.get("random_seed", 42))
    identity_templates_path = output_dir / f"identity_templates_{enrollment_count}.json"
    split_metadata_path = output_dir / f"metadata_with_split_{enrollment_count}.csv"
    identity_payload = build_identity_templates(
        success_rows,
        identity_templates_path,
        split_metadata_path,
        enrollment_count=enrollment_count,
        seed=random_seed,
    )
    split_rows = read_csv_rows(split_metadata_path)

    cases = score_genuine_pairs(
        split_rows=split_rows,
        identity_payload=identity_payload,
        template_dir=template_dir,
        config=config,
        include_impostor=bool(args.include_impostor),
    )
    if not cases:
        raise RuntimeError("No match cases were produced. Try lowering --enrollment_count.")

    top_cases, bottom_cases = select_cases(
        cases,
        int(args.top_k),
        int(args.bottom_k),
        int(args.bottom_min_unique_inliers),
    )
    exported: list[dict[str, Any]] = []
    for index, case in enumerate(top_cases, start=1):
        payload = export_case(case, "top_unique_inliers", index, output_dir, int(args.max_lines))
        if payload is not None:
            exported.append(payload)
    for index, case in enumerate(bottom_cases, start=1):
        payload = export_case(case, "bottom_unique_inliers_nonzero", index, output_dir, int(args.max_lines))
        if payload is not None:
            exported.append(payload)

    csv_rows = [{key: value for key, value in row.items() if key not in {"match_result", "overlap_outputs"}} for row in exported]
    write_csv_rows(output_dir / "selected_cases.csv", csv_rows)
    write_json(
        output_dir / "summary.json",
        {
            "image_root": str(Path(args.image_root).expanduser().resolve()),
            "output_dir": str(output_dir.resolve()),
            "config": str(Path(args.config).expanduser().resolve()),
            "matching_config": config.get("matching", {}),
            "enrollment_count": enrollment_count,
            "random_seed": random_seed,
            "num_images": len(rows),
            "num_templates": len(success_rows),
            "num_cases_scored": len(cases),
            "num_positive_unique_inlier_cases": sum(1 for case in cases if int(case.get("unique_inliers", 0)) > 0),
            "top_k": int(args.top_k),
            "bottom_k": int(args.bottom_k),
            "bottom_min_unique_inliers": int(args.bottom_min_unique_inliers),
            "num_exported_cases": len(exported),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "num_scored_cases": len(cases), "num_exported_cases": len(exported)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
