"""val_new 公共工具。

作用：
    本文件集中放置新验证项目复用的底层逻辑，包括：
    - 配置读取与相对路径解析。
    - Windows 中文路径下的图像读写。
    - RootSIFT 和 HardNet 描述子计算。
    - 与训练阶段一致的 HardNet patch 裁剪。
    - ratio / top-k 匹配策略。
    - RANSAC 几何验证和内点统计。

输入：
    - `val_new/config.yaml`。
    - `outputs/hardnet_dataset/test_pairs.csv` 中的图像对路径。
    - HardNet checkpoint。

输出：
    - 匹配评估结果字典。
    - 可写入 CSV/JSON 的统计值。
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardnet_train.model import HardNet
from patch_sampling import patchable_keypoints


@dataclass
class ImagePair:
    """一对待评估指纹图像的轻量记录。"""

    image_pair_id: str
    finger_id: str
    image_a_path: str
    image_b_path: str


@dataclass
class PreparedPair:
    """同一批 keypoints 上的 RootSIFT/HardNet 描述子。

    RootSIFT 和 HardNet 的 descriptor 行号都与 keypoints/points 对齐。
    这样公平对比时只替换 descriptor，不改变关键点集合。
    """

    pair: ImagePair
    keypoints_a: list[cv2.KeyPoint]
    keypoints_b: list[cv2.KeyPoint]
    points_a: np.ndarray
    points_b: np.ndarray
    rootsift_a: np.ndarray
    rootsift_b: np.ndarray
    hardnet_a: np.ndarray
    hardnet_b: np.ndarray
    original_keypoints_a: int
    original_keypoints_b: int
    skip_reason: str = ""


@dataclass
class MatchResult:
    """一次 descriptor 匹配和 RANSAC 验证的结果。"""

    raw_matches: int
    inliers: int
    unique_query_inliers: int
    unique_train_inliers: int
    one_to_one_inliers: int
    mean_reproj_error: float
    transform: list[list[float]] | None
    skip_reason: str
    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML/JSON 配置，并记录配置文件绝对路径。"""

    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def get_nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """安全读取多层配置字段。"""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def resolve_path(config: Mapping[str, Any], raw_path: str | Path) -> Path:
    """把配置文件里的相对路径解析为绝对路径。"""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(str(config["_config_path"])).resolve()
    return (config_path.parent / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。"""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def seed_everything(seed: int) -> random.Random:
    """设置随机种子，并返回独立随机数生成器。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def choose_device(raw_device: str) -> torch.device:
    """解析 auto/cpu/cuda 设备配置。"""

    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


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
    suffix = target.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def copy_file(src: str | Path, dst: str | Path) -> None:
    """复制原图到输出目录，便于人工分析。"""

    target = Path(dst)
    ensure_dir(target.parent)
    shutil.copy2(str(src), str(target))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    """写 CSV。"""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """写 JSON。"""

    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_image_pairs(csv_path: Path, max_image_pairs: int | None = None) -> list[ImagePair]:
    """从 pair CSV 中按 image_pair_id 去重读取图像对。

    CSV 一行是一个正样本 patch 对；一个 image_pair 会有很多行。
    图像级评估只需要每个 image_pair 的原图路径，所以这里先去重。
    """

    pairs: list[ImagePair] = []
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_pair_id = row.get("image_pair_id", "")
            if not image_pair_id or image_pair_id in seen:
                continue
            seen.add(image_pair_id)
            pairs.append(
                ImagePair(
                    image_pair_id=image_pair_id,
                    finger_id=row.get("finger_id", ""),
                    image_a_path=row.get("image_a_path", ""),
                    image_b_path=row.get("image_b_path", ""),
                )
            )
            if max_image_pairs is not None and len(pairs) >= int(max_image_pairs):
                break
    return pairs


def build_sift(config: Mapping[str, Any]) -> cv2.SIFT:
    """根据配置创建 OpenCV SIFT。"""

    return cv2.SIFT_create(
        nfeatures=int(get_nested(config, "sift", "nfeatures", default=300)),
        nOctaveLayers=int(get_nested(config, "sift", "nOctaveLayers", default=4)),
        contrastThreshold=float(get_nested(config, "sift", "contrastThreshold", default=0.03)),
        edgeThreshold=float(get_nested(config, "sift", "edgeThreshold", default=17.5)),
        sigma=float(get_nested(config, "sift", "sigma", default=1.70)),
    )


def preprocess_for_sift(image: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """SIFT 检测前预处理，默认和 pair_build 保持一致。"""

    output = image
    if bool(get_nested(config, "sift", "enable_clahe", default=True)):
        tile = get_nested(config, "sift", "clahe_tile", default=[8, 8])
        clahe = cv2.createCLAHE(
            clipLimit=float(get_nested(config, "sift", "clahe_clip", default=4.0)),
            tileGridSize=(int(tile[0]), int(tile[1])),
        )
        output = clahe.apply(output)
    if bool(get_nested(config, "sift", "enable_blur", default=True)):
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=float(get_nested(config, "sift", "blur_sigma", default=0.8)))
    return output


def rootsift(descriptors: np.ndarray | None) -> np.ndarray:
    """把 SIFT descriptor 转成 RootSIFT。"""

    if descriptors is None or len(descriptors) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    values = descriptors.astype(np.float32)
    values /= values.sum(axis=1, keepdims=True) + 1e-12
    values = np.sqrt(values)
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
    return values.astype(np.float32)


def detect_sift_keypoints(image: np.ndarray, sift: cv2.SIFT, config: Mapping[str, Any]) -> list[cv2.KeyPoint]:
    """只检测 SIFT keypoints，不计算描述子。

    HardNet 主流程只需要关键点位置和方向来裁 patch，不需要 SIFT/RootSIFT 描述子。
    """

    processed = preprocess_for_sift(image, config)
    keypoints = sift.detect(processed, None)
    return list(keypoints or [])


def detect_rootsift(image: np.ndarray, sift: cv2.SIFT, config: Mapping[str, Any]) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """检测 SIFT keypoints 并计算 RootSIFT descriptor。

    仅供独立 SIFT/RootSIFT 对照实验使用；HardNet 主流程应调用 detect_sift_keypoints。
    """

    processed = preprocess_for_sift(image, config)
    keypoints, descriptors = sift.detectAndCompute(processed, None)
    keypoints = list(keypoints or [])
    descriptors = rootsift(descriptors) if bool(get_nested(config, "sift", "enable_rootsift", default=True)) else np.asarray(descriptors, dtype=np.float32)
    return keypoints, descriptors


def keypoints_to_points(keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """把 keypoints 转成 RANSAC 坐标数组。"""

    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)


class HardNetDescriptor:
    """HardNet checkpoint 加载与批量推理封装。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.device = choose_device(str(get_nested(config, "model", "device", default="auto")))
        self.batch_size = int(get_nested(config, "model", "batch_size", default=512))
        torch.backends.cudnn.benchmark = False
        checkpoint_path = resolve_path(config, get_nested(config, "model", "checkpoint"))
        self.model = HardNet(
            dropout=float(get_nested(config, "model", "dropout", default=0.1)),
            final_bn_affine=bool(get_nested(config, "model", "final_bn_affine", default=False)),
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def describe(self, patches: np.ndarray) -> np.ndarray:
        """计算 `[N, 32, 32]` 或 `[N, 1, 32, 32]` patch 的 HardNet 描述子。"""

        if len(patches) == 0:
            return np.zeros((0, 128), dtype=np.float32)
        patch_array = patches.astype(np.float32)
        if patch_array.ndim == 3:
            patch_array = patch_array[:, None, :, :]
        outputs: list[np.ndarray] = []
        for start in range(0, len(patch_array), self.batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(patch_array[start : start + self.batch_size])).to(self.device)
            outputs.append(self.model(batch).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 128), dtype=np.float32)


def prepare_pair(
    pair: ImagePair,
    config: Mapping[str, Any],
    sift: cv2.SIFT,
    hardnet: HardNetDescriptor | None = None,
) -> PreparedPair:
    """读取图像并准备公平对比所需的 keypoints 和描述子。

    如果 hardnet=None，只准备 RootSIFT，用于第一阶段快速挑选 SIFT 困难样本。
    """

    image_a = imread_grayscale(pair.image_a_path)
    image_b = imread_grayscale(pair.image_b_path)
    if image_a is None or image_b is None:
        return empty_prepared_pair(pair, "image_read_failed")

    keypoints_a_all, rootsift_a_all = detect_rootsift(image_a, sift, config)
    keypoints_b_all, rootsift_b_all = detect_rootsift(image_b, sift, config)
    if len(keypoints_a_all) == 0 or len(keypoints_b_all) == 0:
        return empty_prepared_pair(pair, "no_keypoints", len(keypoints_a_all), len(keypoints_b_all))

    keypoints_a, patches_a, indices_a = patchable_keypoints(image_a, keypoints_a_all, config)
    keypoints_b, patches_b, indices_b = patchable_keypoints(image_b, keypoints_b_all, config)
    if len(keypoints_a) == 0 or len(keypoints_b) == 0:
        return empty_prepared_pair(pair, "no_patchable_keypoints", len(keypoints_a_all), len(keypoints_b_all))

    rootsift_a = rootsift_a_all[np.asarray(indices_a, dtype=int)]
    rootsift_b = rootsift_b_all[np.asarray(indices_b, dtype=int)]
    hardnet_a = hardnet.describe(patches_a) if hardnet is not None else np.zeros((0, 128), dtype=np.float32)
    hardnet_b = hardnet.describe(patches_b) if hardnet is not None else np.zeros((0, 128), dtype=np.float32)

    return PreparedPair(
        pair=pair,
        keypoints_a=keypoints_a,
        keypoints_b=keypoints_b,
        points_a=keypoints_to_points(keypoints_a),
        points_b=keypoints_to_points(keypoints_b),
        rootsift_a=rootsift_a,
        rootsift_b=rootsift_b,
        hardnet_a=hardnet_a,
        hardnet_b=hardnet_b,
        original_keypoints_a=len(keypoints_a_all),
        original_keypoints_b=len(keypoints_b_all),
    )


def empty_prepared_pair(
    pair: ImagePair,
    reason: str,
    keypoints_a: int = 0,
    keypoints_b: int = 0,
) -> PreparedPair:
    """构造失败状态的 PreparedPair。"""

    return PreparedPair(
        pair=pair,
        keypoints_a=[],
        keypoints_b=[],
        points_a=np.zeros((0, 2), dtype=np.float32),
        points_b=np.zeros((0, 2), dtype=np.float32),
        rootsift_a=np.zeros((0, 128), dtype=np.float32),
        rootsift_b=np.zeros((0, 128), dtype=np.float32),
        hardnet_a=np.zeros((0, 128), dtype=np.float32),
        hardnet_b=np.zeros((0, 128), dtype=np.float32),
        original_keypoints_a=keypoints_a,
        original_keypoints_b=keypoints_b,
        skip_reason=reason,
    )


def ratio_matches(desc_a: np.ndarray, desc_b: np.ndarray, ratio_thresh: float, mutual: bool) -> list[cv2.DMatch]:
    """Lowe ratio test，可选双向一致性。"""

    if len(desc_a) == 0 or len(desc_b) < 2:
        return []
    if mutual and len(desc_a) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = one_way_ratio_matches(matcher, desc_a, desc_b, ratio_thresh)
    if not mutual:
        return forward
    backward = one_way_ratio_matches(matcher, desc_b, desc_a, ratio_thresh)
    reciprocal = {(match.trainIdx, match.queryIdx) for match in backward}
    return [match for match in forward if (match.queryIdx, match.trainIdx) in reciprocal]


def one_way_ratio_matches(matcher: cv2.BFMatcher, desc_a: np.ndarray, desc_b: np.ndarray, ratio_thresh: float) -> list[cv2.DMatch]:
    """单向 ratio test。"""

    knn = matcher.knnMatch(desc_a.astype(np.float32), desc_b.astype(np.float32), k=2)
    matches: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if float(best.distance / max(second.distance, 1e-12)) <= ratio_thresh:
            matches.append(best)
    return matches


def topk_matches(desc_a: np.ndarray, desc_b: np.ndarray, top_k: int, mutual: bool) -> list[cv2.DMatch]:
    """top-k 候选匹配，可选双向 top-k 互检。"""

    if len(desc_a) == 0 or len(desc_b) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    k_ab = min(int(top_k), len(desc_b))
    forward_groups = matcher.knnMatch(desc_a.astype(np.float32), desc_b.astype(np.float32), k=k_ab)
    forward = [match for group in forward_groups for match in group]
    if not mutual:
        return forward
    k_ba = min(int(top_k), len(desc_a))
    backward_groups = matcher.knnMatch(desc_b.astype(np.float32), desc_a.astype(np.float32), k=k_ba)
    reciprocal = {(match.trainIdx, match.queryIdx) for group in backward_groups for match in group}
    return [match for match in forward if (match.queryIdx, match.trainIdx) in reciprocal]


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray, strategy: Mapping[str, Any]) -> list[cv2.DMatch]:
    """按策略生成粗匹配候选。"""

    matcher_type = str(strategy.get("matcher", "ratio"))
    mutual = bool(strategy.get("mutual", True))
    if matcher_type == "ratio":
        return ratio_matches(desc_a, desc_b, ratio_thresh=float(strategy.get("ratio_thresh", 0.85)), mutual=mutual)
    if matcher_type == "topk":
        return topk_matches(desc_a, desc_b, top_k=int(strategy.get("top_k", 2)), mutual=mutual)
    raise ValueError(f"Unsupported matcher strategy: {matcher_type}")


def estimate_inliers(
    points_a: np.ndarray,
    points_b: np.ndarray,
    matches: Sequence[cv2.DMatch],
    config: Mapping[str, Any],
) -> MatchResult:
    """RANSAC 估计 partial affine 并统计多种内点指标。"""

    min_matches = int(get_nested(config, "ransac", "min_matches", default=3))
    reproj_thresh = float(get_nested(config, "ransac", "reproj_thresh", default=1.0))
    if len(matches) < min_matches:
        return MatchResult(len(matches), 0, 0, 0, 0, float("nan"), None, "not_enough_matches", list(matches), np.zeros((len(matches),), dtype=bool))

    src = np.asarray([points_a[match.queryIdx] for match in matches], dtype=np.float32)
    dst = np.asarray([points_b[match.trainIdx] for match in matches], dtype=np.float32)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return MatchResult(len(matches), 0, 0, 0, 0, float("nan"), None, "ransac_failed", list(matches), np.zeros((len(matches),), dtype=bool))

    mask = inlier_mask.reshape(-1).astype(bool)
    errors = reprojection_errors(src, dst, matrix)
    inlier_matches = [match for match, is_inlier in zip(matches, mask) if bool(is_inlier)]
    return MatchResult(
        raw_matches=len(matches),
        inliers=int(np.sum(mask)),
        unique_query_inliers=len({match.queryIdx for match in inlier_matches}),
        unique_train_inliers=len({match.trainIdx for match in inlier_matches}),
        one_to_one_inliers=count_one_to_one(inlier_matches),
        mean_reproj_error=float(np.mean(errors[mask])) if np.any(mask) else float("nan"),
        transform=np.asarray(matrix, dtype=float).tolist(),
        skip_reason="",
        matches=list(matches),
        inlier_mask=mask,
    )


def reprojection_errors(src: np.ndarray, dst: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算重投影误差。"""

    ones = np.ones((len(src), 1), dtype=np.float32)
    projected = np.concatenate([src, ones], axis=1) @ matrix.T
    return np.linalg.norm(projected - dst, axis=1)


def count_one_to_one(matches: Sequence[cv2.DMatch]) -> int:
    """对 RANSAC 内点做贪心一对一去重计数。

    top-k 会允许一对多候选。正式比较时，一对一内点数更保守。
    """

    used_query: set[int] = set()
    used_train: set[int] = set()
    count = 0
    for match in sorted(matches, key=lambda item: float(item.distance)):
        if match.queryIdx in used_query or match.trainIdx in used_train:
            continue
        used_query.add(match.queryIdx)
        used_train.add(match.trainIdx)
        count += 1
    return count


def evaluate_descriptor_pair(
    prepared: PreparedPair,
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    strategy: Mapping[str, Any],
    config: Mapping[str, Any],
) -> MatchResult:
    """用某个描述子和同一匹配策略评估一个图像对。"""

    if prepared.skip_reason:
        return MatchResult(0, 0, 0, 0, 0, float("nan"), None, prepared.skip_reason, [], np.zeros((0,), dtype=bool))
    matches = match_descriptors(desc_a, desc_b, strategy)
    return estimate_inliers(prepared.points_a, prepared.points_b, matches, config)


def strategy_label(strategy: Mapping[str, Any]) -> str:
    """把策略字典转换成可读标签。"""

    if "name" in strategy:
        return str(strategy["name"])
    if strategy.get("matcher") == "ratio":
        return f"ratio_{strategy.get('ratio_thresh')}_mutual_{int(bool(strategy.get('mutual', True)))}"
    return f"top{strategy.get('top_k')}_mutual_{int(bool(strategy.get('mutual', True)))}"


def safe_float(value: float | np.ndarray) -> float | None:
    """把 NaN/Inf 转成 JSON 友好的 None。"""

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def format_float(value: float) -> str:
    """把浮点数格式化为 CSV 字符串。"""

    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.6f}"


def draw_inlier_preview(
    pair: ImagePair,
    prepared: PreparedPair,
    result: MatchResult,
    output_path: Path,
    max_inliers: int = 80,
) -> bool:
    """绘制困难图像对的 RootSIFT RANSAC 内点连线预览图。"""

    image_a = imread_grayscale(pair.image_a_path)
    image_b = imread_grayscale(pair.image_b_path)
    if image_a is None or image_b is None:
        return False
    inlier_matches = [match for match, is_inlier in zip(result.matches, result.inlier_mask) if bool(is_inlier)]
    inlier_matches = sorted(inlier_matches, key=lambda match: float(match.distance))[: int(max_inliers)]
    preview = cv2.drawMatches(
        image_a,
        prepared.keypoints_a,
        image_b,
        prepared.keypoints_b,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return imwrite_image(output_path, preview)

