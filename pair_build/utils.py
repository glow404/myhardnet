"""pair_build 公共工具模块。

作用：
- 统一处理配置读取、相对路径解析、日志初始化、CSV/JSON 读写。
- 封装 Windows 中文路径下更稳的 OpenCV 图像读写。
- 提供稳定 ID、角度差、分数映射等多个阶段都会复用的小工具。

这个文件不包含具体的数据构建业务逻辑；它的目标是让其他脚本只关注
index / split / matching / patch / QA 各自的主流程。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML/JSON 配置，并记录配置文件绝对路径。

    记录 `_config_path` 是为了让配置里的相对路径都能相对于配置文件所在目录解析，
    而不是相对于用户当前执行命令的工作目录解析。
    """

    path = Path(config_path).expanduser().resolve()
    raw_text = path.read_text(encoding="utf-8")
    if yaml is not None:
        config = yaml.safe_load(raw_text)
    else:
        config = json.loads(raw_text)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须解析为字典: {path}")
    config["_config_path"] = str(path)
    return config


def get_nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """安全读取多层字典字段。

    例如 get_nested(config, "matching", "ratio_thresh", default=0.85)。
    如果中间任意一级不存在，就返回 default，避免到处写 try/except。
    """

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def resolve_path(config: Mapping[str, Any], raw_path: str | Path) -> Path:
    """把配置里的路径转换成绝对路径。

    绝对路径保持不变；相对路径则以配置文件所在目录为基准。
    这样 `python pair_build/main.py ...` 和从其他目录调用时行为一致。
    """

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(str(config["_config_path"])).resolve()
    return (config_path.parent / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logger(output_dir: str | Path, name: str = "hardnet_dataset", level: str = "INFO") -> logging.Logger:
    """创建同时输出到控制台和 pipeline.log 的 logger。"""

    ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(Path(output_dir) / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
    return target


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """写出 CSV。

    如果没有显式传 fieldnames，就使用第一行的字段顺序。
    空数据也会写出一个空表头文件，方便后续阶段明确知道产物已经生成。
    """

    target = Path(path)
    ensure_dir(target.parent)
    row_list = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(row_list[0].keys()) if row_list else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return target


def sanitize_token(text: Any) -> str:
    """把任意文本转成适合作为文件名片段的短 token。"""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("._") or "unknown"


def stable_id(*parts: Any, readable_parts: int = 4, digest_size: int = 12) -> str:
    """根据多个业务字段生成稳定 ID。

    ID 前半段保留少量可读信息，后半段附带 SHA1 摘要，兼顾人工排查和去重稳定性。
    """

    string_parts = [str(part) for part in parts]
    readable = "__".join(sanitize_token(part) for part in string_parts[:readable_parts])
    digest = hashlib.sha1("|".join(string_parts).encode("utf-8")).hexdigest()[:digest_size]
    return f"{readable}__{digest}"[:180]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def exp_score(value: float, scale: float) -> float:
    scale = max(float(scale), 1e-6)
    return math.exp(-max(float(value), 0.0) / scale)


def angle_diff_deg(angle_a: float, angle_b: float, period: float = 360.0) -> float:
    half = period / 2.0
    return abs((float(angle_a) - float(angle_b) + half) % period - half)


def angle_diff_mod180(angle_a: float, angle_b: float) -> float:
    """计算 180 度二义性下的方向差。

    指纹脊线和局部 patch 常有 180 度方向二义性，因此几何一致性更关注 mod 180
    后的差异，而不是完整 360 度方向差。
    """

    raw = angle_diff_deg(angle_a, angle_b, period=360.0)
    return min(raw, 180.0 - raw)


def imread_grayscale(path: str | Path) -> np.ndarray | None:
    """读取灰度图。

    使用 np.fromfile + cv2.imdecode，而不是 cv2.imread，是为了兼容 Windows 中文路径。
    """

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return image


def imwrite_image(path: str | Path, image: np.ndarray) -> bool:
    """写出图像。

    同样使用 cv2.imencode + tofile，以兼容 Windows 中文路径。
    """

    target = Path(path)
    ensure_dir(target.parent)
    suffix = target.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def image_extensions(config: Mapping[str, Any]) -> list[str]:
    """读取允许扫描的图像后缀列表，并统一转成小写。"""

    return [ext.lower() for ext in get_nested(config, "index", "image_extensions", default=[".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])]
