"""match_new 实验通用工具函数。

创建目录、读取 YAML、解析配置相对路径、读写 CSV/JSON、生成安全模板名。
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml


OUTPUT_SIGNIFICANT_DIGITS = 4
_NUMERIC_TEXT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置；mapping 深合并，其他值由子配置完整覆盖。"""

    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(
    config_path: Path,
    stack: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    """读取单个 YAML，并解析同目录下可选的 ``extends`` 父配置。"""

    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Circular config inheritance: {chain}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    parent_value = raw.pop("extends", None)
    if parent_value is None or parent_value == "":
        return raw, [str(config_path)]
    if not isinstance(parent_value, (str, Path)):
        raise ValueError(
            f"Config extends must be a path string, got {type(parent_value).__name__}: {config_path}"
        )
    parent_path = Path(parent_value).expanduser()
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    if parent_path.parent != config_path.parent:
        raise ValueError(
            "Inherited match configs must be in the same directory so relative data/model "
            f"paths keep one origin: child={config_path}, parent={parent_path}"
        )
    parent, sources = _load_config_mapping(parent_path, (*stack, config_path))
    return _merge_config(parent, raw), [*sources, str(config_path)]


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""

    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML，解析可选继承，并记录路径供相对路径与运行清单使用。"""

    config_path = Path(path).expanduser().resolve()
    config, sources = _load_config_mapping(config_path)
    config["_config_path"] = str(config_path)
    config["_config_sources"] = sources
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """把配置中的路径解析为绝对路径。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV 文件，返回字典行列表。"""

    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _format_float(value: Real, significant_digits: int) -> str:
    """将有限浮点数格式化为指定有效数字，供结果文件和路径展示。"""

    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return format(number, f".{significant_digits}g")


def _format_numeric_text(value: str, significant_digits: int) -> str:
    """仅压缩较长的完整数值字符串，避免改动 identity 和路径等普通文本。"""

    text = value.strip()
    if (
        not text
        or not _NUMERIC_TEXT_PATTERN.fullmatch(text)
        or ("." not in text and "e" not in text.lower())
    ):
        return value
    mantissa = re.split(r"[eE]", text, maxsplit=1)[0]
    significant = mantissa.lstrip("+-").replace(".", "").lstrip("0")
    if len(significant) <= significant_digits:
        return value
    return _format_float(float(text), significant_digits)


def _format_csv_value(value: Any, significant_digits: int) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return _format_numeric_text(value, significant_digits)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _format_float(value, significant_digits)
    return value


def _round_json_numbers(value: Any, significant_digits: int) -> Any:
    """递归舍入 JSON 中的浮点数，同时保持结构和非数值类型不变。"""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return number
        return float(_format_float(number, significant_digits))
    if isinstance(value, Mapping):
        return {
            key: _round_json_numbers(item, significant_digits)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_round_json_numbers(item, significant_digits) for item in value]
    return value


def format_significant_digits(
    value: Real,
    significant_digits: int = OUTPUT_SIGNIFICANT_DIGITS,
) -> str:
    """返回适合展示或文件名使用的有效数字字符串。"""

    return _format_float(value, significant_digits)


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """写 CSV；浮点输出统一为 4 位有效数字，内部计算精度不受影响。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    formatted_rows = [
        {
            key: _format_csv_value(value, OUTPUT_SIGNIFICANT_DIGITS)
            for key, value in row.items()
        }
        for row in rows
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """写 UTF-8 JSON；浮点输出统一为 4 位有效数字。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    formatted_payload = _round_json_numbers(payload, OUTPUT_SIGNIFICANT_DIGITS)
    target.write_text(
        json.dumps(formatted_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def write_yaml(path: str | Path, payload: Any) -> Path:
    """写 UTF-8 YAML，保留中文。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def safe_id(value: str) -> str:
    """把 identity_id/image_id 转为适合文件名使用的安全字符串。"""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def template_filename(identity_id: str, image_id: str) -> str:
    """生成单张图像模板 `.npz` 文件名。"""

    return f"{safe_id(identity_id)}__{safe_id(image_id)}.npz"
