"""match_new 实验通用工具函数。

创建目录、读取 YAML、解析配置相对路径、读写 CSV/JSON、生成安全模板名。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""

    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并记录 `_config_path` 供相对路径解析使用。"""

    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
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


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """写 CSV；未传 fieldnames 时按 rows 中字段首次出现顺序收集列名。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """写 UTF-8 JSON，保留中文。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
