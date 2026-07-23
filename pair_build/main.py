"""HardNet 指纹正样本数据集构建命令行入口。

作用：
- 统一调度 pair_build 项目的五个主要阶段：
  index -> split -> build_positive -> extract_patches -> qa
- 支持单独运行某个阶段，也支持 `all` 一次性重建完整数据集。
- 当单独运行中后期阶段时，会自动补齐缺失的前置产物。

常用命令：
  python pair_build/main.py --config pair_build/config.yaml --stage index
  python pair_build/main.py --config pair_build/config.yaml --stage all
"""

from __future__ import annotations

import argparse
from typing import Any

from index_builder import build_sample_index
from patch_extractor import extract_patches
from positive_builder import build_positive_pairs
from qa_tools import generate_qa_report
from split_builder import build_split
from utils import ensure_dir, get_nested, load_config, resolve_path, setup_logger


def _prepare_logger(config: dict[str, Any]):
    """根据配置创建输出目录和日志器。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    ensure_dir(output_root)
    return output_root, setup_logger(
        output_root,
        name=str(get_nested(config, "logging", "logger_name", default="hardnet_dataset")),
        level=str(get_nested(config, "logging", "level", default="INFO")),
    )


def _ensure_index(config: dict[str, Any], logger) -> None:
    """确保 index 阶段产物存在。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    if not (output_root / "sample_index.csv").exists():
        build_sample_index(config, logger)


def _ensure_split(config: dict[str, Any], logger) -> None:
    """确保 split 阶段产物存在；缺 index 时会自动补做。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    _ensure_index(config, logger)
    if not (output_root / "split_assignments.csv").exists():
        build_split(config, logger)


def _ensure_positive(config: dict[str, Any], logger) -> None:
    """确保 build_positive 阶段产物存在；缺前置产物时自动补做。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    _ensure_split(config, logger)
    if not (output_root / "all_positive_pairs.csv").exists():
        build_positive_pairs(config, logger)


def _ensure_patches(config: dict[str, Any], logger) -> None:
    """确保 patch 提取产物存在；缺正样本 CSV 时自动补做。"""

    output_root = resolve_path(config, get_nested(config, "paths", "output_root", default="../outputs/hardnet_dataset"))
    _ensure_positive(config, logger)
    if not (output_root / "patch_summary.json").exists():
        extract_patches(config, logger)


def run_stage(config: dict[str, Any], stage: str, logger) -> dict[str, Any] | None:
    """按 stage 名称分发到具体流水线阶段。"""

    stage = stage.lower()
    if stage == "index":
        return build_sample_index(config, logger)
    if stage == "split":
        _ensure_index(config, logger)
        return build_split(config, logger)
    if stage == "build_positive":
        _ensure_split(config, logger)
        return build_positive_pairs(config, logger)
    if stage == "extract_patches":
        _ensure_positive(config, logger)
        return extract_patches(config, logger)
    if stage == "qa":
        _ensure_patches(config, logger)
        return generate_qa_report(config, logger)
    if stage == "all":
        return {
            "index": build_sample_index(config, logger),
            "split": build_split(config, logger),
            "build_positive": build_positive_pairs(config, logger),
            "extract_patches": extract_patches(config, logger),
            "qa": generate_qa_report(config, logger),
        }
    raise ValueError(f"不支持的 stage: {stage}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="HardNet 指纹正样本数据集构建入口")
    parser.add_argument("--config", required=True, help="配置文件路径，例如 pair_build/config.yaml")
    parser.add_argument("--stage", required=True, choices=["index", "split", "build_positive", "extract_patches", "qa", "all"])
    return parser.parse_args()


def main() -> None:
    """CLI 主函数。"""

    args = parse_args()
    config = load_config(args.config)
    _, logger = _prepare_logger(config)
    logger.info("开始执行阶段: %s", args.stage)
    result = run_stage(config, args.stage, logger)
    logger.info("阶段完成: %s", args.stage)
    if result is not None:
        logger.info("结果摘要: %s", result)


if __name__ == "__main__":
    main()
