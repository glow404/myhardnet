"""固定阈值在线指纹解锁的命令行入口。

脚本支持两种互斥模式：
    1. ``--image`` 配合 ``--identity``，对一张原始指纹图像执行单次解锁；
    2. ``--benchmark``，复用同一个在线引擎遍历离线划分中的本人查询，
       输出单次耗时明细和汇总统计。

运行前必须先由 ``run_hardnet_matching.py`` 生成注册模板和 query 划分。
脚本通过 ``--artifacts`` 与 ``online_unlock`` 配置直接定位这些文件；在线模型、
预处理、匹配和阈值参数使用当前配置，可以独立调整以测量优化效果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.online_unlock import OnlineUnlockEngine
from match_new.utils import ensure_dir, load_config, read_csv_rows, resolve_path, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"


def parse_args() -> argparse.Namespace:
    """解析单次解锁或在线延迟基准所需的命令行参数。"""

    parser = argparse.ArgumentParser(
        description="执行一次固定阈值在线解锁，或运行本人查询的在线延迟基准。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="本次在线耗时测试使用的 YAML 配置文件。")
    parser.add_argument("--artifacts", default=None, help="离线评估生成的产物目录。")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", default=None, help="单次解锁使用的一张原始查询图像。")
    mode.add_argument("--benchmark", action="store_true", help="对离线划分中的全部本人查询运行在线延迟基准。")
    parser.add_argument("--identity", default=None, help="使用 --image 时要验证的已注册 identity。")
    parser.add_argument("--limit", type=int, default=0, help="限制基准查询数量；0 表示使用全部查询。")
    parser.add_argument("--output-dir", default=None, help="覆盖在线基准结果输出目录。")
    return parser.parse_args()


def resolve_artifacts(config: dict, value: str | None) -> Path:
    """解析离线产物目录，命令行参数优先于配置文件中的输出目录。"""

    if value:
        return Path(value).expanduser().resolve()
    output_dir = dict(config.get("output", {})).get("output_dir")
    if not output_dir:
        raise ValueError("--artifacts is required when output.output_dir is not configured")
    return resolve_path(config, output_dir)


def main() -> None:
    """创建常驻在线引擎，并根据参数执行单次解锁或批量延迟基准。"""

    args = parse_args()
    if args.image and not args.identity:
        raise ValueError("--identity is required with --image")

    config = load_config(args.config)
    artifacts_dir = resolve_artifacts(config, args.artifacts)
    engine = OnlineUnlockEngine(config, artifacts_dir)

    if args.image:
        result = engine.unlock(args.image, args.identity)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    split_path = engine.split_metadata_path
    if not split_path.exists():
        raise FileNotFoundError(f"query 划分文件不存在: {split_path}")
    query_rows = [
        row
        for row in read_csv_rows(split_path)
        if str(row.get("split", "")).lower() == "query"
    ]
    if args.limit > 0:
        query_rows = query_rows[: int(args.limit)]
    if not query_rows:
        raise RuntimeError(f"no benchmark query rows found in {split_path}")

    benchmark = engine.benchmark(query_rows)
    if args.output_dir:
        output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    else:
        relative = str(
            dict(config.get("online_unlock", {})).get(
                "benchmark_output_dir",
                "online_unlock_benchmark",
            )
        )
        output_dir = ensure_dir(artifacts_dir / relative)
    attempts_path = write_csv_rows(output_dir / "online_unlock_attempts.csv", benchmark["attempts"])
    summary_path = write_json(output_dir / "online_unlock_summary.json", benchmark["summary"])
    print(
        json.dumps(
            {
                "attempts_csv": str(attempts_path),
                "summary_json": str(summary_path),
                "summary": benchmark["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
