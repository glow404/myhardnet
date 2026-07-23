"""HardNet L2 匹配策略消融脚本。

作用：
    复用 `run_hardnet_matching.py` 已经构建好的模板、注册/query 划分，
    扫描不同的候选生成参数，例如 top_k、ratio_threshold、candidate_policy。

典型命令：
    python match_new\\ablation.py `
      --base_output_dir match_new\\outputs `
      --output_dir match_new\\outputs_ablation `
      --topk_values 1 2 5 10 `
      --ratio_thresholds 0.85 0.90 0.95 0.98
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.evaluation import run_hardnet_evaluation, summary_row
from match_new.utils import ensure_dir, load_config, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"
DEFAULT_BASE_OUTPUT = MATCH_NEW_DIR / "outputs"
DEFAULT_OUTPUT = MATCH_NEW_DIR / "outputs_ablation"


def parse_args() -> argparse.Namespace:
    """解析消融实验参数。"""

    parser = argparse.ArgumentParser(description="Scan HardNet L2 matching parameters using prebuilt match_new templates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base_output_dir", default=str(DEFAULT_BASE_OUTPUT), help="Directory containing image_templates and split metadata.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--topk_values", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--ratio_thresholds", type=float, nargs="+", default=[0.85, 0.90, 0.95, 0.98])
    parser.add_argument("--candidate_policies", nargs="+", default=["ratio_only", "topk_only", "topk_or_ratio"])
    parser.add_argument("--max_impostor_identities_per_query", type=int, default=0)
    return parser.parse_args()


def require_existing_inputs(base_output_dir: Path) -> tuple[Path, Path, Path]:
    """确认主实验已经生成了消融所需的三个输入。

    必需输入：
        - metadata_with_split_20.csv：固定的 enroll/query 划分；
        - identity_templates_20.json：每个 identity 的 20 张注册模板；
        - image_templates/：每张图像的 keypoints 和 descriptors。
    """

    metadata = base_output_dir / "metadata_with_split_20.csv"
    identities = base_output_dir / "identity_templates_20.json"
    templates = base_output_dir / "image_templates"
    missing = [path for path in [metadata, identities, templates] if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prebuilt experiment inputs: {joined}. Run run_hardnet_matching.py first.")
    return metadata, identities, templates


def set_matching_overrides(config: dict[str, Any], top_k: int, ratio: float, policy: str) -> dict[str, Any]:
    """为单次消融生成一份独立配置。

    使用 deepcopy 是为了避免多组参数之间互相污染。
    """

    updated = copy.deepcopy(config)
    matching = updated.setdefault("matching", {})
    matching["top_k"] = int(top_k)
    matching["ratio_threshold"] = float(ratio)
    matching["candidate_policy"] = str(policy)
    return updated


def main() -> None:
    """按参数网格逐组评估，并写出 ablation_summary。"""

    args = parse_args()
    config = load_config(args.config)
    base_output_dir = Path(args.base_output_dir).expanduser()
    metadata, identities, templates = require_existing_inputs(base_output_dir)
    output_dir = ensure_dir(args.output_dir)
    far_points = [float(point) for point in dict(config.get("evaluation", {})).get("far_points", [0.001, 0.0001])]
    rows: list[dict[str, Any]] = []

    # 三层循环形成参数网格；每个组合独立输出一个 run_dir。
    for policy in args.candidate_policies:
        for top_k in args.topk_values:
            for ratio in args.ratio_thresholds:
                run_config = set_matching_overrides(config, top_k, ratio, policy)
                run_name = f"policy_{policy}__topk_{top_k}__ratio_{ratio:.2f}".replace(".", "p")
                # 消融阶段默认不导出失败样本，避免每个参数组合都复制大量图片。
                result = run_hardnet_evaluation(
                    metadata_path=metadata,
                    identity_templates_path=identities,
                    template_dir=templates,
                    config=run_config,
                    output_dir=output_dir / run_name,
                    max_impostor_identities_per_query=int(args.max_impostor_identities_per_query),
                    export_failures=False,
                )
                row = summary_row(result["metrics"], far_points)
                row.update({"candidate_policy": policy, "top_k": int(top_k), "ratio_threshold": float(ratio), "run_dir": str(output_dir / run_name)})
                rows.append(row)

    summary_csv = output_dir / "ablation_summary.csv"
    summary_json = output_dir / "ablation_summary.json"
    write_csv_rows(summary_csv, rows)
    write_json(summary_json, {"summary": rows, "base_output_dir": str(base_output_dir)})
    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
