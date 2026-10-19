"""面向匹配部署的多指标 checkpoint 选择。"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_SELECTION = {
    "strategy": "matching_composite_v1",
    "weights": {
        "loss": 0.30,
        "distance_gap": 0.20,
        "fpr_at_tpr95": 0.35,
        "pos_p95": 0.15,
    },
}


def normalize_checkpoint_selection_config(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """规范化并校验匹配导向的 checkpoint 选择配置。"""

    raw = value if isinstance(value, Mapping) else {}
    strategy = str(raw.get("strategy", DEFAULT_CHECKPOINT_SELECTION["strategy"])).strip().lower()
    if strategy != "matching_composite_v1":
        raise ValueError(
            "Unsupported checkpoint_selection.strategy: "
            f"{strategy!r}. Expected 'matching_composite_v1'."
        )
    raw_weights = raw.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("checkpoint_selection.weights must be a mapping.")
    expected_weight_names = set(DEFAULT_CHECKPOINT_SELECTION["weights"])
    unknown_weight_names = sorted(set(raw_weights) - expected_weight_names)
    if unknown_weight_names:
        raise ValueError(
            "Unsupported checkpoint_selection weight names: "
            f"{unknown_weight_names}. Supported names are "
            f"{sorted(expected_weight_names)}."
        )
    weights = {
        name: float(raw_weights.get(name, default))
        for name, default in DEFAULT_CHECKPOINT_SELECTION["weights"].items()
    }
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("checkpoint_selection weights must be finite and non-negative.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("checkpoint_selection weights must have a positive sum.")
    weights = {name: weight / total for name, weight in weights.items()}
    return {"strategy": strategy, "weights": weights}


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _unit_interval(value: float | None, *, lower_is_better: bool = False) -> float:
    if value is None:
        return 0.0
    bounded = min(max(float(value), 0.0), 1.0)
    return 1.0 - bounded if lower_is_better else bounded


def _validation_margin(
    config: Mapping[str, Any] | None,
    descriptor_metric: str,
) -> float:
    section_name = (
        "validation"
        if descriptor_metric in {"hamming", "binary_hamming"}
        else "training"
    )
    default = 0.2 if section_name == "validation" else 1.0
    section = config.get(section_name, {}) if isinstance(config, Mapping) else {}
    raw_margin = section.get("margin", default) if isinstance(section, Mapping) else default
    margin = float(raw_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"{section_name}.margin must be finite and non-negative.")
    return margin


def checkpoint_selection_metrics(
    validation_metrics: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """由四项总体验证指标生成唯一的、越高越好的 checkpoint 分数。"""

    selection_config = normalize_checkpoint_selection_config(
        config.get("checkpoint_selection") if isinstance(config, Mapping) else None
    )
    weights = selection_config["weights"]
    descriptor_metric = str(
        config.get("descriptor_metric", "l2") if isinstance(config, Mapping) else "l2"
    ).strip().lower()
    distance_cap = 1.0 if descriptor_metric in {"hamming", "binary_hamming"} else 2.0
    loss_cap = distance_cap + _validation_margin(config, descriptor_metric)

    validation_loss = _finite_metric(validation_metrics, "loss")
    positive_mean = _finite_metric(validation_metrics, "pos_mean")
    negative_mean = _finite_metric(validation_metrics, "neg_mean")
    distance_gap = (
        None
        if positive_mean is None or negative_mean is None
        else (negative_mean - positive_mean) / distance_cap
    )
    fpr = _finite_metric(validation_metrics, "fpr_at_tpr95")
    positive_p95 = _finite_metric(validation_metrics, "pos_p95")

    components = {
        "loss": _unit_interval(
            None if validation_loss is None else validation_loss / loss_cap,
            lower_is_better=True,
        ),
        "distance_gap": _unit_interval(distance_gap),
        "fpr_at_tpr95": _unit_interval(fpr, lower_is_better=True),
        "pos_p95": _unit_interval(
            None if positive_p95 is None else positive_p95 / distance_cap,
            lower_is_better=True,
        ),
    }
    score = sum(weights[name] * components[name] for name in weights)
    return {"checkpoint_selection_score": float(score)}


def checkpoint_selection_state_from_metrics(
    metrics_path: Path,
) -> tuple[float, float, int]:
    """从 metrics.csv 恢复综合分数、早停分数和未改善计数。"""

    if not metrics_path.exists():
        return float("-inf"), float("-inf"), 0
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return float("-inf"), float("-inf"), 0
    scores = [
        float(row["val_checkpoint_selection_score"])
        for row in rows
        if row.get("val_checkpoint_selection_score") not in (None, "")
        and math.isfinite(float(row["val_checkpoint_selection_score"]))
    ]
    if not scores:
        raise ValueError(
            "Existing metrics.csv has no val_checkpoint_selection_score. "
            "Start a new output directory; old selection runs cannot be resumed "
            "under matching_composite_v1."
        )
    last = rows[-1]
    early_best = float(
        last.get("early_stop_best_selection_score")
        or max(scores)
    )
    no_improve = int(float(last.get("no_improve_epochs") or 0))
    return max(scores), early_best, no_improve