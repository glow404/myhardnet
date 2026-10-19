"""二值 metric loss 显式参数和 checkpoint 契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hardnet_train.binary_loss import (
    BINARY_METRIC_LOSS_DEFAULTS,
    BinaryDescriptorLoss,
)
from hardnet_train.train import descriptor_loss_contract, load_config


def test_binary_yaml_explicitly_controls_all_hardnet_metric_parameters() -> None:
    config = load_config(Path(__file__).with_name("config_binary_256.yaml"))
    training = config["training"]
    criterion = BinaryDescriptorLoss.from_config(
        training,
        config["binary_loss"],
    )
    metric = criterion.metric_loss

    expected = {
        "hard_negative_top1_weight": 0.60,
        "positive_tail_loss_weight": 0.05,
        "positive_tail_p95_weight": 0.70,
        "positive_tail_p99_weight": 0.30,
        "positive_tail_p95_target": 0.60,
        "positive_tail_p99_target": 0.80,
    }
    for name, value in expected.items():
        assert name in training
        assert getattr(metric, name) == pytest.approx(value)

    contract = descriptor_loss_contract(config)
    for name, value in expected.items():
        assert contract[name] == pytest.approx(value)
    assert criterion.weights["positive_consistency"] == pytest.approx(0.10)


def test_binary_metric_defaults_are_binary_specific_and_fully_forwarded() -> None:
    criterion = BinaryDescriptorLoss.from_config({}, {})

    for name, value in BINARY_METRIC_LOSS_DEFAULTS.items():
        assert getattr(criterion.metric_loss, name) == pytest.approx(value)

