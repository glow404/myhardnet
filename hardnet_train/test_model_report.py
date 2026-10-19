"""逐层模型结构报告的契约测试。"""

from __future__ import annotations

import csv

import pytest

from hardnet_train.binary_model import build_binary_descriptor_model
from hardnet_train.model import build_descriptor_model, estimate_macs_per_patch
from hardnet_train.model_report import profile_model, write_model_structure_report


@pytest.mark.parametrize(
    ("architecture", "descriptor_dim"),
    [
        ("hardnet", 128),
        ("mobile_hardnet", 192),
        ("hardnet_strong_v2", 256),
    ],
)
def test_profile_covers_all_float_architectures(
    architecture: str,
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": descriptor_dim}
    )

    profile = profile_model(model)

    assert profile.architecture == architecture
    assert profile.descriptor_dim == descriptor_dim
    assert profile.input_shape == "[1,1,32,32]"
    assert profile.output_shape == f"[1,{descriptor_dim}]"
    assert profile.parameters == sum(parameter.numel() for parameter in model.parameters())
    assert profile.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert profile.macs_per_patch == estimate_macs_per_patch(model)
    assert profile.flops_per_patch == profile.macs_per_patch * 2
    assert profile.layers
    assert all(row.input_shape and row.output_shape and row.purpose for row in profile.layers)
    assert sum(row.parameters for row in profile.layers) == profile.parameters


def test_write_model_structure_report_creates_markdown_and_csv(tmp_path) -> None:
    model = build_descriptor_model(
        {"architecture": "hardnet_strong_v2", "descriptor_dim": 128}
    )

    profile, markdown_path, csv_path = write_model_structure_report(model, tmp_path)

    assert markdown_path.parent == csv_path.parent == tmp_path
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "HardNetStrongV2 模型结构报告" in markdown
    assert "逐层明细" in markdown
    assert f"{profile.parameters:,}" in markdown

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(profile.layers)
    assert rows[0]["输入形状"] == "[1,1,32,32]"
    assert any(row["层路径"].startswith("spatial_projection") for row in rows)


def test_profile_supports_frozen_backbone_binary_model() -> None:
    backbone = build_descriptor_model(
        {"architecture": "mobile_hardnet", "descriptor_dim": 128}
    )
    model = build_binary_descriptor_model(
        backbone,
        {
            "hash_bits": 64,
            "hidden_multiplier": 2.0,
            "backbone_trainable": False,
        },
    )
    model.train()

    profile = profile_model(model)

    assert profile.architecture == "residual_binary_hash_v1"
    assert profile.output_shape == "[1,64]"
    assert profile.trainable_parameters < profile.parameters
    assert model.training
    assert not model.backbone.training
    assert any(row.name == "hash_head" for row in profile.layers)
