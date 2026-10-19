"""Descriptor-dimension contracts for all float HardNet architectures."""

from __future__ import annotations

import pytest
import torch

from hardnet_train.model import (
    ContextMixer,
    CoordinateAttention,
    ScalarWeightedFusion,
    StableResBlock,
    build_descriptor_model,
)
from hardnet_train.train import default_output_dir


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [64, 128, 256])
def test_float_architectures_support_configurable_descriptor_dim(
    architecture: str,
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {
            "architecture": architecture,
            "descriptor_dim": descriptor_dim,
            "dropout": 0.1,
            "final_bn_affine": False,
        }
    ).eval()

    with torch.inference_mode():
        features = model.forward_features(torch.randn(2, 1, 32, 32))

    assert model.descriptor_dim == descriptor_dim
    assert features.q.shape == (2, descriptor_dim)
    assert features.f.shape == (2, descriptor_dim)
    torch.testing.assert_close(
        torch.linalg.vector_norm(features.f, dim=1),
        torch.ones(2),
    )


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [64, 256])
def test_custom_descriptor_dim_supports_model_backward(
    architecture: str,
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": descriptor_dim}
    )
    anchor = model(torch.randn(4, 1, 32, 32))
    positive = model(torch.randn(4, 1, 32, 32))
    loss = (1.0 - (anchor * positive).sum(dim=1)).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
def test_float_architectures_keep_128_as_default(architecture: str) -> None:
    automatic = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": "auto"}
    )
    explicit = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": 128}
    )

    assert automatic.descriptor_dim == explicit.descriptor_dim == 128
    assert automatic.state_dict().keys() == explicit.state_dict().keys()
    for name, tensor in automatic.state_dict().items():
        assert tensor.shape == explicit.state_dict()[name].shape

    # Changing descriptor dimension must only resize the configured output head.
    expected_output_shape = {
        "hardnet": (128, 128, 8, 8),
        "mobile_hardnet": (128, 192, 1, 1),
    }[architecture]
    output_weight_key = {
        "hardnet": "features.7.weight",
        "mobile_hardnet": "descriptor_head.2.weight",
    }[architecture]
    assert automatic.state_dict()[output_weight_key].shape == expected_output_shape


def test_coordinate_attention_preserves_shape_and_supports_backward() -> None:
    attention = CoordinateAttention(96, reduction=32)
    inputs = torch.randn(2, 96, 8, 8, requires_grad=True)

    outputs = attention(inputs)
    outputs.square().mean().backward()

    assert attention.bottleneck_channels == 8
    assert outputs.shape == inputs.shape
    assert torch.all(torch.isfinite(outputs))
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_mobile_hardnet_contains_coordinate_attention() -> None:
    model = build_descriptor_model({"architecture": "mobile_hardnet"})

    assert model.architecture == "mobile_hardnet"
    assert isinstance(model.coordinate_attention, CoordinateAttention)


@pytest.mark.parametrize("descriptor_dim", [64, 128, 256])
def test_strong_v2_lightweight_structure_and_output_dimension(
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {
            "architecture": "hardnet_strong_v2",
            "descriptor_dim": descriptor_dim,
            "drop_path_rate": 0.08,
        }
    ).eval()

    with torch.inference_mode():
        features = model.forward_features(torch.randn(2, 1, 32, 32))

    assert features.q.shape == features.f.shape == (2, descriptor_dim)
    torch.testing.assert_close(
        torch.linalg.vector_norm(features.f, dim=1),
        torch.ones(2),
    )
    assert not hasattr(model.stem, "context_branch")
    assert (len(model.stage1), len(model.stage2), len(model.stage3)) == (1, 2, 2)
    assert isinstance(model.context, ContextMixer)
    assert isinstance(model.fusion, ScalarWeightedFusion)

    spatial_depthwise = model.spatial_projection[0]
    assert isinstance(spatial_depthwise, torch.nn.Conv2d)
    assert spatial_depthwise.kernel_size == (8, 8)
    assert spatial_depthwise.groups == spatial_depthwise.in_channels == 128
    assert model.spatial_projection[-1].out_channels == descriptor_dim

    residual_blocks = [
        module for module in model.modules() if isinstance(module, StableResBlock)
    ]
    drop_path_rates = [block.drop_path.probability for block in residual_blocks]
    assert len(residual_blocks) == 7
    assert drop_path_rates == sorted(drop_path_rates)
    assert drop_path_rates[0] == 0.0
    assert drop_path_rates[-1] == pytest.approx(0.08)
    assert all(not hasattr(block, "attention") for block in residual_blocks)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_300_000


def test_strong_v2_lightweight_backbone_supports_backward() -> None:
    model = build_descriptor_model(
        {
            "architecture": "hardnet_strong_v2",
            "descriptor_dim": 128,
            "drop_path_rate": 0.08,
        }
    )
    anchor = model(torch.randn(4, 1, 32, 32))
    positive = model(torch.randn(4, 1, 32, 32))
    loss = (1.0 - (anchor * positive).sum(dim=1)).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert model.fusion.weight_logit.grad is not None


@pytest.mark.parametrize("drop_path_rate", [-0.01, 1.0])
def test_strong_v2_rejects_invalid_drop_path_rate(drop_path_rate: float) -> None:
    with pytest.raises(ValueError, match="drop_path_rate must be in"):
        build_descriptor_model(
            {
                "architecture": "hardnet_strong_v2",
                "drop_path_rate": drop_path_rate,
            }
        )


def test_default_output_directory_includes_custom_descriptor_dim() -> None:
    assert "hardnet_train_hardnet_256_" in default_output_dir("hardnet", 256)
    assert "hardnet_train_mobile_64_" in default_output_dir("mobile_hardnet", 64)
    assert "hardnet_train_strong_v2_384_" in default_output_dir(
        "hardnet_strong_v2", 384
    )


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [0, -1])
def test_float_architectures_reject_non_positive_descriptor_dim(
    architecture: str,
    descriptor_dim: int,
) -> None:
    with pytest.raises(ValueError, match="descriptor_dim must be positive"):
        build_descriptor_model(
            {"architecture": architecture, "descriptor_dim": descriptor_dim}
        )
