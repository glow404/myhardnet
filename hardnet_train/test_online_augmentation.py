"""训练 patch 在线增强及续训契约测试。"""

from __future__ import annotations

import csv
import random

import numpy as np
import pytest
import torch
from PIL import Image

from hardnet_train.data import (
    FingerprintPairDataset,
    PatchAugmentationConfig,
    augment_patch,
)
from hardnet_train.train import (
    descriptor_loss_contract,
    negative_sampling_contract,
    train_augmentation_contract,
    validate_resume_negative_sampling_contract,
)


def _training_config(*, augmentation_enabled: bool) -> dict[str, object]:
    return {
        "data": {
            "train_augmentation": {
                "enabled": augmentation_enabled,
                "probability": 0.8,
            }
        },
        "training": {
            "margin": 0.8,
            "hard_negative_strategy": "same_finger_allowed",
            "hard_negative_top_k": 3,
            "hard_negative_top1_weight": 0.6,
            "same_finger_min_coordinate_separation_px": 16.0,
        },
        "validation": {
            "protocol": "fixed_candidate_pool_v3",
            "seed": 10042,
            "finger_count": 2,
            "batch_count": 1,
            "batch_size": 2,
        },
    }


def test_light_augmentation_preserves_shape_range_and_is_reproducible() -> None:
    patch = np.tile(np.arange(32, dtype=np.float32), (32, 1)) * 8.0
    config = PatchAugmentationConfig.from_mapping(
        {
            "enabled": True,
            "probability": 1.0,
            "blur_probability": 1.0,
            "noise_probability": 1.0,
        }
    )

    random.seed(17)
    np.random.seed(17)
    first = augment_patch(patch, config)
    random.seed(17)
    np.random.seed(17)
    second = augment_patch(patch, config)

    np.testing.assert_allclose(first, second)
    assert first.shape == patch.shape
    assert first.dtype == np.float32
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 255.0
    assert not np.allclose(first, patch)


def test_disabled_augmentation_is_exact_identity() -> None:
    patch = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) % 256

    output = augment_patch(patch, PatchAugmentationConfig())

    np.testing.assert_array_equal(output, patch)


def test_training_entry_defaults_to_enabled_augmentation() -> None:
    assert train_augmentation_contract({"data": {}})["enabled"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("probability", 1.1),
        ("noise_probability", -0.1),
        ("scale_jitter", 1.0),
        ("max_rotation_degrees", -1.0),
    ],
)
def test_invalid_augmentation_config_is_rejected(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        PatchAugmentationConfig.from_mapping({"enabled": True, name: value})


def test_dataset_applies_augmentation_before_patch_normalization(tmp_path) -> None:
    patch_path = tmp_path / "patch.png"
    csv_path = tmp_path / "pairs.csv"
    patch = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) * 8
    Image.fromarray(patch, mode="L").save(patch_path)
    fieldnames = [
        "patch_a_path",
        "patch_p_path",
        "finger_id",
        "image_pair_id",
        "image_a_id",
        "image_b_id",
        "kp_a_idx",
        "kp_b_idx",
        "x_a",
        "y_a",
        "x_b",
        "y_b",
        "pair_id",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "patch_a_path": str(patch_path),
                "patch_p_path": str(patch_path),
                "finger_id": "finger-1",
                "image_pair_id": "pair-1",
                "image_a_id": "image-a",
                "image_b_id": "image-b",
                "kp_a_idx": "1",
                "kp_b_idx": "2",
                "x_a": "10",
                "y_a": "11",
                "x_b": "12",
                "y_b": "13",
                "pair_id": "positive-1",
            }
        )
    dataset = FingerprintPairDataset(
        csv_path,
        augmentation={
            "enabled": True,
            "probability": 1.0,
            "noise_probability": 1.0,
            "max_noise_std": 2.0,
        },
    )

    random.seed(3)
    np.random.seed(3)
    sample = dataset[0]
    anchor = sample["anchor"]
    positive = sample["positive"]

    assert isinstance(anchor, torch.Tensor)
    assert isinstance(positive, torch.Tensor)
    assert anchor.shape == positive.shape == (1, 32, 32)
    torch.testing.assert_close(anchor.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(anchor.std(correction=0), torch.tensor(1.0), atol=1e-5, rtol=0)
    assert not torch.equal(anchor, positive)


def test_enabled_augmentation_rejects_legacy_resume_checkpoint() -> None:
    config = _training_config(augmentation_enabled=True)
    checkpoint = {
        "negative_sampling_contract": negative_sampling_contract(config),
        "descriptor_loss_contract": descriptor_loss_contract(config),
    }

    with pytest.raises(ValueError, match="online-augmentation contract"):
        validate_resume_negative_sampling_contract(checkpoint, config)

    checkpoint["train_augmentation_contract"] = train_augmentation_contract(config)
    validate_resume_negative_sampling_contract(checkpoint, config)
