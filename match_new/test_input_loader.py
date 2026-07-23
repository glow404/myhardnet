from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from match_new.input_loader import scan_image_metadata, validate_identity_image_counts


def write_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    assert cv2.imwrite(str(path), image)


def test_scan_image_metadata_uses_configured_identity_depth(tmp_path: Path) -> None:
    write_test_image(tmp_path / "batch_a" / "finger_1" / "image_1.bmp")
    write_test_image(tmp_path / "batch_a" / "finger_2" / "image_2.bmp")

    rows = scan_image_metadata(tmp_path, identity_depth=2)

    assert [(row["identity_id"], row["image_id"]) for row in rows] == [
        ("batch_a/finger_1", "image_1"),
        ("batch_a/finger_2", "image_2"),
    ]
    assert all(Path(row["image_path"]).is_absolute() for row in rows)


def test_scan_image_metadata_disambiguates_duplicate_stems(tmp_path: Path) -> None:
    write_test_image(tmp_path / "finger_1" / "capture_a" / "image.bmp")
    write_test_image(tmp_path / "finger_1" / "capture_b" / "image.bmp")

    rows = scan_image_metadata(tmp_path, identity_depth=1)

    assert {row["identity_id"] for row in rows} == {"finger_1"}
    assert len({row["image_id"] for row in rows}) == 2
    assert all(row["image_id"].startswith("image__") for row in rows)


def test_scan_image_metadata_rejects_images_without_identity_directory(tmp_path: Path) -> None:
    write_test_image(tmp_path / "image.bmp")

    with pytest.raises(ValueError, match="identity directory"):
        scan_image_metadata(tmp_path, identity_depth=1)


def test_validate_identity_image_counts_reports_short_identities() -> None:
    rows = [
        {"identity_id": "finger_1", "image_id": "1", "image_path": "1.bmp", "split": ""},
        {"identity_id": "finger_1", "image_id": "2", "image_path": "2.bmp", "split": ""},
    ]

    with pytest.raises(ValueError, match="fewer than 3"):
        validate_identity_image_counts(rows, 3, context="test input")
