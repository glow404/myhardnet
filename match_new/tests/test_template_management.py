from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from match_new.template_builder import load_image_template
from match_new.template_learning import common_area_metrics, template_content_hash
from match_new.template_library import TemplateLibraryManager
from match_new.template_ranking import insert_template_first, touch_template
from match_new.template_replacement import select_lru_victim


def textured_image(offset: int = 0) -> np.ndarray:
    """生成具有稳定局部对比度、但可通过offset区分内容哈希的测试纹理。"""

    yy, xx = np.indices((48, 48))
    image = np.where(((xx + offset) // 3) % 2 == 0, 35 + offset, 220 - offset)
    image = np.where((yy // 12) % 2 == 0, image, np.clip(image + 8, 0, 255))
    return image.astype(np.uint8)


def write_template(path: Path, identity_id: str, image_id: str, offset: int) -> dict[str, Any]:
    image = textured_image(offset)
    keypoints = np.asarray([[8.0, 8.0], [20.0, 20.0], [36.0, 36.0]], dtype=np.float32)
    descriptors = np.zeros((3, 128), dtype=np.float32)
    descriptors[:, offset % 128] = 1.0
    np.savez_compressed(
        path,
        template_format_version=np.asarray(2, dtype=np.uint8),
        identity_id=np.asarray(identity_id),
        image_id=np.asarray(image_id),
        image_path=np.asarray(str(path.with_suffix(".png"))),
        keypoints_xy=keypoints,
        keypoints_size=np.ones((3,), dtype=np.float32),
        keypoints_angle=np.zeros((3,), dtype=np.float32),
        keypoints_response=np.ones((3,), dtype=np.float32),
        hardnet_descriptors=descriptors,
        overlap_image=image,
    )
    return load_image_template(path, require="hardnet")


def successful_match(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "score": 0.92,
        "unique_inliers": 14,
        "texture_available": True,
        "texture_similarity": 0.88,
        "affine_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }


class RankingAndReplacementTests(unittest.TestCase):
    def test_lru_touch_and_insert(self) -> None:
        entries = [
            {"template_path": "a", "successful_match_count": 0},
            {"template_path": "b", "successful_match_count": 0},
            {"template_path": "c", "successful_match_count": 0},
        ]
        touch_template(entries, "b", 1)
        self.assertEqual([entry["template_path"] for entry in entries], ["b", "a", "c"])
        insert_template_first(entries, {"template_path": "d"}, 2)
        self.assertEqual([entry["template_path"] for entry in entries], ["d", "b", "a", "c"])

    def test_victim_skips_protected_tail(self) -> None:
        entries = [
            {"template_path": "learned-new", "protected": False},
            {"template_path": "learned-old", "protected": False},
            {"template_path": "seed", "protected": True},
        ]
        self.assertEqual(select_lru_victim(entries)["template_path"], "learned-old")


class LearningGeometryTests(unittest.TestCase):
    def test_common_area_uses_real_texture_mask(self) -> None:
        image = textured_image()
        identity = common_area_metrics(
            image,
            image,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            {"coverage_min_std": 5.0},
        )
        shifted = common_area_metrics(
            image,
            image,
            [[1.0, 0.0, 36.0], [0.0, 1.0, 0.0]],
            {"coverage_min_std": 5.0},
        )
        self.assertTrue(identity["available"])
        self.assertGreater(identity["common_area_ratio"], 0.9)
        self.assertLess(shifted["common_area_ratio"], identity["common_area_ratio"])

    def test_content_hash_ignores_metadata(self) -> None:
        template_a = {
            "image_id": "a",
            "overlap_image": textured_image(),
            "keypoints_xy": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "hardnet_descriptors": np.ones((1, 128), dtype=np.float32),
        }
        template_b = {**template_a, "image_id": "b", "image_path": "different.png"}
        self.assertEqual(template_content_hash(template_a), template_content_hash(template_b))


class TemplateLibraryManagerTests(unittest.TestCase):
    def test_persist_retries_transient_windows_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identities = [{"identity_id": "finger", "template_paths": [], "template_image_ids": []}]
            manager = TemplateLibraryManager(
                identities,
                root / "output",
                {
                    "persist_replace_retries": 3,
                    "persist_retry_delay_ms": 0,
                },
            )
            real_replace = __import__("os").replace
            calls = 0

            def locked_twice(source: str | Path, target: str | Path) -> None:
                nonlocal calls
                calls += 1
                if calls <= 2:
                    raise PermissionError(5, "file is temporarily locked", str(target))
                real_replace(source, target)

            with patch("match_new.template_library.os.replace", side_effect=locked_twice):
                self.assertTrue(manager.persist())

            self.assertEqual(calls, 3)
            self.assertTrue(manager.library_path.exists())

    def test_loads_newer_pending_state_after_failed_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            identities = [{"identity_id": "finger", "template_paths": [], "template_image_ids": []}]
            manager = TemplateLibraryManager(
                identities,
                output,
                {
                    "persist_replace_retries": 1,
                    "persist_retry_delay_ms": 0,
                },
            )
            manager.state["update_counter"] = 7
            real_replace = __import__("os").replace

            def lock_library_only(source: str | Path, target: str | Path) -> None:
                if Path(target) == manager.library_path:
                    raise PermissionError(5, "locked", str(target))
                real_replace(source, target)

            with patch("match_new.template_library.os.replace", side_effect=lock_library_only):
                self.assertTrue(manager.persist())

            recovered = TemplateLibraryManager(
                identities,
                output,
                {
                    "reset_library_on_start": False,
                    "persist_replace_retries": 1,
                    "persist_retry_delay_ms": 0,
                },
            )
            self.assertEqual(recovered.state["update_counter"], 7)

    def test_twenty_seeds_are_protected_and_lru_replaces_only_learned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_map: dict[str, dict[str, Any]] = {}
            seed_paths: list[str] = []
            seed_ids: list[str] = []
            for index in range(20):
                path = root / f"seed_{index}.npz"
                template = write_template(path, "finger", f"seed_{index}", index + 1)
                template_map[str(path)] = template
                seed_paths.append(str(path))
                seed_ids.append(f"seed_{index}")
            identities = [
                {
                    "identity_id": "finger",
                    "template_paths": seed_paths,
                    "template_image_ids": seed_ids,
                    "num_templates": 20,
                }
            ]
            management = {
                "reset_library_on_start": True,
                "max_active_templates": 21,
                "protected_seed_templates": 20,
                "learn_score_threshold": 0.85,
                "learn_min_unique_inliers": 12,
                "learn_min_texture_similarity": 0.75,
                "confirm_score_threshold": 0.70,
                "confirmation_templates": 2,
                "require_seed_confirmation": True,
                "min_common_area_ratio": 0.35,
                "min_common_area_pixels": 128,
            }
            full_config = {
                "identification": {"match_score_threshold": 0.55},
                "template_management": management,
            }
            manager = TemplateLibraryManager(identities, root / "output", management)

            def loader(path: str) -> dict[str, Any]:
                if path not in template_map:
                    template_map[path] = load_image_template(path, require="hardnet")
                return template_map[path]

            first_path = root / "query_first.npz"
            first_query = write_template(first_path, "finger", "query_first", 31)
            first_event = manager.process_query(
                first_query,
                identities[0],
                full_config,
                template_loader=loader,
                matcher=successful_match,
            )
            self.assertTrue(first_event["learned"])
            first_learned = first_event["learned_template_path"]
            self.assertEqual(len(manager.ordered_entries("finger")), 21)
            self.assertEqual(manager.ordered_entries("finger")[0]["template_path"], first_learned)
            self.assertEqual(sum(bool(item["protected"]) for item in manager.ordered_entries("finger")), 20)

            second_path = root / "query_second.npz"
            second_query = write_template(second_path, "finger", "query_second", 37)
            second_event = manager.process_query(
                second_query,
                identities[0],
                full_config,
                template_loader=loader,
                matcher=successful_match,
            )
            self.assertTrue(second_event["learned"])
            self.assertEqual(second_event["replaced_template_path"], first_learned)
            self.assertEqual(len(manager.ordered_entries("finger")), 21)
            self.assertTrue(all(item["protected"] for item in manager.ordered_entries("finger") if item["source"] == "seed"))

    def test_exact_duplicate_is_not_learned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_path = root / "seed.npz"
            seed = write_template(seed_path, "finger", "seed", 4)
            duplicate_path = root / "duplicate.npz"
            duplicate = write_template(duplicate_path, "finger", "duplicate", 4)
            identities = [{"identity_id": "finger", "template_paths": [str(seed_path)], "template_image_ids": ["seed"]}]
            management = {
                "reset_library_on_start": True,
                "max_active_templates": 40,
                "protected_seed_templates": 20,
                "confirmation_templates": 1,
                "require_seed_confirmation": True,
                "min_common_area_pixels": 128,
            }
            manager = TemplateLibraryManager(identities, root / "output", management)
            event = manager.process_query(
                duplicate,
                identities[0],
                {"identification": {"match_score_threshold": 0.55}},
                template_loader=lambda _path: seed,
                matcher=successful_match,
            )
            self.assertFalse(event["learned"])
            self.assertIn("duplicate_template_content", event["learning_reasons"])


if __name__ == "__main__":
    unittest.main()
