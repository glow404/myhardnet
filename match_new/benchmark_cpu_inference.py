"""Small, repeatable CPU FP32 benchmark on identical real keypoint patches.

The default configuration is ONNX Runtime CPU; only ONNX Runtime variants are benchmarked.
Image/SIFT/patch and descriptor/matching rules are taken unchanged from the base config. Startup, export, compilation
and warmup are recorded separately. No enrollment artifacts are written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import multiprocessing
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from match_new.hardnet_matcher import build_descriptor_candidates, match_templates_descriptor
from match_new.input_loader import load_raw_image_metadata
from match_new.runtime import HARDNET_INFERENCE_STAGE_FIELDS, HardNetDescriptor, build_sift
from match_new.template_builder import detect_keypoints_only, extract_keypoint_patches, hardnet_template_payload
from match_new.utils import load_config, resolve_path, write_csv_rows, write_json
from match_new.runtime import imread_grayscale


DEFAULT_VARIANTS = (
    ("onnxruntime_16", {"inference_backend": "onnxruntime", "backend_num_threads": 16}),
    ("onnxruntime_28", {"inference_backend": "onnxruntime", "backend_num_threads": 28}),
    ("onnxruntime_28_fused", {"inference_backend": "onnxruntime", "backend_num_threads": 28, "fuse_conv_bn": True}),
)


def prepare_samples(
    config: dict[str, Any], limit: int, image_paths: list[str] | None = None
) -> list[dict[str, Any]]:
    """Select images and extract each patch set exactly once."""
    if image_paths:
        selected = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Benchmark image does not exist: {path}")
            selected.append({
                "identity_id": path.parent.name,
                "image_id": path.stem,
                "image_path": str(path),
                "split": "",
            })
        if len(selected) > limit:
            selected = selected[:limit]
    else:
        discovery = copy.deepcopy(config)
        discovery.setdefault("data", {})["validate_readable"] = False
        rows = sorted(load_raw_image_metadata(discovery), key=lambda row: (row["identity_id"], row["image_id"]))
        # Two images per identity give both genuine and impostor comparisons.
        selected, counts = [], {}
        for row in rows:
            identity = row["identity_id"]
            if counts.get(identity, 0) >= 2:
                continue
            selected.append(row)
            counts[identity] = counts.get(identity, 0) + 1
            if len(selected) >= limit:
                break
    sift = build_sift(config)
    patch_config = dict(config.get("patch", {}))
    samples = []
    for row in selected:
        image = imread_grayscale(row["image_path"])
        if image is None:
            raise ValueError(f"Unreadable benchmark image: {row['image_path']}")
        keypoints = detect_keypoints_only(image, config, sift)
        max_keypoints = int(config.get("keypoint_filter", {}).get("max_keypoints", 400))
        if max_keypoints > 0 and len(keypoints) > max_keypoints:
            order = np.argsort([kp.response for kp in keypoints])[::-1][:max_keypoints]
            keypoints = [keypoints[int(index)] for index in order]
        keypoints, patches, _ = extract_keypoint_patches(
            image, keypoints,
            crop_size=int(patch_config.get("crop_size", 32)),
            out_size=int(patch_config.get("out_size", 32)),
            min_overlap_ratio=float(patch_config.get("min_overlap_ratio", 0.55)),
            normalize=bool(patch_config.get("normalize", True)),
        )
        if not len(patches):
            raise ValueError(f"No usable benchmark patches: {row['image_path']}")
        samples.append({
            "row": row, "image": image, "keypoints": keypoints, "patches": patches,
            "patch_sha256": hashlib.sha256(patches.tobytes()).hexdigest(),
        })
    return samples


@torch.inference_mode()
def raw_q(runtime: HardNetDescriptor, patches: np.ndarray) -> np.ndarray:
    """Use the same batching/layout settings as describe(), without postprocessing."""
    size = runtime.fixed_inference_batch_size or runtime.batch_size
    chunks = []
    for start in range(0, len(patches), size):
        batch = torch.from_numpy(np.ascontiguousarray(patches[start : start + size, None]))
        count = len(batch)
        if runtime.fixed_inference_batch_size and count < size:
            padded = torch.zeros((size, *batch.shape[1:]), dtype=torch.float32)
            padded[:count].copy_(batch)
            batch = padded
        if runtime.cpu_backend is None:
            raise RuntimeError("ONNX Runtime backend was not initialized.")
        q = runtime.cpu_backend(runtime.cpu_backend.prepare_input(batch))
        chunks.append(q[:count].cpu())
    return torch.cat(chunks).numpy()


def descriptor_difference(reference: np.ndarray, actual: np.ndarray, bitorder: str) -> dict[str, Any]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("Benchmark descriptors must have the same shape and dtype.")
    if reference.dtype == np.uint8:
        xor = np.unpackbits(np.bitwise_xor(reference, actual), axis=1, bitorder=bitorder)
        return {
            "changed_bits": int(xor.sum()), "total_bits": int(xor.size),
            "changed_descriptor_rows": int(np.count_nonzero(xor.any(axis=1))),
            "byte_exact": bool(np.array_equal(reference, actual)),
        }
    return {
        "max_descriptor_abs_error": float(np.max(np.abs(reference - actual))),
        "descriptor_allclose": bool(np.allclose(reference, actual, rtol=1e-4, atol=1e-5)),
    }


def template(runtime: HardNetDescriptor, sample: dict[str, Any], descriptors: np.ndarray) -> dict[str, Any]:
    return hardnet_template_payload(
        sample["row"], sample["keypoints"], descriptors, sample["image"],
        descriptor_kind=runtime.descriptor_kind, descriptor_metric=runtime.descriptor_metric,
        descriptor_dim=runtime.descriptor_dim, descriptor_storage=runtime.descriptor_storage,
        descriptor_bitorder=runtime.descriptor_bitorder,
        descriptor_transform_name=runtime.descriptor_transform_name,
        descriptor_transform_id=runtime.descriptor_transform_id,
    )


def match_comparisons(templates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare real pairs plus self matches, including positive threshold decisions."""
    threshold = float(config.get("identification", {}).get("match_score_threshold", 0.8))
    results = []
    for query_index in range(len(templates)):
        for gallery_index in range(query_index, len(templates)):
            query, gallery = templates[query_index], templates[gallery_index]
            candidates = build_descriptor_candidates(query, gallery, dict(config.get("matching", {})))
            cv2.setRNGSeed(42)
            match = match_templates_descriptor(query, gallery, config)
            results.append({
                "query_index": query_index, "gallery_index": gallery_index,
                "genuine": query["identity_id"] == gallery["identity_id"],
                "self_match": query_index == gallery_index,
                "candidate_signature": [(c.query_idx, c.gallery_idx, c.distance) for c in candidates],
                "score": float(match["score"]), "accepted": bool(match["score"] >= threshold),
                "num_candidates": int(match["num_candidates"]),
                "unique_inliers": int(match["unique_inliers"]),
            })
    return results


def _measure_variant(arguments: tuple[Any, ...]) -> dict[str, Any]:
    """One fresh process per variant: no shared Torch/oneDNN/thread-pool state."""
    config, patch_sets, variant_name, overrides, warmup, repeats = arguments
    active_config = copy.deepcopy(config)
    active_config.setdefault("model", {}).update({
        "device": "cpu", "inference_precision": "fp32", "fixed_inference_batch_size": 0,
        "batch_size": 512, "cpu_num_threads": 1, "channels_last": False,
        "inference_backend": "onnxruntime", "fuse_conv_bn": False, "profile_inference": True,
        "backend_num_threads": 28, "backend_inter_op_num_threads": 1,
        "backend_graph_optimization": "all",
        **overrides,
    })
    started = time.perf_counter()
    runtime = HardNetDescriptor(active_config)
    startup_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    for _ in range(warmup):
        for patches in patch_sets:
            runtime.describe(patches)
    warmup_ms = (time.perf_counter() - started) * 1000
    measurements, descriptors, raw_features = [], [], []
    for sample_index, patches in enumerate(patch_sets):
        for repeat in range(repeats):
            output, timings = runtime.describe_with_timing(patches)
            measurements.append({
                "variant": variant_name, "sample_index": sample_index,
                "num_patches": len(patches), "repeat": repeat, **timings,
            })
        descriptors.append(output)
        raw_features.append(raw_q(runtime, patches))
    fields = (
        "descriptor_kind", "descriptor_metric", "descriptor_dim", "descriptor_storage",
        "descriptor_bitorder", "descriptor_transform_name", "descriptor_transform_id",
    )
    return {
        "measurements": measurements, "descriptors": descriptors, "raw_q": raw_features,
        "contract": {field: getattr(runtime, field) for field in fields},
        "settings": runtime.inference_settings(),
        "model_initialization_ms": startup_ms, "warmup_ms": warmup_ms,
    }


def run_benchmark(config: dict[str, Any], samples: list[dict[str, Any]], variants: list[tuple[str, dict[str, Any]]], warmup: int, repeats: int) -> dict[str, Any]:
    measurements, results = [], []
    reference_q, reference_descriptors, reference_matches = [], [], []
    patch_sets = [sample["patches"] for sample in samples]
    arguments = [(config, patch_sets, name, overrides, warmup, repeats) for name, overrides in variants]
    # CPU-intensive measurements run sequentially, each in its own new process.
    with multiprocessing.get_context("spawn").Pool(processes=1, maxtasksperchild=1) as pool:
        for (variant_name, _), argument in zip(variants, arguments):
            measured = pool.apply(_measure_variant, (argument,))
            variant_measurements = measured["measurements"]
            measurements.extend(variant_measurements)
            descriptors = measured["descriptors"]
            contract = SimpleNamespace(**measured["contract"])
            templates = [template(contract, sample, desc) for sample, desc in zip(samples, descriptors)]
            matches = match_comparisons(templates, config)
            regressions = []
            for index, sample in enumerate(samples):
                q = measured["raw_q"][index]
                if not results:
                    reference_q.append(q)
                    reference_descriptors.append(descriptors[index])
                regressions.append({
                    "sample_index": index, "num_patches": len(sample["patches"]),
                    "raw_q_max_abs_error": float(np.max(np.abs(q - reference_q[index]))),
                    "raw_q_allclose": bool(np.allclose(q, reference_q[index], rtol=1e-4, atol=1e-5)),
                    **descriptor_difference(reference_descriptors[index], descriptors[index], contract.descriptor_bitorder),
                })
            if not results:
                reference_matches = matches
            match_regressions = [{
                **match,
                "candidates_identical": match["candidate_signature"] == original["candidate_signature"],
                "decision_identical": match["accepted"] == original["accepted"],
                "score_difference": match["score"] - original["score"],
            } for match, original in zip(matches, reference_matches)]
            for entry in match_regressions:
                entry.pop("candidate_signature")
            result = {
                "variant": variant_name, "settings": measured["settings"],
                "model_initialization_ms": measured["model_initialization_ms"],
                "warmup_ms": measured["warmup_ms"],
                "median_inference_ms": float(np.median([row["hardnet_inference_ms"] for row in variant_measurements])),
                "per_sample_median_ms": [float(np.median([row["hardnet_inference_ms"] for row in variant_measurements if row["sample_index"] == index])) for index in range(len(samples))],
                "median_stages_ms": {field: float(np.median([row[field] for row in variant_measurements])) for field in HARDNET_INFERENCE_STAGE_FIELDS},
                "descriptor_regressions": regressions, "match_regressions": match_regressions,
            }
            results.append(result)
            print(f"{variant_name}: median {result['median_inference_ms']:.3f} ms; raw-q close={all(row['raw_q_allclose'] for row in regressions)}; changed bits={sum(row.get('changed_bits', 0) for row in regressions)}", flush=True)
    return {"measurements": measurements, "variants": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_match_new.yaml")))
    parser.add_argument("--limit-images", type=int, default=4)
    parser.add_argument("--images", nargs="+", help="Explicit image paths; enables exact-patch comparisons across projects.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variants", nargs="+", choices=[name for name, _ in DEFAULT_VARIANTS])
    parser.add_argument("--output-dir", default="outputs/cpu_backend_smoke")
    args = parser.parse_args()
    if args.limit_images <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("images/repeats must be positive and warmup cannot be negative")
    config = load_config(args.config)
    samples = prepare_samples(config, args.limit_images, args.images)
    variants = [entry for entry in DEFAULT_VARIANTS if args.variants is None or entry[0] in args.variants]
    payload = run_benchmark(config, samples, variants, args.warmup, args.repeats)
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_csv_rows(output_dir / "cpu_inference_measurements.csv", payload.pop("measurements"))
    payload.update({
        "scope": "small_fixed_real_patch_regression_not_full_FAR_FRR", "baseline_variant": variants[0][0],
        "measurement_isolation": "one_fresh_sequential_process_per_variant",
        "config_path": config["_config_path"], "checkpoint_path": str(resolve_path(config, config["model"]["checkpoint"])),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "torch_version": str(torch.__version__), "warmup_runs": args.warmup, "repeats": args.repeats,
        "samples": [{"row": sample["row"], "num_patches": len(sample["patches"]), "patch_sha256": sample["patch_sha256"]} for sample in samples],
    })
    write_json(output_dir / "cpu_inference_summary.json", payload)
    print(f"Summary: {output_dir / 'cpu_inference_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
