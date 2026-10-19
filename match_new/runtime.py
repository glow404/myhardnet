"""HardNet 工程匹配模块使用的生产运行时。

本模块提供离线模板构建和匹配评估共用的底层能力：
    1. 读取原始灰度指纹图像；
    2. 创建 SIFT 检测器并提取关键点；
    3. 加载 HardNet checkpoint，并在 CPU 或 CUDA 上批量生成描述子。

这里不包含阈值扫描、ROC/FAR/FRR 计算或算法对照实验。工程匹配代码因此不再
依赖 ``val_new`` 验证模块，两个模块可以独立修改、测试和部署。
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from hardnet_train.binary_model import (
    BINARY_DESCRIPTOR_KIND,
    BINARY_STORAGE,
    HAMMING_DISTANCE_METRIC,
    build_binary_descriptor_model,
    pack_binary_codes,
)
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
    checkpoint_model_config,
    normalize_model_architecture,
)
from match_new.descriptor_contract import (
    DEFAULT_BINARY_BITORDER,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    L2_DISTANCE_METRIC,
)
from match_new.hadamard_binarization import (
    HadamardBinarizer,
    hadamard_binarization_enabled,
)
from match_new.inference_optimization import fuse_sequential_conv_bn_eval
from match_new.utils import resolve_path


HARDNET_INFERENCE_STAGE_FIELDS = (
    "hardnet_input_prepare_ms",
    "hardnet_batch_prepare_ms",
    "hardnet_layout_conversion_ms",
    "hardnet_cnn_ms",
    "hardnet_normalization_ms",
    "hardnet_hash_head_ms",
    "hardnet_output_transfer_ms",
    "hardnet_binarization_ms",
    "hardnet_inference_overhead_ms",
)


def get_nested(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """安全读取多层配置；任意一级不存在时返回默认值。"""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def imread_grayscale(path: str | Path) -> np.ndarray | None:
    """读取灰度图像，并兼容 Windows 中文路径。"""

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def build_sift(config: Mapping[str, Any]) -> cv2.SIFT:
    """根据工程匹配配置创建 OpenCV SIFT 关键点检测器。"""

    return cv2.SIFT_create(
        nfeatures=int(get_nested(config, "sift", "nfeatures", default=300)),
        nOctaveLayers=int(
            get_nested(config, "sift", "nOctaveLayers", default=4)
        ),
        contrastThreshold=float(
            get_nested(config, "sift", "contrastThreshold", default=0.03)
        ),
        edgeThreshold=float(
            get_nested(config, "sift", "edgeThreshold", default=17.5)
        ),
        sigma=float(get_nested(config, "sift", "sigma", default=1.70)),
    )


def preprocess_for_sift(
    image: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """按照工程配置执行 SIFT 检测前的可选增强和降噪。"""

    output = image
    if bool(get_nested(config, "sift", "enable_clahe", default=True)):
        tile = get_nested(config, "sift", "clahe_tile", default=[8, 8])
        clahe = cv2.createCLAHE(
            clipLimit=float(
                get_nested(config, "sift", "clahe_clip", default=4.0)
            ),
            tileGridSize=(int(tile[0]), int(tile[1])),
        )
        output = clahe.apply(output)
    if bool(get_nested(config, "sift", "enable_blur", default=True)):
        output = cv2.GaussianBlur(
            output,
            (0, 0),
            sigmaX=float(
                get_nested(config, "sift", "blur_sigma", default=0.8)
            ),
        )
    return output


def detect_sift_keypoints(
    image: np.ndarray,
    sift: cv2.SIFT,
    config: Mapping[str, Any],
) -> list[cv2.KeyPoint]:
    """只检测 SIFT 关键点，不计算工程匹配不需要的 SIFT 描述子。"""

    processed = preprocess_for_sift(image, config)
    return list(sift.detect(processed, None) or [])


def overlap_ratio(
    image_shape: tuple[int, int],
    x: float,
    y: float,
    crop_size: int,
) -> float:
    """计算未旋转裁剪窗口落在原图内部的面积比例。"""

    height, width = image_shape
    half = crop_size / 2.0
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    intersection = (
        max(0.0, ix1 - ix0)
        * max(0.0, iy1 - iy0)
    )
    return intersection / max(float(crop_size * crop_size), 1.0)


def extract_aligned_patch(
    image: np.ndarray,
    keypoint: cv2.KeyPoint,
    crop_size: int,
    out_size: int,
) -> np.ndarray:
    """按关键点位置和方向直接从原图采样局部 patch。

    只为输出 patch 构造反向坐标映射，不再为每个关键点旋转整张图。非整数
    坐标使用双线性插值，越界位置补零。
    """

    x, y = keypoint.pt
    angle = float(keypoint.angle if keypoint.angle >= 0 else 0.0)
    offset_x, offset_y = _centered_patch_grid(int(crop_size))
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    map_x = (
        float(x) + cos_a * offset_x - sin_a * offset_y
    ).astype(np.float32)
    map_y = (
        float(y) + sin_a * offset_x + cos_a * offset_y
    ).astype(np.float32)
    patch = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(
        patch,
        (int(out_size), int(out_size)),
        interpolation=cv2.INTER_AREA,
    )


@lru_cache(maxsize=8)
def _centered_patch_grid(crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """缓存局部 patch 相对中心的坐标网格。"""

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    offsets = (
        np.arange(crop_size, dtype=np.float32)
        - (float(crop_size) - 1.0) / 2.0
    )
    return np.meshgrid(offsets, offsets)


def normalize_patch(
    patch: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """把灰度 patch 转为 HardNet 输入，并执行单 patch 标准化。"""

    values = patch.astype(np.float32)
    if not normalize:
        return values / 255.0
    std = max(float(values.std()), 1e-6)
    return (values - float(values.mean())) / std


def patchable_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    config: Mapping[str, Any],
) -> tuple[list[cv2.KeyPoint], np.ndarray, list[int]]:
    """过滤严重越界的关键点，并用局部反向坐标采样构建对齐的 patch 数组。

    标准化在全部 patch 上批量完成。
    """

    crop_size = int(
        get_nested(config, "patch", "crop_size", default=32)
    )
    out_size = int(
        get_nested(config, "patch", "out_size", default=32)
    )
    min_overlap = float(
        get_nested(config, "patch", "min_overlap_ratio", default=0.55)
    )
    normalize_flag = bool(
        get_nested(config, "patch", "normalize", default=True)
    )

    # ── 过滤越界关键点 ──
    selected_keypoints: list[cv2.KeyPoint] = []
    selected_indices: list[int] = []
    for index, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        if (
            overlap_ratio(
                image.shape[:2],
                float(x),
                float(y),
                crop_size,
            )
            < min_overlap
        ):
            continue
        selected_keypoints.append(keypoint)
        selected_indices.append(index)

    if not selected_keypoints:
        return (
            [],
            np.zeros((0, out_size, out_size), dtype=np.float32),
            [],
        )

    patches_raw = np.stack(
        [
            extract_aligned_patch(
                image, kp, crop_size=crop_size, out_size=out_size,
            ).astype(np.float32)
            for kp in selected_keypoints
        ]
    )

    # ── 批量标准化 ──
    if normalize_flag:
        mean = patches_raw.mean(axis=(1, 2), keepdims=True)
        std = np.maximum(patches_raw.std(axis=(1, 2), keepdims=True), 1e-6)
        patches = (patches_raw - mean) / std
    else:
        patches = patches_raw / 255.0

    return (
        selected_keypoints,
        patches.astype(np.float32),
        selected_indices,
    )


class HardNetDescriptor:
    """HardNet descriptor runtime with an ONNX Runtime CPU backbone.

    PyTorch remains responsible for model loading and descriptor postprocessing;
    only the CNN backbone executes through ONNX Runtime.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.inference_backend = str(
            get_nested(config, "model", "inference_backend", default="onnxruntime")
        ).strip().lower()
        if self.inference_backend != "onnxruntime":
            raise ValueError(
                "Only model.inference_backend=onnxruntime is supported; "
                f"got {self.inference_backend!r}."
            )
        requested_device = str(
            get_nested(config, "model", "device", default="cpu")
        ).strip().lower()
        if requested_device != "cpu":
            raise ValueError("ONNX Runtime backend requires model.device=cpu.")
        self.device = torch.device("cpu")
        self.cpu_backend = None
        self.backend_model_path = ""
        self.backend_initialization_ms = 0.0
        self.backend_export_ms = 0.0
        self.backend_cache_prepare_ms = 0.0
        self.backend_cache_hit: bool | None = None
        self.last_timings: dict[str, Any] = {}
        self.profile_inference = bool(
            get_nested(config, "model", "profile_inference", default=False)
        )
        self.cpu_num_threads = int(
            get_nested(config, "model", "cpu_num_threads", default=1) or 1
        )
        if self.cpu_num_threads <= 0:
            raise ValueError("model.cpu_num_threads must be positive.")
        torch.set_num_threads(self.cpu_num_threads)
        self.inference_precision = "fp32"
        self.batch_size = int(
            get_nested(config, "model", "batch_size", default=512)
        )
        if self.batch_size <= 0:
            raise ValueError("model.batch_size 必须大于 0")
        self.fixed_inference_batch_size = int(
            get_nested(
                config,
                "model",
                "fixed_inference_batch_size",
                default=0,
            )
        )
        if self.fixed_inference_batch_size < 0:
            raise ValueError(
                "model.fixed_inference_batch_size 不能小于 0"
            )
        self.patch_size = int(
            get_nested(config, "patch", "out_size", default=32)
        )
        if self.patch_size <= 0:
            raise ValueError("patch.out_size 必须大于 0")
        checkpoint_path = resolve_path(
            config,
            get_nested(config, "model", "checkpoint"),
        )
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        descriptor_kind = str(
            checkpoint.get("descriptor_kind", FLOAT_DESCRIPTOR_KIND)
        ).strip().lower()
        descriptor_metric = str(
            checkpoint.get("descriptor_metric", L2_DISTANCE_METRIC)
        ).strip().lower()
        supported = {
            (FLOAT_DESCRIPTOR_KIND, L2_DISTANCE_METRIC),
            (BINARY_DESCRIPTOR_KIND, HAMMING_DISTANCE_METRIC),
        }
        if (descriptor_kind, descriptor_metric) not in supported:
            raise ValueError(
                "Unsupported descriptor checkpoint contract: "
                f"kind={descriptor_kind}, metric={descriptor_metric}, path={checkpoint_path}."
            )
        requested_kind = str(
            get_nested(config, "model", "descriptor_kind", default="auto")
        ).strip().lower()
        output_descriptor_kind = (
            BINARY_DESCRIPTOR_KIND
            if hadamard_binarization_enabled(config)
            else descriptor_kind
        )
        if requested_kind not in {"", "auto", output_descriptor_kind}:
            raise ValueError(
                "Configured descriptor kind does not match runtime output: "
                f"config={requested_kind}, output={output_descriptor_kind}."
            )

        saved_descriptor_dim = checkpoint_descriptor_dim(checkpoint)
        requested_descriptor_dim = get_nested(
            config,
            "model",
            "descriptor_dim",
            default="auto",
        )
        requested_descriptor_dim_text = str(
            requested_descriptor_dim if requested_descriptor_dim is not None else ""
        ).strip().lower()
        if requested_descriptor_dim_text not in {"", "auto"}:
            configured_descriptor_dim = int(requested_descriptor_dim_text)
            if configured_descriptor_dim != saved_descriptor_dim:
                raise ValueError(
                    "Configured descriptor dimension does not match checkpoint: "
                    f"config={configured_descriptor_dim}, checkpoint={saved_descriptor_dim}."
                )

        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        requested_architecture = str(
            get_nested(config, "model", "architecture", default="auto")
        ).strip().lower()
        if descriptor_kind == FLOAT_DESCRIPTOR_KIND:
            saved_architecture = checkpoint_model_architecture(checkpoint)
            if requested_architecture in {"", "auto"}:
                architecture = saved_architecture
            else:
                architecture = normalize_model_architecture(requested_architecture)
                if architecture != saved_architecture:
                    raise ValueError(
                        "Configured model architecture does not match checkpoint: "
                        f"config={architecture}, checkpoint={saved_architecture}."
                    )

            saved_model_config = checkpoint_model_config(checkpoint)
            runtime_model_config = config.get("model")
            if isinstance(runtime_model_config, Mapping):
                for key in ("dropout", "final_bn_affine"):
                    if key in runtime_model_config:
                        saved_model_config[key] = runtime_model_config[key]
            saved_model_config["architecture"] = architecture
            saved_model_config["descriptor_dim"] = saved_descriptor_dim
            self.model = build_descriptor_model(saved_model_config)
            self.model.load_state_dict(state)
            self.descriptor_storage = FLOAT32_STORAGE
            self.descriptor_bitorder = ""
        else:
            binary_architecture = str(
                checkpoint.get("model_architecture", "residual_binary_hash_v1")
            ).strip().lower()
            if requested_architecture not in {"", "auto", binary_architecture}:
                raise ValueError(
                    "Configured model architecture does not match binary checkpoint: "
                    f"config={requested_architecture}, checkpoint={binary_architecture}."
                )
            saved_config = checkpoint.get("resolved_config", checkpoint.get("config", {}))
            saved_config = saved_config if isinstance(saved_config, Mapping) else {}
            saved_backbone = saved_config.get("backbone", {})
            saved_backbone = saved_backbone if isinstance(saved_backbone, Mapping) else {}
            backbone_architecture = str(
                checkpoint.get(
                    "backbone_architecture",
                    saved_backbone.get("architecture", "hardnet"),
                )
            )
            float_descriptor_dim = int(
                checkpoint.get(
                    "float_descriptor_dim",
                    saved_backbone.get("descriptor_dim", 128),
                )
            )
            hidden_dim = int(
                checkpoint.get("hash_head_hidden_dim", saved_descriptor_dim * 2)
            )
            binary_config = {
                "hash_bits": saved_descriptor_dim,
                "hidden_multiplier": hidden_dim / saved_descriptor_dim,
                "dropout": float(checkpoint.get("hash_head_dropout", 0.1)),
                "temperature_start": float(
                    checkpoint.get("quantization_temperature", 1.0)
                ),
                "backbone_trainable": bool(
                    checkpoint.get("backbone_trainable", False)
                ),
                "bitorder": str(
                    checkpoint.get("binary_bitorder", DEFAULT_BINARY_BITORDER)
                ),
            }
            backbone_config = {
                "architecture": backbone_architecture,
                "descriptor_dim": float_descriptor_dim,
                "dropout": float(saved_backbone.get("dropout", 0.1)),
            }
            load_error: RuntimeError | None = None
            self.model = None
            for final_bn_affine in (False, True):
                backbone_config["final_bn_affine"] = final_bn_affine
                candidate = build_binary_descriptor_model(
                    build_descriptor_model(backbone_config),
                    binary_config,
                )
                try:
                    candidate.load_state_dict(state)
                except RuntimeError as exc:
                    load_error = exc
                    continue
                self.model = candidate
                break
            if self.model is None:
                raise ValueError(
                    f"Binary checkpoint model structure is incompatible: {checkpoint_path}"
                ) from load_error
            self.descriptor_storage = str(
                checkpoint.get("binary_storage", BINARY_STORAGE)
            ).strip().lower()
            self.descriptor_bitorder = str(
                checkpoint.get("binary_bitorder", DEFAULT_BINARY_BITORDER)
            ).strip().lower()
            requested_storage = str(
                get_nested(
                    config,
                    "model",
                    "binary_storage",
                    default=self.descriptor_storage,
                )
            ).strip().lower()
            requested_bitorder = str(
                get_nested(
                    config,
                    "model",
                    "binary_bitorder",
                    default="auto",
                )
            ).strip().lower()
            if requested_storage not in {"", "auto", self.descriptor_storage}:
                raise ValueError(
                    "Configured binary storage does not match checkpoint: "
                    f"config={requested_storage}, checkpoint={self.descriptor_storage}."
                )
            if requested_bitorder not in {"", "auto", self.descriptor_bitorder}:
                raise ValueError(
                    "Configured binary bitorder does not match checkpoint: "
                    f"config={requested_bitorder}, checkpoint={self.descriptor_bitorder}."
                )

        self.model_descriptor_kind = descriptor_kind
        self.hadamard_binarizer: HadamardBinarizer | None = None
        self.descriptor_transform_metadata: dict[str, Any] = {}
        if hadamard_binarization_enabled(config):
            if descriptor_kind != FLOAT_DESCRIPTOR_KIND:
                raise ValueError(
                    "Hadamard post-binarization requires a float/L2 checkpoint; "
                    f"got kind={descriptor_kind}, metric={descriptor_metric}."
                )
            self.hadamard_binarizer = HadamardBinarizer.from_config(config)
            if saved_descriptor_dim != self.hadamard_binarizer.order:
                raise ValueError(
                    "Hadamard order must equal the float checkpoint descriptor dimension: "
                    f"order={self.hadamard_binarizer.order}, "
                    f"descriptor_dim={saved_descriptor_dim}."
                )
            descriptor_kind = BINARY_DESCRIPTOR_KIND
            descriptor_metric = HAMMING_DISTANCE_METRIC
            saved_descriptor_dim = self.hadamard_binarizer.order
            self.descriptor_storage = BINARY_STORAGE
            self.descriptor_bitorder = self.hadamard_binarizer.bitorder
            requested_storage = str(
                get_nested(config, "model", "binary_storage", default="auto")
            ).strip().lower()
            requested_bitorder = str(
                get_nested(config, "model", "binary_bitorder", default="auto")
            ).strip().lower()
            if requested_storage not in {"", "auto", self.descriptor_storage}:
                raise ValueError(
                    "Configured binary storage does not match Hadamard output: "
                    f"config={requested_storage}, output={self.descriptor_storage}."
                )
            if requested_bitorder not in {"", "auto", self.descriptor_bitorder}:
                raise ValueError(
                    "Configured binary bitorder does not match Hadamard output: "
                    f"config={requested_bitorder}, output={self.descriptor_bitorder}."
                )
            self.descriptor_transform_metadata = self.hadamard_binarizer.metadata()

        self.descriptor_kind = descriptor_kind
        self.descriptor_metric = descriptor_metric
        self.descriptor_dim = saved_descriptor_dim
        self.descriptor_transform_name = str(
            self.descriptor_transform_metadata.get("descriptor_transform_name", "")
        )
        self.descriptor_transform_id = str(
            self.descriptor_transform_metadata.get("descriptor_transform_id", "")
        )
        self.descriptor_columns = (
            (self.descriptor_dim + 7) // 8
            if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
            else self.descriptor_dim
        )
        self.model.eval()
        self.fuse_conv_bn = bool(
            get_nested(config, "model", "fuse_conv_bn", default=False)
        )
        self.fused_conv_bn_pairs = 0
        if self.fuse_conv_bn:
            self.model, self.fused_conv_bn_pairs = fuse_sequential_conv_bn_eval(self.model)
        self.model.to(device="cpu", memory_format=torch.contiguous_format)
        self.model.eval()

        from match_new.cpu_inference_backends import create_cpu_backend, ensure_onnx_model

        backend_started = time.perf_counter()
        configured_path = get_nested(config, "model", "onnx_path", default=None)
        if configured_path:
            configured_path = resolve_path(dict(config), configured_path)
        export_info: dict[str, Any] = {}
        onnx_path = ensure_onnx_model(
            self.model,
            checkpoint_path=self.checkpoint_path,
            native_binary=self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND,
            patch_size=self.patch_size,
            fused_conv_bn_pairs=self.fused_conv_bn_pairs,
            configured_path=configured_path,
            opset_version=get_nested(config, "model", "onnx_opset_version", default=17),
            export_info=export_info,
        )
        self.backend_export_ms = float(export_info["export_ms"])
        self.backend_cache_prepare_ms = max(
            0.0, (time.perf_counter() - backend_started) * 1000.0 - self.backend_export_ms
        )
        self.backend_cache_hit = bool(export_info["cache_hit"])
        self.backend_model_path = str(onnx_path)
        backend_started = time.perf_counter()
        self.cpu_backend = create_cpu_backend(self.inference_backend, onnx_path, config)
        self.backend_initialization_ms = (time.perf_counter() - backend_started) * 1000.0
        expected_q_dim = (
            self.model.float_descriptor_dim
            if self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND
            else self.descriptor_dim
        )
        if (
            self.cpu_backend.descriptor_dim != expected_q_dim
            or self.cpu_backend.patch_height != self.patch_size
            or self.cpu_backend.patch_width != self.patch_size
        ):
            raise ValueError("ONNX Runtime raw-q shape does not match the loaded model.")

    def inference_settings(self) -> dict[str, Any]:
        """Return effective ONNX Runtime settings and separate startup timings."""

        return {
            "inference_backend": self.inference_backend,
            "device": str(self.device),
            "inference_precision": self.inference_precision,
            "batch_size": self.batch_size,
            "fixed_inference_batch_size": self.fixed_inference_batch_size,
            "cpu_num_threads": int(torch.get_num_threads()),
            "fused_conv_bn_pairs": self.fused_conv_bn_pairs,
            "backend_model_path": self.backend_model_path,
            "backend_export_ms": self.backend_export_ms,
            "backend_cache_prepare_ms": self.backend_cache_prepare_ms,
            "backend_cache_hit": self.backend_cache_hit,
            "backend_initialization_ms": self.backend_initialization_ms,
            "backend_settings": self.cpu_backend.settings(),
        }

    @torch.inference_mode()
    def describe(self, patches: np.ndarray) -> np.ndarray:
        """Generate descriptors through ONNX Runtime without backend fallback."""

        self.last_timings = {}
        if self.profile_inference:
            return self.describe_with_timing(patches)[0]
        return self._describe_impl(patches)

    @torch.inference_mode()
    def describe_with_timing(
        self, patches: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Time CPU FP32 inference stages; startup/export/compilation are separate."""

        total_started = time.perf_counter()
        timings = {field: 0.0 for field in HARDNET_INFERENCE_STAGE_FIELDS}
        descriptors = self._describe_impl(patches, timings=timings)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        timings["hardnet_inference_overhead_ms"] = max(
            0.0, total_ms - sum(timings.values())
        )
        self.last_timings = {
            **timings,
            "hardnet_inference_ms": total_ms,
            "hardnet_inference_backend": self.inference_backend,
            "hardnet_inference_device": str(self.device),
            "hardnet_inference_precision": self.inference_precision,
            "hardnet_inference_cpu_threads": int(torch.get_num_threads()),
            "hardnet_fused_conv_bn_pairs": self.fused_conv_bn_pairs,
        }
        return descriptors, dict(self.last_timings)

    def warmup(self, runs: int, *, batch_size: int | None = None) -> None:
        """预热独立计时；动态模式可传入预计的真实 patch 数。"""

        warmup_runs = max(0, int(runs))
        if warmup_runs == 0:
            return
        warmup_batch_size = int(
            batch_size if batch_size is not None
            else self.fixed_inference_batch_size or self.batch_size
        )
        if warmup_batch_size <= 0:
            raise ValueError("warmup batch_size must be positive.")
        patches = np.zeros(
            (warmup_batch_size, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        for _ in range(warmup_runs):
            self.describe(patches)
        self.last_timings = {}

    def _describe_impl(
        self,
        patches: np.ndarray,
        *,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """主干以外沿用原后处理；分段计时仅测实际执行的工作。"""

        def start_stage() -> float:
            return time.perf_counter() if timings is not None else 0.0

        def end_stage(field: str, started: float) -> None:
            if timings is not None:
                timings[field] += (time.perf_counter() - started) * 1000.0

        raw = np.asarray(patches)
        if (
            raw.ndim not in {3, 4}
            or tuple(raw.shape[-2:]) != (self.patch_size, self.patch_size)
            or (raw.ndim == 4 and raw.shape[1] != 1)
        ):
            raise ValueError(
                f"HardNet patches must be [N,{self.patch_size},{self.patch_size}] "
                f"or [N,1,{self.patch_size},{self.patch_size}], got {raw.shape}."
            )
        if len(raw) == 0:
            dtype = np.uint8 if self.descriptor_kind == BINARY_DESCRIPTOR_KIND else np.float32
            return np.zeros((0, self.descriptor_columns), dtype=dtype)
        stage_started = start_stage()
        patch_array = np.ascontiguousarray(raw, dtype=np.float32)
        if not np.all(np.isfinite(patch_array)):
            raise ValueError("HardNet patches contain NaN or infinite values.")
        if patch_array.ndim == 3:
            patch_array = patch_array[:, None, :, :]
        host_tensor = torch.from_numpy(patch_array)
        end_stage("hardnet_input_prepare_ms", stage_started)

        inference_batch_size = self.fixed_inference_batch_size or self.batch_size
        outputs: list[torch.Tensor] = []
        for start in range(0, len(patch_array), inference_batch_size):
            stage_started = start_stage()
            source_batch = host_tensor[start : start + inference_batch_size]
            valid_count = len(source_batch)
            if self.fixed_inference_batch_size and valid_count < inference_batch_size:
                padded_batch = torch.zeros(
                    (inference_batch_size, *source_batch.shape[1:]),
                    dtype=source_batch.dtype,
                )
                padded_batch[:valid_count].copy_(source_batch)
                source_batch = padded_batch
            batch = source_batch
            end_stage("hardnet_batch_prepare_ms", stage_started)

            stage_started = start_stage()
            backend_batch = self.cpu_backend.prepare_input(batch)
            end_stage("hardnet_layout_conversion_ms", stage_started)
            stage_started = start_stage()
            q = self.cpu_backend(backend_batch)
            end_stage("hardnet_cnn_ms", stage_started)
            if self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND:
                stage_started = start_stage()
                logits = self.model.hash_head(q)
                end_stage("hardnet_hash_head_ms", stage_started)
                stage_started = start_stage()
                batch_output = pack_binary_codes(
                    (logits >= 0).to(torch.uint8), self.model.bitorder
                )
                end_stage("hardnet_binarization_ms", stage_started)
            else:
                stage_started = start_stage()
                batch_output = F.normalize(q, p=2, dim=1)
                end_stage("hardnet_normalization_ms", stage_started)
            outputs.append(batch_output[:valid_count])

        stage_started = start_stage()
        output = torch.cat(outputs, dim=0).cpu()
        if self.hadamard_binarizer is not None:
            float_descriptors = output.float().numpy()
            end_stage("hardnet_output_transfer_ms", stage_started)
            stage_started = start_stage()
            descriptors = self.hadamard_binarizer.binarize(float_descriptors, packed=True)
            end_stage("hardnet_binarization_ms", stage_started)
        else:
            descriptors = (
                output.to(dtype=torch.uint8).numpy()
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else output.float().numpy()
            )
            end_stage("hardnet_output_transfer_ms", stage_started)
        if descriptors.shape != (len(patch_array), self.descriptor_columns):
            raise ValueError(
                "Descriptor model output shape mismatch: "
                f"expected [{len(patch_array)},{self.descriptor_columns}], "
                f"got {descriptors.shape}."
            )
        if not np.all(np.isfinite(descriptors)):
            raise ValueError("Descriptor model output contains NaN or infinite values.")
        return descriptors
