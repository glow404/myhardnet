"""HardNet 工程匹配模块使用的生产运行时。

本模块只提供注册和在线解锁都需要的底层能力：
    1. 读取原始灰度指纹图像；
    2. 创建 SIFT 检测器并提取关键点；
    3. 按训练阶段相同的方向裁剪并归一化 patch；
    4. 加载 HardNet checkpoint，并在 CPU 或 CUDA 上批量生成描述子。

这里不包含阈值扫描、ROC/FAR/FRR 计算或算法对照实验。工程匹配代码因此不再
依赖 ``val_new`` 验证模块，两个模块可以独立修改、测试和部署。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from hardnet_train.model import HardNet
from match_new.utils import resolve_path


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


def choose_device(
    raw_device: str,
    *,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """解析匹配设备，并在 CUDA 不可用时按配置回退到 CPU。

    ``auto`` 始终优先选择 CUDA，没有可用 CUDA 时选择 CPU。即使显式配置
    ``cuda``，默认也会回退 CPU 并发出警告；只有关闭 ``fallback_to_cpu`` 后
    才会将 CUDA 不可用视为错误。
    """

    requested = str(raw_device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if fallback_to_cpu:
            warnings.warn(
                "CUDA 不可用，HardNet 匹配将自动回退到 CPU。",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        raise RuntimeError(
            "工程匹配配置要求 CUDA，但当前 PyTorch 是 CPU 版本或 CUDA 初始化失败。"
            "请安装 CUDA 版 PyTorch，并确认 torch.cuda.is_available() 返回 True；"
            "如需显式使用 CPU，请设置 model.device=cpu。"
        )
    return torch.device(requested)


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
    """按关键点方向旋转原图，再裁剪与训练阶段一致的局部 patch。

    当前保留已有模型使用的旧裁剪语义。后续若切换成直接局部仿射采样，必须让
    训练数据生成和本函数同时切换，并重新训练模型、标定阈值。
    """

    x, y = keypoint.pt
    angle = float(keypoint.angle if keypoint.angle >= 0 else 0.0)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (float(x), float(y)),
        angle,
        1.0,
    )
    rotated = cv2.warpAffine(
        image,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    patch = cv2.getRectSubPix(
        rotated,
        patchSize=(int(crop_size), int(crop_size)),
        center=(float(x), float(y)),
    )
    return cv2.resize(
        patch,
        (int(out_size), int(out_size)),
        interpolation=cv2.INTER_AREA,
    )


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
    """过滤严重越界的关键点，并构建与关键点行号对齐的 patch 数组。"""

    crop_size = int(
        get_nested(config, "patch", "crop_size", default=64)
    )
    out_size = int(
        get_nested(config, "patch", "out_size", default=32)
    )
    min_overlap = float(
        get_nested(config, "patch", "min_overlap_ratio", default=0.55)
    )
    normalize = bool(
        get_nested(config, "patch", "normalize", default=True)
    )
    selected_keypoints: list[cv2.KeyPoint] = []
    selected_indices: list[int] = []
    patches: list[np.ndarray] = []
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
        patches.append(
            normalize_patch(
                extract_aligned_patch(
                    image,
                    keypoint,
                    crop_size=crop_size,
                    out_size=out_size,
                ),
                normalize=normalize,
            )
        )
        selected_keypoints.append(keypoint)
        selected_indices.append(index)
    if not patches:
        return (
            [],
            np.zeros((0, out_size, out_size), dtype=np.float32),
            [],
        )
    return (
        selected_keypoints,
        np.stack(patches).astype(np.float32),
        selected_indices,
    )


class HardNetDescriptor:
    """工程注册和在线解锁共用的 HardNet 批量推理器。

    CUDA 模式支持 FP16/BF16 autocast、channels-last、锁页内存异步传输和
    cuDNN benchmark。所有 batch 在 GPU 上拼接后只回传一次，减少同步次数。
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.fallback_to_cpu = bool(
            get_nested(
                config,
                "model",
                "fallback_to_cpu",
                default=True,
            )
        )
        self.device = choose_device(
            str(get_nested(config, "model", "device", default="auto")),
            fallback_to_cpu=self.fallback_to_cpu,
        )
        self.batch_size = int(
            get_nested(config, "model", "batch_size", default=512)
        )
        self.pin_memory = (
            self.device.type == "cuda"
            and bool(
                get_nested(config, "model", "pin_memory", default=True)
            )
        )
        self.channels_last = (
            self.device.type == "cuda"
            and bool(
                get_nested(config, "model", "channels_last", default=True)
            )
        )
        self._configure_precision(config)
        self._configure_cuda_backend(config)

        checkpoint_path = resolve_path(
            config,
            get_nested(config, "model", "checkpoint"),
        )
        self.model = HardNet(
            dropout=float(
                get_nested(config, "model", "dropout", default=0.1)
            ),
            final_bn_affine=bool(
                get_nested(
                    config,
                    "model",
                    "final_bn_affine",
                    default=False,
                )
            ),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state)
        try:
            self.model.to(self.device)
            if self.channels_last:
                self.model.to(memory_format=torch.channels_last)
        except RuntimeError as exc:
            if self.device.type != "cuda" or not self.fallback_to_cpu:
                raise
            warnings.warn(
                f"HardNet 模型初始化 CUDA 失败，将回退到 CPU：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._activate_cpu_fallback()
        self.model.eval()

    def _activate_cpu_fallback(self) -> None:
        """把模型和推理选项切换为 CPU FP32。"""

        self.device = torch.device("cpu")
        self.pin_memory = False
        self.channels_last = False
        self.amp_enabled = False
        self.amp_dtype = None
        self.inference_precision = "fp32"
        self.model.to(
            device=self.device,
            memory_format=torch.contiguous_format,
        )

    def _configure_precision(self, config: Mapping[str, Any]) -> None:
        """解析工程推理精度，并配置 autocast 数据类型。"""

        precision = str(
            get_nested(
                config,
                "model",
                "inference_precision",
                default="fp32",
            )
        ).strip().lower()
        if precision in {"", "off", "none", "fp32", "float32"}:
            self.amp_enabled = False
            self.amp_dtype: torch.dtype | None = None
            self.inference_precision = "fp32"
            return
        if precision in {"fp16", "float16"}:
            self.amp_enabled = self.device.type == "cuda"
            self.amp_dtype = (
                torch.float16 if self.amp_enabled else None
            )
            self.inference_precision = (
                "fp16" if self.amp_enabled else "fp32"
            )
            return
        if precision in {"bf16", "bfloat16"}:
            if (
                self.device.type == "cuda"
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "当前 GPU/PyTorch 不支持 BF16 推理，请设置 "
                    "model.inference_precision=fp16。"
                )
            self.amp_enabled = self.device.type == "cuda"
            self.amp_dtype = (
                torch.bfloat16 if self.amp_enabled else None
            )
            self.inference_precision = (
                "bf16" if self.amp_enabled else "fp32"
            )
            return
        raise ValueError(
            f"不支持的 model.inference_precision: {precision!r}"
        )

    def _configure_cuda_backend(
        self,
        config: Mapping[str, Any],
    ) -> None:
        """配置只在 CUDA 模式生效的 cuDNN 和 TF32 选项。"""

        if self.device.type != "cuda":
            return
        torch.backends.cudnn.benchmark = bool(
            get_nested(
                config,
                "model",
                "cudnn_benchmark",
                default=True,
            )
        )
        allow_tf32 = bool(
            get_nested(
                config,
                "model",
                "allow_tf32",
                default=True,
            )
        )
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(
            "high" if allow_tf32 else "highest"
        )

    @torch.inference_mode()
    def describe(self, patches: np.ndarray) -> np.ndarray:
        """批量生成描述子；CUDA 运行失败时自动切换 CPU 并重试一次。"""

        try:
            return self._describe_impl(patches)
        except RuntimeError as exc:
            if self.device.type != "cuda" or not self.fallback_to_cpu:
                raise
            warnings.warn(
                f"HardNet CUDA 推理失败，将回退到 CPU 并重试：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._activate_cpu_fallback()
            return self._describe_impl(patches)

    def _describe_impl(self, patches: np.ndarray) -> np.ndarray:
        """在当前设备上执行一次描述子批量推理。"""

        if len(patches) == 0:
            return np.zeros((0, 128), dtype=np.float32)
        patch_array = np.ascontiguousarray(
            patches,
            dtype=np.float32,
        )
        if patch_array.ndim == 3:
            patch_array = patch_array[:, None, :, :]
        host_tensor = torch.from_numpy(patch_array)
        if self.pin_memory:
            host_tensor = host_tensor.pin_memory()

        outputs: list[torch.Tensor] = []
        for start in range(0, len(patch_array), self.batch_size):
            batch = host_tensor[
                start : start + self.batch_size
            ].to(
                self.device,
                non_blocking=self.pin_memory,
            )
            if self.channels_last:
                batch = batch.contiguous(
                    memory_format=torch.channels_last
                )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                outputs.append(self.model(batch))
        if not outputs:
            return np.zeros((0, 128), dtype=np.float32)
        return torch.cat(outputs, dim=0).float().cpu().numpy()
