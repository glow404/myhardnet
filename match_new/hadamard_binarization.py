"""使用 Sylvester 哈达玛变换进行描述子二值化。

二值化流程为 ``bit = sign(H_n f)``，其中 ``H_n`` 是 Sylvester 型哈达玛
矩阵。实现使用快速沃尔什-哈达玛变换（FWHT），不会显式创建 ``n×n`` 矩阵。

这里不再使用随机置换矩阵 P 或对角符号矩阵 D。只要描述子维度、位序和零值
约定相同，训练后处理、模板构建和离线评估会使用完全相同的确定性变换，因而
不需要保存或加载额外的 P/D state 文件。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


HADAMARD_ORDER = 256
HADAMARD_TRANSFORM_NAME = "hadamard_fwht_v2"


def _mapping_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """读取 Hadamard 配置，并兼容完整配置和直接配置段。"""

    if config is None or not isinstance(config, Mapping):
        return {}
    for name in ("hadamard_binarization", "hadamard_binary", "hadamard"):
        value = config.get(name)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping.")
            return dict(value)
    for parent_name in ("model", "matching"):
        parent = config.get(parent_name)
        if not isinstance(parent, Mapping):
            continue
        for name in ("hadamard_binarization", "hadamard_binary", "hadamard"):
            value = parent.get(name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{parent_name}.{name} must be a mapping.")
                return dict(value)
    if any(key in config for key in ("enabled", "order", "hadamard_order")):
        return dict(config)
    return {}


def hadamard_binarization_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """返回 Hadamard 二值化配置段。"""

    return _mapping_section(config)


def hadamard_binarization_enabled(config: Mapping[str, Any] | None) -> bool:
    """判断是否启用浮点 HardNet 的 Hadamard 二值化后处理。"""

    return bool(_mapping_section(config).get("enabled", False))


def _configured_value(section: Mapping[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name not in section:
            continue
        value = section[name]
        if value is None or (isinstance(value, str) and value == ""):
            continue
        return value
    return default


def resolve_hadamard_order(config: Mapping[str, Any] | None) -> int:
    """解析 Hadamard 阶数；默认使用 256 阶。"""

    section = _mapping_section(config)
    order = int(
        _configured_value(
            section,
            "order",
            "hadamard_order",
            "dimension",
            "transform_dim",
            default=HADAMARD_ORDER,
        )
    )
    if order < 2 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a power of two >= 2, got {order}.")
    return order


def _normalize_bitorder(value: Any) -> str:
    bitorder = str(value if value is not None else "little").strip().lower()
    if bitorder not in {"little", "big"}:
        raise ValueError(
            f"Hadamard binary bitorder must be 'little' or 'big', got {value!r}."
        )
    return bitorder


def _normalize_zero_is_one(section: Mapping[str, Any]) -> bool:
    value = _configured_value(
        section,
        "zero_is_one",
        "sign_zero_is_one",
        "zero_bit_is_one",
        default=True,
    )
    return bool(value)


def resolve_hadamard_bitorder(config: Mapping[str, Any] | None) -> str:
    """解析二值描述子的打包位序。"""

    return _normalize_bitorder(
        _configured_value(
            _mapping_section(config),
            "bitorder",
            "binary_bitorder",
            default="little",
        )
    )


def resolve_hadamard_zero_is_one(config: Mapping[str, Any] | None) -> bool:
    """解析零系数的符号约定。"""

    return _normalize_zero_is_one(_mapping_section(config))


def _transform_id(order: int, bitorder: str, zero_is_one: bool) -> str:
    """为不含 P/D 的确定性变换生成稳定标识。"""

    digest = hashlib.sha256()
    digest.update(HADAMARD_TRANSFORM_NAME.encode("ascii"))
    digest.update(np.asarray([int(order)], dtype="<i8").tobytes())
    digest.update(str(bitorder).encode("ascii"))
    digest.update(b"1" if zero_is_one else b"0")
    return f"{HADAMARD_TRANSFORM_NAME}-{digest.hexdigest()[:24]}"


def fwht(
    values: np.ndarray,
    *,
    axis: int = -1,
    normalize: bool = False,
) -> np.ndarray:
    """沿指定轴执行未归一化的 Sylvester FWHT。"""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("fwht expects an array with at least one dimension.")
    axis = int(axis) % array.ndim
    n = int(array.shape[axis])
    if n < 2 or n & (n - 1):
        raise ValueError(f"FWHT length must be a power of two >= 2, got {n}.")
    dtype = np.float64 if np.issubdtype(array.dtype, np.float64) else np.float32
    work = np.moveaxis(np.asarray(array, dtype=dtype), axis, -1).copy()
    width = 1
    while width < n:
        grouped = work.reshape(*work.shape[:-1], n // (2 * width), 2, width)
        left = grouped[..., 0, :].copy()
        right = grouped[..., 1, :].copy()
        grouped[..., 0, :] = left + right
        grouped[..., 1, :] = left - right
        width *= 2
    if normalize:
        work /= math.sqrt(float(n))
    return np.moveaxis(work, -1, axis)


def fwht_torch(values: Any, *, dim: int = -1, normalize: bool = False) -> Any:
    """保留 autograd 图的 Torch 版 FWHT。"""

    import torch

    if not torch.is_tensor(values):
        raise TypeError("fwht_torch expects a torch.Tensor.")
    if values.ndim == 0:
        raise ValueError("fwht_torch expects a tensor with at least one dimension.")
    dim = int(dim) % values.ndim
    n = int(values.shape[dim])
    if n < 2 or n & (n - 1):
        raise ValueError(f"FWHT length must be a power of two >= 2, got {n}.")
    work = values.movedim(dim, -1).clone()
    width = 1
    while width < n:
        grouped = work.reshape(*work.shape[:-1], n // (2 * width), 2, width)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        work = torch.stack((left + right, left - right), dim=-2).reshape(
            *work.shape[:-1], n
        )
        width *= 2
    if normalize:
        work = work / math.sqrt(float(n))
    return work.movedim(-1, dim)


@dataclass(frozen=True)
class HadamardTransformState:
    """不含矩阵状态的确定性 Hadamard 变换参数。"""

    order: int
    bitorder: str
    zero_is_one: bool
    transform_id: str

    @property
    def pd_id(self) -> str:
        """兼容旧报告字段；当前标识不包含 P/D。"""

        return self.transform_id


class HadamardBinarizer:
    """对浮点描述子执行直接 FWHT、sign 和 bit packing。"""

    def __init__(
        self,
        *,
        order: int = HADAMARD_ORDER,
        bitorder: str = "little",
        zero_is_one: bool = True,
    ) -> None:
        order = int(order)
        if order < 2 or order & (order - 1):
            raise ValueError(f"Hadamard order must be a power of two >= 2, got {order}.")
        bitorder = _normalize_bitorder(bitorder)
        zero_is_one = bool(zero_is_one)
        self.state = HadamardTransformState(
            order=order,
            bitorder=bitorder,
            zero_is_one=zero_is_one,
            transform_id=_transform_id(order, bitorder, zero_is_one),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "HadamardBinarizer":
        """从完整工程配置或直接配置段构造二值化器。"""

        return cls(
            order=resolve_hadamard_order(config),
            bitorder=resolve_hadamard_bitorder(config),
            zero_is_one=resolve_hadamard_zero_is_one(config),
        )

    @property
    def order(self) -> int:
        return self.state.order

    @property
    def bitorder(self) -> str:
        return self.state.bitorder

    @property
    def zero_is_one(self) -> bool:
        return self.state.zero_is_one

    @property
    def transform_id(self) -> str:
        return self.state.transform_id

    def transform(self, values: np.ndarray, *, normalize: bool = False) -> np.ndarray:
        """计算 ``Hf``，返回未二值化的 Hadamard 系数。"""

        array = np.asarray(values)
        if array.ndim == 0 or int(array.shape[-1]) != self.order:
            raise ValueError(
                "Hadamard binarizer expects descriptors with last dimension "
                f"{self.order}, got {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Hadamard descriptors must be numeric, got {array.dtype}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Hadamard descriptors contain NaN or infinite values.")
        return fwht(array, axis=-1, normalize=normalize)

    def bits(self, values: np.ndarray) -> np.ndarray:
        """计算未打包的 ``uint8`` 0/1 bits。"""

        transformed = self.transform(values)
        bits = transformed >= 0.0 if self.zero_is_one else transformed > 0.0
        return np.asarray(bits, dtype=np.uint8)

    def binarize(self, values: np.ndarray, *, packed: bool = True) -> np.ndarray:
        """将浮点描述子转换成 bit 或 packed ``uint8`` 描述子。"""

        bits = self.bits(values)
        if not packed:
            return bits
        return np.packbits(bits, axis=-1, bitorder=self.bitorder).astype(np.uint8)

    def __call__(self, values: np.ndarray, *, packed: bool = True) -> np.ndarray:
        return self.binarize(values, packed=packed)

    def transform_torch(self, values: Any, *, normalize: bool = False) -> Any:
        """Torch 版 ``Hf``。"""

        import torch

        if not torch.is_tensor(values):
            raise TypeError("Hadamard transform expects a torch.Tensor.")
        if values.ndim == 0 or int(values.shape[-1]) != self.order:
            raise ValueError(
                "Hadamard torch transform expects last dimension "
                f"{self.order}, got {tuple(values.shape)}."
            )
        if not torch.all(torch.isfinite(values)):
            raise ValueError("Hadamard torch descriptors contain non-finite values.")
        return fwht_torch(values, dim=-1, normalize=normalize)

    def metadata(self) -> dict[str, Any]:
        """返回可写入模板和评估报告的二值化元数据。"""

        return {
            "hadamard_binarization_enabled": True,
            "hadamard_transform_name": HADAMARD_TRANSFORM_NAME,
            "descriptor_transform_name": HADAMARD_TRANSFORM_NAME,
            "hadamard_order": int(self.order),
            "hadamard_bitorder": self.bitorder,
            "hadamard_zero_is_one": bool(self.zero_is_one),
            "hadamard_transform_id": self.transform_id,
            "descriptor_transform_id": self.transform_id,
        }


def binarize_float_descriptors(
    descriptors: np.ndarray,
    *,
    order: int = HADAMARD_ORDER,
    bitorder: str = "little",
    zero_is_one: bool = True,
    packed: bool = True,
) -> np.ndarray:
    """便捷函数：用直接 ``Hf`` 对一批浮点描述子进行二值化。"""

    return HadamardBinarizer(
        order=order,
        bitorder=bitorder,
        zero_is_one=zero_is_one,
    ).binarize(descriptors, packed=packed)


HadamardTransform = HadamardBinarizer
