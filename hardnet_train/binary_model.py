"""独立二值描述子网络与 residual hash head。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from hardnet_train.model import model_architecture, model_descriptor_dim


BINARY_MODEL_ARCHITECTURE = "residual_binary_hash_v1"
BINARY_DESCRIPTOR_KIND = "binary"
HAMMING_DISTANCE_METRIC = "hamming"
BINARY_ENCODING = "sign_zero_is_one"
BINARY_STORAGE = "packed_uint8"


class BinaryDescriptorFeatures(NamedTuple):
    """二值训练所需的 teacher、连续 hash 与量化输出。"""

    teacher_q: torch.Tensor
    teacher_f: torch.Tensor
    logits: torch.Tensor
    continuous: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor


class ResidualHashHead(nn.Module):
    """`float_dim → B` 投影后执行 `B → 2B → B` residual hash 变换。"""

    architecture = BINARY_MODEL_ARCHITECTURE

    def __init__(
        self,
        input_dim: int,
        hash_bits: int,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hash_bits < 8 or hash_bits > 512 or hash_bits % 8 != 0:
            raise ValueError(
                "hash_bits must be an 8-aligned value in [8, 512], "
                f"got {hash_bits}."
            )
        if hidden_multiplier < 1.0:
            raise ValueError(
                f"hidden_multiplier must be >= 1, got {hidden_multiplier}."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = int(input_dim)
        self.hash_bits = int(hash_bits)
        hidden_dim = max(
            self.hash_bits,
            int(round(self.hash_bits * float(hidden_multiplier))),
        )
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(
            self.input_dim,
            self.hash_bits,
            bias=False,
        )
        self.residual_norm = nn.LayerNorm(self.hash_bits)
        self.expand = nn.Linear(self.hash_bits, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=float(dropout))
        self.contract = nn.Linear(self.hidden_dim, self.hash_bits)
        self.residual_scale = nn.Parameter(torch.full((self.hash_bits,), 0.1))
        self.output_norm = nn.LayerNorm(self.hash_bits)
        self.output_projection = nn.Linear(
            self.hash_bits,
            self.hash_bits,
            bias=True,
        )
        self.output_scale = nn.Parameter(torch.full((self.hash_bits,), 0.1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        nn.init.xavier_uniform_(self.contract.weight)
        nn.init.zeros_(self.contract.bias)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.5)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        nn.init.ones_(self.residual_norm.weight)
        nn.init.zeros_(self.residual_norm.bias)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"ResidualHashHead expects [N,{self.input_dim}], got {tuple(features.shape)}."
            )
        base = self.input_projection(self.input_norm(features))
        residual = self.residual_norm(base)
        residual = self.expand(residual)
        residual = self.dropout(self.activation(residual))
        residual = self.contract(residual)
        refined = base + residual * self.residual_scale
        output = self.output_projection(self.output_norm(refined))
        return refined + output * self.output_scale


def pack_binary_codes(codes: torch.Tensor, bitorder: str = "little") -> torch.Tensor:
    """把末维 0/1 bit 确定性打包为 uint8，非 8 倍数位数在尾部补 0。"""

    if codes.ndim < 1:
        raise ValueError("codes must have at least one dimension.")
    order = str(bitorder).strip().lower()
    if order not in {"little", "big"}:
        raise ValueError(f"bitorder must be 'little' or 'big', got {bitorder!r}.")
    values = codes.to(dtype=torch.uint8)
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("codes must contain only 0/1 values.")
    hash_bits = int(values.shape[-1])
    padding = (-hash_bits) % 8
    if padding:
        values = torch.cat(
            [values, values.new_zeros((*values.shape[:-1], padding))],
            dim=-1,
        )
    grouped = values.reshape(*values.shape[:-1], -1, 8)
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128]
        if order == "little"
        else [128, 64, 32, 16, 8, 4, 2, 1],
        dtype=torch.uint8,
        device=values.device,
    )
    return torch.sum(grouped * weights, dim=-1).to(dtype=torch.uint8)


class BinaryDescriptorModel(nn.Module):
    """浮点 backbone 与独立 residual hash head 的组合训练网络。"""

    architecture = BINARY_MODEL_ARCHITECTURE

    def __init__(
        self,
        backbone: nn.Module,
        hash_bits: int = 256,
        hidden_multiplier: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 1.0,
        backbone_trainable: bool = False,
        bitorder: str = "little",
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        normalized_bitorder = str(bitorder).strip().lower()
        if normalized_bitorder not in {"little", "big"}:
            raise ValueError(
                f"bitorder must be 'little' or 'big', got {bitorder!r}."
            )
        if not hasattr(backbone, "forward_features"):
            raise ValueError(
                f"Backbone {type(backbone).__name__} must expose forward_features()."
            )

        self.backbone = backbone
        self.backbone_architecture = model_architecture(backbone)
        self.float_descriptor_dim = model_descriptor_dim(backbone)
        self.hash_bits = int(hash_bits)
        self.descriptor_dim = self.hash_bits
        self.bitorder = normalized_bitorder
        self.hash_head = ResidualHashHead(
            input_dim=self.float_descriptor_dim,
            hash_bits=self.hash_bits,
            hidden_multiplier=hidden_multiplier,
            dropout=dropout,
        )
        self.register_buffer(
            "_quantization_temperature",
            torch.tensor(float(temperature), dtype=torch.float32),
        )
        self.backbone_trainable = False
        self.set_backbone_trainable(backbone_trainable)

    @property
    def quantization_temperature(self) -> float:
        return float(self._quantization_temperature.item())

    def set_quantization_temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"temperature must be positive, got {value}.")
        self._quantization_temperature.fill_(float(value))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_trainable = bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.backbone_trainable)
        if not self.backbone_trainable:
            self.backbone.eval()

    def train(self, mode: bool = True) -> BinaryDescriptorModel:
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()
        return self

    def _backbone_features(self, patches: torch.Tensor):
        if self.backbone_trainable:
            return self.backbone.forward_features(patches)
        with torch.no_grad():
            return self.backbone.forward_features(patches)

    def forward_features(self, patches: torch.Tensor) -> BinaryDescriptorFeatures:
        teacher = self._backbone_features(patches)
        logits = self.hash_head(teacher.q)
        temperature = self._quantization_temperature.to(
            device=logits.device,
            dtype=logits.dtype,
        ).clamp_min(1e-6)
        continuous = torch.tanh(logits / temperature)
        hard_values = torch.where(
            logits >= 0,
            torch.ones_like(logits),
            -torch.ones_like(logits),
        )
        quantized = continuous + (hard_values - continuous).detach()
        codes = logits >= 0
        return BinaryDescriptorFeatures(
            teacher_q=teacher.q,
            teacher_f=teacher.f,
            logits=logits,
            continuous=continuous,
            quantized=quantized,
            codes=codes,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """返回用于可微匹配损失的单位长度 STE 二值向量。"""

        quantized = self.forward_features(patches).quantized
        return F.normalize(quantized, p=2, dim=1)

    @torch.inference_mode()
    def encode_binary(
        self,
        patches: torch.Tensor,
        *,
        packed: bool = False,
    ) -> torch.Tensor:
        """返回确定性 0/1 码；`packed=True` 时按 checkpoint bitorder 打包。"""

        codes = self.forward_features(patches).codes.to(dtype=torch.uint8)
        return pack_binary_codes(codes, self.bitorder) if packed else codes


def build_binary_descriptor_model(
    backbone: nn.Module,
    config: Mapping[str, Any] | None = None,
) -> BinaryDescriptorModel:
    """从 binary_model 配置段构建可训练二值网络。"""

    options = config or {}
    return BinaryDescriptorModel(
        backbone=backbone,
        hash_bits=int(options.get("hash_bits", 256)),
        hidden_multiplier=float(options.get("hidden_multiplier", 2.0)),
        dropout=float(options.get("dropout", 0.1)),
        temperature=float(options.get("temperature_start", 1.0)),
        backbone_trainable=bool(options.get("backbone_trainable", False)),
        bitorder=str(options.get("bitorder", "little")),
    )


def binary_model_metadata(model: BinaryDescriptorModel) -> dict[str, object]:
    """返回 checkpoint、导出和未来 Hamming matcher 共用的二值契约。"""

    return {
        "model_architecture": model.architecture,
        "descriptor_kind": BINARY_DESCRIPTOR_KIND,
        "descriptor_metric": HAMMING_DISTANCE_METRIC,
        "descriptor_dim": model.hash_bits,
        "hash_bits": model.hash_bits,
        "hash_packed_bytes": (model.hash_bits + 7) // 8,
        "hash_head_architecture": model.hash_head.architecture,
        "hash_head_input_dim": model.hash_head.input_dim,
        "hash_head_hidden_dim": model.hash_head.hidden_dim,
        "hash_head_dropout": float(model.hash_head.dropout.p),
        "binary_encoding": BINARY_ENCODING,
        "binary_storage": BINARY_STORAGE,
        "binary_bitorder": model.bitorder,
        "backbone_architecture": model.backbone_architecture,
        "float_descriptor_dim": model.float_descriptor_dim,
        "backbone_trainable": model.backbone_trainable,
    }