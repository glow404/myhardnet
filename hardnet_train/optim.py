"""描述子训练优化器工厂与 weight-decay 参数分组。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


_OPTIMIZER_ALIASES = {
    "sgd": "sgd",
    "sgd_nesterov": "sgd",
    "nesterov": "sgd",
    "adamw": "adamw",
}

_NORMALIZATION_MODULES = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
)


def normalize_optimizer_name(value: str | None) -> str:
    key = str(value or "sgd").strip().lower()
    if key not in _OPTIMIZER_ALIASES:
        supported = ", ".join(sorted(_OPTIMIZER_ALIASES))
        raise ValueError(f"Unsupported optimizer: {value!r}. Supported: {supported}")
    return _OPTIMIZER_ALIASES[key]


def split_weight_decay_parameters(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Conv/Linear 权重应用 decay；归一化参数和所有 bias 不应用 decay。"""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter_name == "bias" or isinstance(module, _NORMALIZATION_MODULES):
                no_decay.append(parameter)
            elif isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
                decay.append(parameter)
            elif parameter.ndim <= 1:
                no_decay.append(parameter)
            else:
                decay.append(parameter)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(seen) != len({id(parameter) for parameter in trainable}):
        raise RuntimeError("Optimizer parameter grouping did not cover every trainable parameter exactly once.")
    return [
        {"name": "decay", "params": decay, "weight_decay": float(weight_decay)},
        {"name": "no_decay", "params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """根据配置构建 SGD+Nesterov 或 AdamW，共用明确的 decay/no-decay 分组。"""
    name = normalize_optimizer_name(config.get("name"))
    lr = float(config.get("lr", 0.1))
    parameter_groups = split_weight_decay_parameters(
        model,
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    if name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        nesterov = bool(config.get("nesterov", True))
        if nesterov and momentum <= 0.0:
            raise ValueError("SGD Nesterov requires optimizer.momentum > 0.")
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=0.0,
        )
    else:
        raw_betas = config.get("betas", (0.9, 0.999))
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two numbers.")
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=lr,
            betas=(float(raw_betas[0]), float(raw_betas[1])),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=0.0,
        )

    setattr(optimizer, "hardnet_optimizer_name", name)
    return optimizer


def optimizer_name(optimizer: torch.optim.Optimizer) -> str:
    explicit = getattr(optimizer, "hardnet_optimizer_name", None)
    if explicit is not None:
        return normalize_optimizer_name(str(explicit))
    if isinstance(optimizer, torch.optim.SGD):
        return "sgd"
    if isinstance(optimizer, torch.optim.AdamW):
        return "adamw"
    return optimizer.__class__.__name__.lower()