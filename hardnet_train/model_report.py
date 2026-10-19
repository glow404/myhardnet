"""生成随训练产物保存的逐层模型结构与计算量报告。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class LayerProfile:
    """一次实际前向调用对应的层级统计。"""

    index: int
    name: str
    layer_type: str
    input_shape: str
    output_shape: str
    purpose: str
    parameters: int
    trainable_parameters: int
    macs_per_patch: int | None


@dataclass(frozen=True)
class ModelProfile:
    """单个 32×32 patch 的完整模型统计。"""

    architecture: str
    model_class: str
    descriptor_dim: int | None
    input_shape: str
    output_shape: str
    parameters: int
    trainable_parameters: int
    macs_per_patch: int
    flops_per_patch: int
    layers: tuple[LayerProfile, ...]


_STRUCTURAL_LAYER_TYPES = {
    "BinaryDescriptorModel",
    "ContrastAwareStem",
    "CoordinateAttention",
    "HardNetStrongV2",
    "InvertedResidual",
    "ContextMixer",
    "ResidualHashHead",
    "ScalarWeightedFusion",
    "StableResBlock",
}


def _shape_text(value: Any) -> str:
    if torch.is_tensor(value):
        return "[" + ",".join(str(int(size)) for size in value.shape) + "]"
    if isinstance(value, (tuple, list)):
        values = [_shape_text(item) for item in value]
        return "(" + ", ".join(values) + ")"
    if isinstance(value, dict):
        values = [f"{key}: {_shape_text(item)}" for key, item in value.items()]
        return "{" + ", ".join(values) + "}"
    return type(value).__name__


def _direct_parameter_counts(module: nn.Module) -> tuple[int, int]:
    parameters = list(module.parameters(recurse=False))
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return int(total), int(trainable)


def _layer_macs(module: nn.Module, output: Any) -> int | None:
    if not torch.is_tensor(output) or output.ndim == 0:
        return None
    batch_size = max(int(output.shape[0]), 1)
    output_per_patch = output.numel() // batch_size
    if isinstance(module, nn.Conv2d):
        kernel_macs = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        return int(output_per_patch * kernel_macs)
    if isinstance(module, nn.Linear):
        return int(output_per_patch * module.in_features)
    return None


def _tuple_text(value: int | tuple[int, ...]) -> str:
    if isinstance(value, tuple):
        return "×".join(str(item) for item in value)
    return str(value)


def _layer_purpose(module: nn.Module) -> str:
    """返回面向网络阅读者的层作用说明。"""

    if isinstance(module, nn.Conv2d):
        kernel = _tuple_text(module.kernel_size)
        stride = _tuple_text(module.stride)
        dilation = _tuple_text(module.dilation)
        if module.groups == module.in_channels and module.groups > 1:
            role = "逐通道空间特征提取"
        elif module.kernel_size == (1, 1):
            role = "通道投影与融合"
        elif module.kernel_size[0] >= 8:
            role = "可学习的全局空间投影"
        else:
            role = "局部空间特征提取"
        return (
            f"{role}；Conv {module.in_channels}→{module.out_channels}，"
            f"kernel={kernel}，stride={stride}，dilation={dilation}，"
            f"groups={module.groups}"
        )
    if isinstance(module, nn.Linear):
        return f"全连接特征投影 {module.in_features}→{module.out_features}"
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return f"批归一化，稳定特征分布；affine={module.affine}"
    if isinstance(module, nn.GroupNorm):
        return f"组归一化；groups={module.num_groups}，channels={module.num_channels}"
    if isinstance(module, nn.LayerNorm):
        return f"对末端维度 {tuple(module.normalized_shape)} 做层归一化"
    if isinstance(module, nn.ReLU6):
        return "ReLU6 非线性激活，限制移动网络激活范围"
    if isinstance(module, nn.ReLU):
        return "ReLU 非线性激活"
    if isinstance(module, nn.SiLU):
        return "SiLU 平滑非线性激活"
    if isinstance(module, nn.Hardswish):
        return "Hardswish 轻量平滑激活"
    if isinstance(module, nn.GELU):
        return "GELU 平滑非线性激活"
    if isinstance(module, nn.Sigmoid):
        return "Sigmoid 将注意力权重限制到 0～1"
    if isinstance(module, nn.Dropout2d):
        return f"训练时按通道随机失活，p={module.p}"
    if isinstance(module, nn.Dropout):
        return f"训练时随机失活，p={module.p}"
    if isinstance(module, nn.AvgPool2d):
        return (
            f"平均池化抗混叠下采样；kernel={_tuple_text(module.kernel_size)}，"
            f"stride={_tuple_text(module.stride)}"
        )
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return f"自适应全局/区域平均池化到 {module.output_size}"
    if isinstance(module, nn.Identity):
        return "恒等映射，保持残差捷径不变"

    layer_type = type(module).__name__
    if layer_type == "InvertedResidual":
        residual = "带残差相加" if bool(getattr(module, "use_residual", False)) else "不使用残差"
        return f"MobileNetV2 倒残差块：扩展、深度卷积、线性投影；{residual}；计算量由子层统计"
    if layer_type == "CoordinateAttention":
        return "分别编码高度与宽度上下文并重标定特征；计算量由子层统计"
    if layer_type == "ContrastAwareStem":
        return "融合 3×3 局部纹理和局部对比度两条输入分支；计算量由子层统计"
    if layer_type == "DropPath":
        return f"训练时按样本随机丢弃残差路径，p={module.probability}"
    if layer_type == "StableResBlock":
        return "预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计"
    if layer_type == "ContextMixer":
        return "以扩张 depthwise 3×3 和 pointwise 1×1 单路径混合上下文；计算量由子层统计"
    if layer_type == "ScalarWeightedFusion":
        return "单位化空间/全局描述子，并以一个可学习标量加权融合；计算量由子层统计"
    if layer_type == "GeM":
        return "可学习广义均值全局池化，兼顾峰值响应和整体稳定性"
    if layer_type == "ResidualHashHead":
        return "浮点描述子投影及残差哈希变换；计算量由子层统计"
    if layer_type == "BinaryDescriptorModel":
        return "浮点主干与二值哈希头组合；计算量由子层统计"
    return layer_type


def _is_reported_layer(module: nn.Module) -> bool:
    return not any(module.children()) or type(module).__name__ in _STRUCTURAL_LAYER_TYPES


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 1, 32, 32),
) -> ModelProfile:
    """实际执行一次推理，记录各层形状、参数与 Conv/Linear MACs。"""

    if len(input_shape) != 4 or any(int(size) <= 0 for size in input_shape):
        raise ValueError(f"input_shape must contain four positive sizes, got {input_shape}.")

    rows: list[LayerProfile] = []
    pending_rows: dict[int, list[int]] = {}
    handles: list[Any] = []
    module_states = {module: module.training for module in model.modules()}

    def make_pre_hook(name: str, module: nn.Module):
        def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            parameter_count, trainable_count = _direct_parameter_counts(module)
            row_index = len(rows)
            rows.append(
                LayerProfile(
                    index=row_index + 1,
                    name=name,
                    layer_type=type(module).__name__,
                    input_shape=_shape_text(inputs[0] if len(inputs) == 1 else inputs),
                    output_shape="",
                    purpose=_layer_purpose(module),
                    parameters=parameter_count,
                    trainable_parameters=trainable_count,
                    macs_per_patch=None,
                )
            )
            pending_rows.setdefault(id(module), []).append(row_index)

        return pre_hook

    def make_forward_hook(module: nn.Module):
        def forward_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            row_index = pending_rows[id(module)].pop()
            rows[row_index].output_shape = _shape_text(output)
            rows[row_index].macs_per_patch = _layer_macs(module, output)

        return forward_hook

    for name, module in model.named_modules():
        if not name or not _is_reported_layer(module):
            continue
        handles.append(module.register_forward_pre_hook(make_pre_hook(name, module)))
        handles.append(module.register_forward_hook(make_forward_hook(module)))

    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = (
        reference.dtype
        if reference is not None and reference.dtype.is_floating_point
        else torch.float32
    )
    sample = torch.zeros(tuple(int(size) for size in input_shape), device=device, dtype=dtype)
    try:
        model.eval()
        with torch.inference_mode():
            output = model(sample)
    finally:
        for handle in handles:
            handle.remove()
        # 精确恢复各子模块原来的 train/eval 状态，兼容冻结的二值 backbone。
        for module, training in module_states.items():
            module.training = training

    parameter_ids: set[int] = set()
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        if id(parameter) in parameter_ids:
            continue
        parameter_ids.add(id(parameter))
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    total_macs = sum(row.macs_per_patch or 0 for row in rows)
    architecture = str(getattr(model, "architecture", type(model).__name__))
    descriptor_dim = getattr(model, "descriptor_dim", None)
    return ModelProfile(
        architecture=architecture,
        model_class=type(model).__name__,
        descriptor_dim=int(descriptor_dim) if descriptor_dim is not None else None,
        input_shape=_shape_text(sample),
        output_shape=_shape_text(output),
        parameters=int(total_parameters),
        trainable_parameters=int(trainable_parameters),
        macs_per_patch=int(total_macs),
        flops_per_patch=int(total_macs * 2),
        layers=tuple(rows),
    )


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _write_markdown(path: Path, model: nn.Module, profile: ModelProfile) -> None:
    lines = [
        f"# {profile.model_class} 模型结构报告",
        "",
        "## 汇总",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| 架构标识 | `{profile.architecture}` |",
        f"| 模型类 | `{profile.model_class}` |",
        f"| 描述子维度 | {profile.descriptor_dim if profile.descriptor_dim is not None else '—'} |",
        f"| 输入形状 | `{profile.input_shape}` |",
        f"| 输出形状 | `{profile.output_shape}` |",
        f"| 总参数量 | {_format_count(profile.parameters)} |",
        f"| 可训练参数量 | {_format_count(profile.trainable_parameters)} |",
        f"| Conv/Linear MACs / patch | {_format_count(profile.macs_per_patch)} |",
        f"| 约算 FLOPs / patch | {_format_count(profile.flops_per_patch)} |",
        "",
        "> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；"
        "FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、"
        "注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，"
        "不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs"
        "在各自明细行统计，避免重复。",
        "",
        "## PyTorch 模块树",
        "",
        "```text",
        str(model),
        "```",
        "",
        "## 逐层明细（按实际前向调用顺序）",
        "",
        "| # | 层路径 | 类型 | 输入 | 输出 | 作用 | 参数量 | 可训练参数 | MACs/patch | FLOPs/patch≈ |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in profile.layers:
        macs = "—" if row.macs_per_patch is None else _format_count(row.macs_per_patch)
        flops = "—" if row.macs_per_patch is None else _format_count(row.macs_per_patch * 2)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.index),
                    f"`{_markdown_cell(row.name)}`",
                    row.layer_type,
                    f"`{_markdown_cell(row.input_shape)}`",
                    f"`{_markdown_cell(row.output_shape)}`",
                    _markdown_cell(row.purpose),
                    _format_count(row.parameters),
                    _format_count(row.trainable_parameters),
                    macs,
                    flops,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, profile: ModelProfile) -> None:
    fieldnames = [
        "序号",
        "层路径",
        "类型",
        "输入形状",
        "输出形状",
        "作用",
        "参数量",
        "可训练参数量",
        "MACs_per_patch",
        "FLOPs_per_patch_approx",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile.layers:
            writer.writerow(
                {
                    "序号": row.index,
                    "层路径": row.name,
                    "类型": row.layer_type,
                    "输入形状": row.input_shape,
                    "输出形状": row.output_shape,
                    "作用": row.purpose,
                    "参数量": row.parameters,
                    "可训练参数量": row.trainable_parameters,
                    "MACs_per_patch": "" if row.macs_per_patch is None else row.macs_per_patch,
                    "FLOPs_per_patch_approx": (
                        "" if row.macs_per_patch is None else row.macs_per_patch * 2
                    ),
                }
            )


def write_model_structure_report(
    model: nn.Module,
    output_dir: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 1, 32, 32),
) -> tuple[ModelProfile, Path, Path]:
    """在 checkpoint 所在目录写入 Markdown 与 CSV 逐层结构表。"""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile = profile_model(model, input_shape=input_shape)
    markdown_path = root / "model_structure.md"
    csv_path = root / "model_structure.csv"
    _write_markdown(markdown_path, model, profile)
    _write_csv(csv_path, profile)
    return profile, markdown_path, csv_path
