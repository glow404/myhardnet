"""HardNet 指纹 patch 描述子训练入口。

作用：
    1. 读取 YAML 配置与命令行覆盖参数。
    2. 构造训练随机 sampler 与确定性的固定验证 batch 计划。
    3. 构造描述子网络、top-k hardest-in-batch triplet loss 和可配置优化器。
    4. 执行带预热的余弦学习率调度训练。
    5. 每个 epoch 输出固定 in-batch 验证指标、checkpoint 和 metrics.csv。

典型用法：
    python -m hardnet_train.train --config hardnet_train/config.yaml --device cuda

输出：
    outputs/hardnet_train/
        best.pt              固定协议 `matching_composite_v1` 综合分数最高的模型
        last.pt              最后一个 epoch 的模型
        metrics.csv          每个 epoch 的训练/固定验证指标
        resolved_config.json 实际使用的配置快照
        model_structure.md   可读的模型树、汇总和逐层结构表
        model_structure.csv  可供 Excel/脚本分析的逐层结构表
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 当前 Windows 环境中，PyTorch / OpenCV / numpy 可能同时带 Intel OpenMP runtime。
# 训练入口不导入 cv2，但为了避免某些依赖间接触发冲突，这里提前设置兜底环境变量。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hardnet_train.checkpoint_selection import (
    checkpoint_selection_metrics,
    checkpoint_selection_state_from_metrics,
    normalize_checkpoint_selection_config,
)
from hardnet_train.data import (
    FingerImagePairBatchSampler,
    FingerprintPairDataset,
    PatchAugmentationConfig,
)
from hardnet_train.loss import HardNetLoss, normalize_hard_negative_strategy
from hardnet_train.metrics import RunningMean
from hardnet_train.negative_sampling import (
    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
    normalize_min_coordinate_separation,
)
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
    count_parameters,
    estimate_macs_per_patch,
    model_architecture,
    model_descriptor_dim,
    normalize_model_architecture,
    resolve_descriptor_dim,
)
from hardnet_train.model_report import write_model_structure_report
from hardnet_train.optim import build_optimizer, normalize_optimizer_name, optimizer_name
from hardnet_train.validation import (
    build_fixed_validation_batch_plan,
    evaluate_fixed_validation_batches,
    make_fixed_validation_loader,
    normalize_validation_finger_count,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并记录配置文件绝对路径，方便解析相对路径。"""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    """把配置中的路径解析成绝对路径。

    相对路径一律相对配置文件所在目录，而不是相对当前 shell 所在目录。
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def default_output_dir(architecture: str, descriptor_dim: Any = None) -> str:
    """按架构和描述子维度返回互不冲突的默认训练输出目录。"""

    normalized = normalize_model_architecture(architecture)
    dimension = resolve_descriptor_dim(normalized, descriptor_dim)
    architecture_label = {
        "hardnet_strong_v2": "strong_v2",
        "mobile_hardnet": "mobile",
        "hardnet": "hardnet",
    }[normalized]
    return (
        f"../outputs/models/hardnet_train_{architecture_label}_{dimension}_"
        "fixed_candidate_pool_v3"
    )


def resolve_resume_checkpoint(
    output_dir: Path,
    resume_arg: str | None,
    allow_missing: bool,
) -> Path | None:
    """在构建数据流前解析续训 checkpoint，并区分强制恢复与可选恢复。"""
    if not resume_arg:
        return None
    checkpoint_path = (
        output_dir / "last.pt"
        if resume_arg == "auto"
        else Path(resume_arg).expanduser()
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if checkpoint_path.exists():
        return checkpoint_path
    if allow_missing:
        return None
    raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")


def negative_sampling_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """提取会改变训练标签、验证配对与 checkpoint 选模的协议。"""

    training = config.get("training", {})
    validation = config.get("validation", {})
    if not isinstance(training, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("training and validation config sections must be mappings.")
    return {
        "contract_version": 5,
        "training_hard_negative_strategy": normalize_hard_negative_strategy(
            training.get("hard_negative_strategy", "same_finger_allowed")
        ),
        "training_hard_negative_top_k": int(
            training.get("hard_negative_top_k", 3)
        ),
        "training_same_finger_min_coordinate_separation_px": (
            normalize_min_coordinate_separation(
                training.get(
                    "same_finger_min_coordinate_separation_px",
                    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
                )
            )
        ),
        "validation_protocol": str(
            validation.get("protocol", "fixed_candidate_pool_v3")
        )
        .strip()
        .lower(),
        "validation_seed": int(validation.get("seed", 10_042)),
        "validation_finger_count": normalize_validation_finger_count(
            validation.get("finger_count", "auto")
        ),
        "validation_batch_count": int(
            validation.get("batch_count", 256)
        ),
        "validation_batch_size": int(
            validation.get("batch_size", 24)
        ),
        "validation_candidate_rule": "fixed_full_legal_pool_for_roc_topk_for_loss",
        "checkpoint_selection": normalize_checkpoint_selection_config(
            config.get("checkpoint_selection")
        ),
    }


def descriptor_loss_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """提取会改变描述子优化目标的 loss 契约，防止错误续训。"""

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("training config section must be a mapping.")
    return {
        "contract_version": 1,
        "margin": float(training.get("margin", 1.0)),
        "hard_negative_top_k": int(training.get("hard_negative_top_k", 3)),
        "hard_negative_top1_weight": float(
            training.get("hard_negative_top1_weight", 0.6)
        ),
        "positive_tail_loss_weight": float(
            training.get("positive_tail_loss_weight", 0.10)
        ),
        "positive_tail_p95_weight": float(
            training.get("positive_tail_p95_weight", 0.7)
        ),
        "positive_tail_p99_weight": float(
            training.get("positive_tail_p99_weight", 0.3)
        ),
        "positive_tail_p95_target": float(
            training.get("positive_tail_p95_target", 0.75)
        ),
        "positive_tail_p99_target": float(
            training.get("positive_tail_p99_target", 0.90)
        ),
    }


def train_augmentation_contract(config: Mapping[str, Any]) -> dict[str, bool | float]:
    """返回会改变训练输入分布的在线增强契约。"""

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data config section must be a mapping.")
    augmentation = data_config.get("train_augmentation")
    if augmentation is None:
        # 训练入口默认启用轻量增强；FingerprintPairDataset 本身仍默认关闭，
        # 因此固定验证与其他只读调用不会被意外扰动。
        augmentation = {"enabled": True}
    if augmentation is not None and not isinstance(augmentation, Mapping):
        raise ValueError("data.train_augmentation must be a mapping.")
    return PatchAugmentationConfig.from_mapping(augmentation).as_dict()


_LEGACY_FIXED_PAIR_VALIDATION_KEYS = {
    "positive_count",
    "same_finger_min_coordinate_separation_px",
}


def normalize_fixed_in_batch_validation_config(
    validation_config: dict[str, Any],
    *,
    default_seed: int,
) -> None:
    """规范化固定候选池验证配置，并拒绝旧协议和旧配额字段。"""

    protocol = str(
        validation_config.get("protocol", "fixed_candidate_pool_v3")
    ).strip().lower()
    if protocol != "fixed_candidate_pool_v3":
        raise ValueError(
            f"Unsupported validation.protocol: {protocol!r}. Expected "
            "'fixed_candidate_pool_v3'; older protocols use grouped validation "
            "metrics or dynamic top-k for ROC and cannot share an output directory "
            "with the new overall metrics."
        )
    legacy_keys = sorted(
        key
        for key in _LEGACY_FIXED_PAIR_VALIDATION_KEYS
        if key in validation_config
    )
    if legacy_keys:
        raise ValueError(
            "Legacy fixed-pair validation fields are not supported by "
            f"fixed_candidate_pool_v3: {legacy_keys}. Configure finger_count, "
            "batch_count and batch_size instead."
        )
    if "image_pairs_per_finger" in validation_config:
        raise ValueError(
            "validation.image_pairs_per_finger has been replaced by "
            "validation.batch_count. The new plan cycles deterministically through "
            "each selected finger's available image_pair values."
        )

    validation_config["protocol"] = protocol
    validation_config["seed"] = int(
        validation_config.get("seed", default_seed)
    )
    validation_config["finger_count"] = normalize_validation_finger_count(
        validation_config.get("finger_count", "auto")
    )
    validation_config["batch_count"] = int(
        validation_config.get("batch_count", 256)
    )
    validation_config["batch_size"] = int(
        validation_config.get("batch_size", 24)
    )
    if validation_config["batch_count"] < 1:
        raise ValueError("validation.batch_count must be >= 1.")
    if validation_config["batch_size"] < 2:
        raise ValueError(
            "validation.batch_size must be >= 2 so in-batch negatives can exist."
        )
    if (
        isinstance(validation_config["finger_count"], int)
        and validation_config["batch_size"] < validation_config["finger_count"]
    ):
        raise ValueError(
            "validation.batch_size must be >= validation.finger_count. Set "
            "validation.finger_count=auto or adjust the two numeric values."
        )


def validate_resume_negative_sampling_contract(
    checkpoint: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> None:
    """禁止把旧训练/验证候选协议的优化器状态续接到新实验。"""

    saved_contract = checkpoint.get("negative_sampling_contract")
    if not isinstance(saved_contract, Mapping):
        raise ValueError(
            "Checkpoint predates the fixed in-batch validation contract; do not "
            "resume its optimizer state under the new protocol. Start a new "
            "experiment and use the checkpoint only as pretrained weights."
        )
    saved_contract = dict(saved_contract)
    current_contract = negative_sampling_contract(current_config)
    if saved_contract != current_contract:
        raise ValueError(
            "Checkpoint negative-sampling/validation protocol mismatch: "
            f"checkpoint={saved_contract}, configured={current_contract}. "
            "The fixed in-batch protocol changes validation batches and candidate "
            "selection; start a new output directory instead of resuming this "
            "checkpoint."
        )
    saved_loss_contract = checkpoint.get("descriptor_loss_contract")
    if not isinstance(saved_loss_contract, Mapping):
        raise ValueError(
            "Checkpoint predates the weighted-hard-negative/positive-tail loss "
            "contract; do not resume its optimizer state under the new loss. "
            "Start a new experiment and use the checkpoint only as pretrained weights."
        )
    current_loss_contract = descriptor_loss_contract(current_config)
    if dict(saved_loss_contract) != current_loss_contract:
        raise ValueError(
            "Checkpoint descriptor-loss contract mismatch: "
            f"checkpoint={dict(saved_loss_contract)}, configured={current_loss_contract}. "
            "Start a new output directory instead of resuming this checkpoint."
        )
    current_augmentation_contract = train_augmentation_contract(current_config)
    saved_augmentation_contract = checkpoint.get("train_augmentation_contract")
    if not isinstance(saved_augmentation_contract, Mapping):
        if bool(current_augmentation_contract["enabled"]):
            raise ValueError(
                "Checkpoint predates the online-augmentation contract, but "
                "data.train_augmentation.enabled=true. Start a new output directory "
                "instead of resuming with a changed input distribution."
            )
    elif dict(saved_augmentation_contract) != current_augmentation_contract:
        raise ValueError(
            "Checkpoint online-augmentation contract mismatch: "
            f"checkpoint={dict(saved_augmentation_contract)}, "
            f"configured={current_augmentation_contract}. Start a new output "
            "directory instead of resuming this checkpoint."
        )


def set_seed(seed: int) -> None:
    """固定 Python / numpy / PyTorch 随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(config: dict[str, Any], split: str, batch_size: int, steps: int, seed: int) -> DataLoader:
    """构造使用随机 hardest-in-batch sampler 的训练 DataLoader。

    验证使用同一采样语义，但 batch 索引在训练启动时一次性固化。
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    csv_path = resolve_path(config, data_cfg[f"{split}_csv"])
    max_rows = data_cfg.get(f"max_{split}_rows")
    augmentation = PatchAugmentationConfig.from_mapping(
        data_cfg.get("train_augmentation", {"enabled": True})
        if split == "train"
        else None
    )
    dataset = FingerprintPairDataset(
        csv_path=csv_path,
        max_rows=max_rows,
        max_rows_per_finger=data_cfg.get(f"max_{split}_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
        augmentation=augmentation,
    )
    if augmentation.enabled:
        print(
            "online augmentation enabled: "
            f"p={augmentation.probability} "
            f"rotation=±{augmentation.max_rotation_degrees}deg "
            f"translation=±{augmentation.max_translation_pixels}px "
            f"scale=±{augmentation.scale_jitter} "
            f"contrast=±{augmentation.contrast_jitter} "
            f"gamma=±{augmentation.gamma_jitter} "
            f"blur_p={augmentation.blur_probability} "
            f"noise_p={augmentation.noise_probability}",
            flush=True,
        )
    available_fingers = len({record.finger_id for record in dataset.records})
    requested_fingers_per_batch = int(train_cfg.get("fingers_per_batch", 8))
    if requested_fingers_per_batch < 1:
        raise ValueError("training.fingers_per_batch must be >= 1.")
    if available_fingers < 1:
        raise ValueError(f"No usable fingers found for split={split}.")
    fingers_per_batch = min(requested_fingers_per_batch, available_fingers)
    if fingers_per_batch > batch_size:
        raise ValueError(
            f"Effective fingers_per_batch={fingers_per_batch} exceeds "
            f"training.batch_size={batch_size} for split={split}. Increase "
            "batch_size or reduce training.fingers_per_batch."
        )
    if fingers_per_batch != requested_fingers_per_batch:
        print(
            "configuration adjustment: "
            f"training.fingers_per_batch={requested_fingers_per_batch} reduced "
            f"to {fingers_per_batch} because split={split} contains only "
            f"{available_fingers} usable fingers.",
            flush=True,
        )
    train_cfg["fingers_per_batch"] = fingers_per_batch
    hard_negative_strategy = normalize_hard_negative_strategy(
        train_cfg.get("hard_negative_strategy", "same_finger_allowed")
    )
    if hard_negative_strategy == "different_finger" and fingers_per_batch < 2:
        raise ValueError(
            "hard_negative_strategy='different_finger' requires at least 2 fingers per batch "
            f"for split={split}. Got fingers_per_batch={fingers_per_batch}, available_fingers={available_fingers}."
        )
    sampler = FingerImagePairBatchSampler(
        records=dataset.records,
        batch_size=batch_size,
        fingers_per_batch=fingers_per_batch,
        batches_per_epoch=steps,
        seed=seed,
        drop_incomplete=True,
    )
    num_workers = max(0, int(data_cfg.get("num_workers", 0)))
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)) and torch.cuda.is_available(),
    }
    # 只有多进程 DataLoader 支持持续 worker 和预取参数。持续 worker 可以避免
    # Windows 在每个 epoch 都重新创建进程，预取则让 CPU 读图与 GPU 计算重叠。
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_options["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))
    return DataLoader(
        **loader_options,
    )


def resolve_device(raw_device: str) -> torch.device:
    """解析训练设备，并在明确要求 CUDA 但环境不可用时给出可操作错误。"""

    requested = str(raw_device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "训练配置要求 CUDA，但当前 PyTorch 不支持 CUDA。请安装 CUDA 版 PyTorch，"
            "并确认 torch.cuda.is_available() 返回 True；如需主动使用 CPU，请设置 "
            "training.device=cpu。"
        )
    return torch.device(requested)


def resolve_amp(
    device: torch.device,
    raw_precision: str,
) -> tuple[bool, torch.dtype | None]:
    """解析混合精度配置，返回是否启用 autocast 及其数据类型。"""

    precision = str(raw_precision).strip().lower()
    if precision in {"", "off", "none", "fp32", "float32"}:
        return False, None
    if device.type != "cuda":
        return False, None
    if precision in {"fp16", "float16"}:
        return True, torch.float16
    if precision in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU/PyTorch 不支持 BF16，请改用 training.mixed_precision=fp16。")
        return True, torch.bfloat16
    raise ValueError(f"不支持的 training.mixed_precision: {raw_precision!r}")


def configure_cuda(device: torch.device, train_cfg: dict[str, Any]) -> None:
    """应用只影响 CUDA 执行性能的后端选项。"""

    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def linear_lr(base_lr: float, step: int, total_steps: int) -> float:
    """论文设置：学习率在总训练步数内线性衰减到 0。"""
    if total_steps <= 1:
        return base_lr
    return base_lr * max(0.0, 1.0 - step / float(total_steps - 1))


def scheduled_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    scheduler: str,
    warmup_steps: int = 0,
    eta_min: float = 1e-4,
) -> float:
    """计算当前 step 的学习率。

    支持：
        linear:
            论文原始策略，从 base_lr 线性衰减到 0。
        warmup_cosine:
            先从 0 线性 warmup 到 base_lr，再余弦退火到 eta_min。
            这是当前指纹实验的默认策略，避免训练后期直接 lr=0。
    """
    scheduler = scheduler.lower()
    if scheduler == "linear":
        return linear_lr(base_lr, step, total_steps)
    if scheduler != "warmup_cosine":
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    cosine_total = max(1, total_steps - warmup_steps)
    cosine_step = min(max(step - warmup_steps, 0), cosine_total)
    progress = cosine_step / float(cosine_total)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(eta_min) + (float(base_lr) - float(eta_min)) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """更新优化器中所有参数组的学习率。"""
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(
    model: torch.nn.Module,
    criterion: HardNetLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    global_step: int,
    base_lr: float,
    scheduler: str,
    warmup_steps: int,
    eta_min: float,
    log_interval: int,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> tuple[dict[str, float], int]:
    """训练一个 epoch。

    返回：
        metrics:
            当前 epoch 的 loss、正样本距离、负样本距离均值。
        global_step:
            全局 step 计数，用于跨 epoch 线性衰减学习率。
    """
    model.train()
    loss_meter = RunningMean()
    pos_meter = RunningMean()
    neg_meter = RunningMean()
    tail_meter = RunningMean()
    pos_p95_meter = RunningMean()
    pos_p99_meter = RunningMean()
    start = time.time()
    lr = base_lr

    for step, batch in enumerate(loader, start=1):
        # 当前默认使用 warmup + cosine decay；这里每个 step 更新一次学习率。
        lr = scheduled_lr(base_lr, global_step, total_steps, scheduler, warmup_steps, eta_min)
        set_optimizer_lr(optimizer, lr)

        # DataLoader 返回 CPU tensor；non_blocking=True 在 pin_memory + CUDA 时可加速拷贝。
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        point_group = batch["point_group"].to(device, non_blocking=True)
        finger_group = batch["finger_group"].to(device, non_blocking=True)
        anchor_xy = batch["anchor_xy"].to(device, non_blocking=True)
        positive_xy = batch["positive_xy"].to(device, non_blocking=True)
        anchor_coordinate_frame_group = batch[
            "anchor_coordinate_frame_group"
        ].to(device, non_blocking=True)
        positive_coordinate_frame_group = batch[
            "positive_coordinate_frame_group"
        ].to(device, non_blocking=True)
        if channels_last:
            anchor = anchor.contiguous(memory_format=torch.channels_last)
            positive = positive.contiguous(memory_format=torch.channels_last)

        # 两个分支共享同一个 HardNet 权重，等价于 siamese/two-stream CNN。
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            anchor_desc = model(anchor)
            positive_desc = model(positive)
        # point_group 会屏蔽同一物理点，避免伪负样本参与 hardest negative。
        # 描述子网络使用 AMP，距离矩阵和 hardest-negative loss 保持 FP32，
        # 避免极接近样本在半精度下出现排序抖动。
        loss, stats = criterion(
            anchor_desc.float(),
            positive_desc.float(),
            point_group=point_group,
            finger_group=finger_group,
            anchor_xy=anchor_xy,
            positive_xy=positive_xy,
            anchor_coordinate_frame_group=anchor_coordinate_frame_group,
            positive_coordinate_frame_group=positive_coordinate_frame_group,
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 日志中的 pos/neg 是 batch 内平均距离，用于观察正负分布是否逐渐拉开。
        batch_size = anchor.size(0)
        valid = stats["valid_triplets"]
        loss_meter.update(float(loss.item()), batch_size)
        pos_meter.update(float(stats["pos_dist"].mean().item()), batch_size)
        if torch.any(valid):
            valid_count = int(valid.sum().item())
            neg_meter.update(float(stats["neg_dist"][valid].mean().item()), valid_count)
            positive_p95 = stats["positive_p95"]
            positive_p99 = stats["positive_p99"]
            if bool(torch.isfinite(positive_p95).item()) and bool(torch.isfinite(positive_p99).item()):
                pos_p95_meter.update(float(positive_p95.item()), batch_size)
                pos_p99_meter.update(float(positive_p99.item()), batch_size)
            tail_meter.update(float(stats["positive_tail_loss"].item()), batch_size)

        if log_interval > 0 and step % log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"epoch={epoch} step={step}/{len(loader)} lr={lr:.4g} "
                f"loss={loss_meter.value:.4g} pos={pos_meter.value:.4g} "
                f"neg={neg_meter.value:.4g} pos_tail={tail_meter.value:.4g} "
                f"pos_p95={pos_p95_meter.value:.4g} pos_p99={pos_p99_meter.value:.4g} "
                f"samples/s={loss_meter.count / elapsed:.4g}",
                flush=True,
            )
        global_step += 1

    return {
        "loss": loss_meter.value,
        "pos_dist": pos_meter.value,
        "neg_dist": neg_meter.value,
        "positive_tail_loss": tail_meter.value,
        "positive_p95": pos_p95_meter.value,
        "positive_p99": pos_p99_meter.value,
        "lr": lr,
    }, global_step


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
    model_macs_per_patch: int | None = None,
) -> None:
    """保存模型、优化器、配置和当前验证指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint_format_version": 2,
        "training_task": "float_descriptor",
        "descriptor_kind": "float",
        "descriptor_metric": "l2",
        "negative_sampling_contract": negative_sampling_contract(resolved_config),
        "descriptor_loss_contract": descriptor_loss_contract(resolved_config),
        "train_augmentation_contract": train_augmentation_contract(resolved_config),
        "model_architecture": model_architecture(model),
        "descriptor_dim": model_descriptor_dim(model),
        "model_parameter_count": count_parameters(model),
        "model_macs_per_patch": (
            estimate_macs_per_patch(model)
            if model_macs_per_patch is None
            else int(model_macs_per_patch)
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": optimizer_name(optimizer),
        "metrics": metrics,
        "checkpoint_selection_strategy": (
            resolved_config.get("checkpoint_selection", {}).get("strategy")
            if isinstance(resolved_config.get("checkpoint_selection"), Mapping)
            else None
        ),
        "checkpoint_selection_score": metrics.get("checkpoint_selection_score"),
        "config": resolved_config,
        "resolved_config": resolved_config,
    }
    if scaler is not None and scaler.is_enabled():
        payload["amp_scaler"] = scaler.state_dict()
    torch.save(payload, path)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """把 optimizer state 中的 tensor 移动到当前训练设备。

    从 CPU checkpoint 恢复到 CUDA，或从 CUDA 恢复到 CPU 时都需要这一步。
    """
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
    config: Mapping[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int]:
    """加载 checkpoint，返回下一轮 epoch 和 global_step。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    validate_resume_negative_sampling_contract(checkpoint, config)
    saved_architecture = checkpoint_model_architecture(checkpoint)
    current_architecture = model_architecture(model)
    if saved_architecture != current_architecture:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"checkpoint={saved_architecture}, model={current_architecture}."
        )
    saved_descriptor_dim = checkpoint_descriptor_dim(checkpoint)
    current_descriptor_dim = model_descriptor_dim(model)
    if saved_descriptor_dim != current_descriptor_dim:
        raise ValueError(
            "Checkpoint descriptor dimension mismatch: "
            f"checkpoint={saved_descriptor_dim}, model={current_descriptor_dim}."
        )
    model.load_state_dict(checkpoint["model"])
    saved_optimizer_name = checkpoint.get("optimizer_name")
    if saved_optimizer_name is not None:
        normalized_saved_optimizer = normalize_optimizer_name(str(saved_optimizer_name))
        if normalized_saved_optimizer != optimizer_name(optimizer):
            raise ValueError(
                "Checkpoint optimizer mismatch: "
                f"checkpoint={normalized_saved_optimizer}, configured={optimizer_name(optimizer)}. "
                "Changing optimizer requires a new output directory and training from scratch."
            )
    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "Checkpoint optimizer state is incompatible with the configured decay/no-decay groups. "
                "Start a new experiment instead of restoring the old optimizer state."
            ) from exc
        move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and scaler.is_enabled() and "amp_scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler"])
    finished_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", finished_epoch * steps_per_epoch))
    return finished_epoch + 1, global_step


def _format_metric_value(value: Any) -> Any:
    """将浮点指标压缩为 4 位有效数字，整数状态保持原样。"""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".4g")
    return value


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的精简指标追加写入 CSV，并拒绝混写不同 schema。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row.keys())
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_fields = csv.DictReader(handle).fieldnames
        if existing_fields != fieldnames:
            raise ValueError(
                "metrics.csv schema does not match the focused training metrics. "
                "Use a new output directory instead of appending to an old experiment."
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {name: _format_metric_value(value) for name, value in row.items()}
        )


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    """根据 metrics.csv 绘制四项 checkpoint 选模信号。

    如果 matplotlib 不可用，则跳过绘图，不影响训练结果。
    """
    if not metrics_path.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plotting: matplotlib unavailable ({exc})", flush=True)
        return

    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    if not rows:
        return

    def values(name: str, *fallback_names: str) -> list[float]:
        names = (name, *fallback_names)
        result: list[float] = []
        for row in rows:
            value = next((row.get(candidate) for candidate in names if row.get(candidate) not in (None, "")), None)
            if value is not None:
                result.append(float(value))
        return result

    epochs = values("epoch")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)

    axes[0, 0].plot(epochs, values("train_loss"), label="train_loss")
    axes[0, 0].plot(epochs, values("val_loss"), label="val_loss")
    axes[0, 0].set_title("Train vs Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        values("val_pos_p95", "val_pos_mean"),
        label="val_pos_p95",
    )
    axes[0, 1].set_title("Positive Distance P95")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        epochs,
        values("val_fpr_at_tpr95"),
        label="fpr_at_tpr95",
    )
    axes[1, 0].set_title("FPR at TPR=95%")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    distance_gap = [
        negative - positive
        for negative, positive in zip(
            values("val_neg_mean"),
            values("val_pos_mean"),
        )
    ]
    axes[1, 1].plot(
        epochs,
        distance_gap,
        label="val_neg_mean - val_pos_mean",
    )
    axes[1, 1].set_title("Mean Distance Gap")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    YAML 提供默认值；命令行参数用于临时覆盖常用训练超参。
    """
    parser = argparse.ArgumentParser(description="Train HardNet fingerprint patch descriptor.")
    parser.add_argument("--config", default="hardnet_train/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--val-finger-count",
        type=normalize_validation_finger_count,
        default=None,
    )
    parser.add_argument("--val-batch-count", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fingers-per-batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=None,
        help="Maximum Strong V2 stochastic-depth rate; ignored by other architectures.",
    )
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--hard-negative-strategy", choices=["same_finger_allowed", "different_finger"], default=None)
    parser.add_argument(
        "--hard-negative-top-k",
        type=int,
        default=None,
        help="Number of hardest legal negatives used per positive pair (default from YAML, normally 3).",
    )
    parser.add_argument(
        "--hard-negative-top1-weight",
        type=float,
        default=None,
        help="Weight assigned to the nearest selected negative; remaining top-k weight is shared.",
    )
    parser.add_argument("--positive-tail-loss-weight", type=float, default=None)
    parser.add_argument("--positive-tail-p95-weight", type=float, default=None)
    parser.add_argument("--positive-tail-p99-weight", type=float, default=None)
    parser.add_argument("--positive-tail-p95-target", type=float, default=None)
    parser.add_argument("--positive-tail-p99-target", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--scheduler", choices=["linear", "warmup_cosine"], default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path to resume from, or 'auto' to require output_dir/last.pt.",
    )
    parser.add_argument("--resume-auto", action="store_true", help="Resume from output_dir/last.pt if it exists.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate training_curves.png after training.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    optim_cfg = config.setdefault("optimizer", {})
    validation_cfg = config.setdefault("validation", {})
    data_cfg = config.setdefault("data", {})
    data_cfg["train_augmentation"] = train_augmentation_contract(config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_name = args.device or train_cfg.get("device", "auto")
    device = resolve_device(str(device_name))
    configure_cuda(device, train_cfg)
    amp_enabled, amp_dtype = resolve_amp(
        device,
        str(train_cfg.get("mixed_precision", "fp16")),
    )
    channels_last = device.type == "cuda" and bool(train_cfg.get("channels_last", True))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        train_cfg["steps_per_epoch"] = int(args.steps_per_epoch)
    if args.val_finger_count is not None:
        validation_cfg["finger_count"] = args.val_finger_count
    if args.val_batch_count is not None:
        validation_cfg["batch_count"] = int(args.val_batch_count)
    if args.val_batch_size is not None:
        validation_cfg["batch_size"] = int(args.val_batch_size)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        train_cfg["device"] = str(args.device)
    if args.fingers_per_batch is not None:
        train_cfg["fingers_per_batch"] = int(args.fingers_per_batch)
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)
    if args.optimizer is not None:
        optim_cfg["name"] = args.optimizer
    if args.dropout is not None:
        model_cfg["dropout"] = float(args.dropout)
    if args.drop_path_rate is not None:
        model_cfg["drop_path_rate"] = float(args.drop_path_rate)
    if args.margin is not None:
        train_cfg["margin"] = float(args.margin)
    if args.hard_negative_strategy is not None:
        train_cfg["hard_negative_strategy"] = args.hard_negative_strategy
    if args.hard_negative_top_k is not None:
        train_cfg["hard_negative_top_k"] = int(args.hard_negative_top_k)
    if args.hard_negative_top1_weight is not None:
        train_cfg["hard_negative_top1_weight"] = float(args.hard_negative_top1_weight)
    if args.positive_tail_loss_weight is not None:
        train_cfg["positive_tail_loss_weight"] = float(args.positive_tail_loss_weight)
    if args.positive_tail_p95_weight is not None:
        train_cfg["positive_tail_p95_weight"] = float(args.positive_tail_p95_weight)
    if args.positive_tail_p99_weight is not None:
        train_cfg["positive_tail_p99_weight"] = float(args.positive_tail_p99_weight)
    if args.positive_tail_p95_target is not None:
        train_cfg["positive_tail_p95_target"] = float(args.positive_tail_p95_target)
    if args.positive_tail_p99_target is not None:
        train_cfg["positive_tail_p99_target"] = float(args.positive_tail_p99_target)
    if args.early_stop_patience is not None:
        train_cfg["early_stop_patience"] = int(args.early_stop_patience)
    if args.early_stop_min_delta is not None:
        train_cfg["early_stop_min_relative_improvement"] = float(args.early_stop_min_delta)
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.warmup_epochs is not None:
        train_cfg["warmup_epochs"] = float(args.warmup_epochs)
    if args.eta_min is not None:
        train_cfg["eta_min"] = float(args.eta_min)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    epochs = int(train_cfg.get("epochs", 10))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1000))
    batch_size = int(train_cfg.get("batch_size", 128))
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("training.epochs and training.steps_per_epoch must be > 0.")
    if batch_size < 2:
        raise ValueError(
            "training.batch_size must be >= 2 so in-batch negatives can exist."
        )
    train_cfg["epochs"] = epochs
    train_cfg["steps_per_epoch"] = steps_per_epoch
    train_cfg["batch_size"] = batch_size
    optim_cfg["name"] = normalize_optimizer_name(optim_cfg.get("name"))
    optim_cfg["lr"] = float(optim_cfg.get("lr", 0.1))
    optim_cfg["weight_decay"] = float(optim_cfg.get("weight_decay", 1e-4))
    normalize_fixed_in_batch_validation_config(
        validation_cfg,
        default_seed=seed + 10_000,
    )
    config["checkpoint_selection"] = normalize_checkpoint_selection_config(
        config.get("checkpoint_selection")
    )
    train_cfg["same_finger_min_coordinate_separation_px"] = (
        normalize_min_coordinate_separation(
            train_cfg.get(
                "same_finger_min_coordinate_separation_px",
                DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
            )
        )
    )

    model_cfg["architecture"] = normalize_model_architecture(
        model_cfg.get("architecture")
    )
    model_cfg["descriptor_dim"] = resolve_descriptor_dim(
        model_cfg["architecture"],
        model_cfg.get("descriptor_dim"),
    )
    if model_cfg["architecture"] == "hardnet_strong_v2":
        model_cfg["drop_path_rate"] = float(model_cfg.get("drop_path_rate", 0.08))
    train_cfg["hard_negative_strategy"] = normalize_hard_negative_strategy(train_cfg.get("hard_negative_strategy", "same_finger_allowed"))
    train_cfg["hard_negative_top_k"] = int(train_cfg.get("hard_negative_top_k", 3))
    configured_validation_fingers = validation_cfg["finger_count"]
    if (
        train_cfg["hard_negative_strategy"] == "different_finger"
        and isinstance(configured_validation_fingers, int)
        and configured_validation_fingers < 2
    ):
        raise ValueError(
            "hard_negative_strategy='different_finger' requires "
            "validation.finger_count >= 2 or 'auto' with at least two usable "
            "validation fingers."
        )
    if train_cfg["hard_negative_top_k"] < 1:
        raise ValueError(
            f"training.hard_negative_top_k must be >= 1, got {train_cfg['hard_negative_top_k']}."
        )

    configured_output_dir = config.get("output_dir")
    if configured_output_dir is None or str(configured_output_dir).strip().lower() in {
        "",
        "auto",
    }:
        config["output_dir"] = default_output_dir(
            model_cfg["architecture"],
            model_cfg["descriptor_dim"],
        )
    output_dir = resolve_path(config, config["output_dir"])
    resume_arg = "auto" if args.resume_auto else args.resume
    resume_path = resolve_resume_checkpoint(
        output_dir=output_dir,
        resume_arg=resume_arg,
        allow_missing=args.resume_auto,
    )
    if args.resume_auto and resume_path is None:
        print(f"resume auto skipped: checkpoint not found at {output_dir / 'last.pt'}", flush=True)
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and resume_path is None:
        raise FileExistsError(
            f"Output directory already contains metrics: {metrics_path}. "
            "Use a new output directory for a fresh fixed-protocol experiment, or explicitly resume it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练 batch 随 epoch 改变；验证 batch 在启动时按固定 seed 构建一次并复用。
    train_loader = make_loader(
        config,
        "train",
        batch_size=batch_size,
        steps=steps_per_epoch,
        seed=seed,
    )
    data_cfg = config["data"]
    validation_dataset = FingerprintPairDataset(
        csv_path=resolve_path(config, data_cfg["val_csv"]),
        max_rows=data_cfg.get("max_val_rows"),
        max_rows_per_finger=data_cfg.get("max_val_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    validation_plan = build_fixed_validation_batch_plan(
        records=validation_dataset.records,
        finger_count=validation_cfg["finger_count"],
        batch_count=int(validation_cfg["batch_count"]),
        batch_size=int(validation_cfg["batch_size"]),
        seed=int(validation_cfg["seed"]),
    )
    validation_cfg["finger_count"] = validation_plan.finger_count
    if (
        train_cfg["hard_negative_strategy"] == "different_finger"
        and validation_plan.finger_count < 2
    ):
        raise ValueError(
            "hard_negative_strategy='different_finger' requires at least two "
            "usable validation fingers."
        )
    validation_loader = make_fixed_validation_loader(
        dataset=validation_dataset,
        plan=validation_plan,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )
    model = build_descriptor_model(model_cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = HardNetLoss(
        margin=float(train_cfg.get("margin", 1.0)),
        hard_negative_strategy=str(train_cfg.get("hard_negative_strategy", "same_finger_allowed")),
        hard_negative_top_k=int(train_cfg.get("hard_negative_top_k", 3)),
        hard_negative_top1_weight=float(
            train_cfg.get("hard_negative_top1_weight", 0.6)
        ),
        positive_tail_loss_weight=float(
            train_cfg.get("positive_tail_loss_weight", 0.10)
        ),
        positive_tail_p95_weight=float(
            train_cfg.get("positive_tail_p95_weight", 0.7)
        ),
        positive_tail_p99_weight=float(
            train_cfg.get("positive_tail_p99_weight", 0.3)
        ),
        positive_tail_p95_target=float(
            train_cfg.get("positive_tail_p95_target", 0.75)
        ),
        positive_tail_p99_target=float(
            train_cfg.get("positive_tail_p99_target", 0.90)
        ),
        same_finger_min_coordinate_separation_px=float(
            train_cfg["same_finger_min_coordinate_separation_px"]
        ),
    )
    optimizer = build_optimizer(model, optim_cfg)

    scheduler = str(train_cfg.get("scheduler", "warmup_cosine"))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 2.0))
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    eta_min = float(train_cfg.get("eta_min", 1e-4))
    total_steps = max(1, epochs * steps_per_epoch)
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    early_stop_min_relative = float(train_cfg.get("early_stop_min_relative_improvement", 0.01))
    optimizer_detail = (
        f"momentum={optim_cfg.get('momentum', 0.9)} nesterov={optim_cfg.get('nesterov', True)}"
        if optimizer_name(optimizer) == "sgd"
        else f"betas={optim_cfg.get('betas', [0.9, 0.999])} eps={optim_cfg.get('eps', 1e-8)}"
    )
    model_profile, model_report_md, model_report_csv = write_model_structure_report(
        model,
        output_dir,
    )
    model_macs_per_patch = model_profile.macs_per_patch
    print(
        f"saved model structure report to {model_report_md} and {model_report_csv}",
        flush=True,
    )
    print(
        f"device={device} architecture={model_architecture(model)} "
        f"descriptor_dim={model_descriptor_dim(model)} "
        f"params={count_parameters(model)} macs_per_patch={model_macs_per_patch} "
        f"batch={batch_size} "
        f"fingers_per_batch={train_cfg.get('fingers_per_batch', 8)} "
        f"dropout={model_cfg.get('dropout', 0.1)} "
        f"drop_path_rate={model_cfg.get('drop_path_rate', 0.0)} "
        f"optimizer={optimizer_name(optimizer)} "
        f"lr={optim_cfg.get('lr', 0.1)} {optimizer_detail} "
        f"margin={train_cfg.get('margin', 1.0)} steps_per_epoch={steps_per_epoch} epochs={epochs} "
        f"hard_negative_strategy={train_cfg.get('hard_negative_strategy', 'same_finger_allowed')} "
        f"hard_negative_top_k={train_cfg.get('hard_negative_top_k', 3)} "
        f"hard_negative_top1_weight={train_cfg.get('hard_negative_top1_weight', 0.6)} "
        f"positive_tail_loss_weight={train_cfg.get('positive_tail_loss_weight', 0.10)} "
        f"positive_tail_targets="
        f"{train_cfg.get('positive_tail_p95_target', 0.75)}/"
        f"{train_cfg.get('positive_tail_p99_target', 0.90)} "
        f"same_finger_min_coordinate_separation_px="
        f"{train_cfg['same_finger_min_coordinate_separation_px']} "
        f"scheduler={scheduler} warmup_epochs={warmup_epochs} eta_min={eta_min} "
        f"early_stop_patience={early_stop_patience} early_stop_min_delta={early_stop_min_relative} "
        f"mixed_precision={train_cfg.get('mixed_precision', 'fp16') if amp_enabled else 'fp32'} "
        f"channels_last={channels_last} "
        f"validation_fingers={validation_plan.finger_count} "
        f"validation_batches={validation_plan.batch_count} "
        f"validation_batch_size={validation_plan.batch_size} "
        f"validation_unique_image_pairs_per_finger="
        f"{min(validation_plan.unique_image_pair_counts)}-"
        f"{max(validation_plan.unique_image_pair_counts)} "
        f"validation_fingers_with_image_pair_reuse="
        f"{validation_plan.repeated_image_pair_finger_count} "
        f"validation_positives={validation_plan.positive_count}",
        flush=True,
    )

    best_selection_score, early_best_selection_score, no_improve_epochs = (
        checkpoint_selection_state_from_metrics(metrics_path)
        if resume_path is not None
        else (float("-inf"), float("-inf"), 0)
    )
    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            resume_path,
            model,
            optimizer,
            device,
            steps_per_epoch,
            config=config,
            scaler=scaler,
        )
        print(f"resumed from {resume_path} | start_epoch={start_epoch} global_step={global_step}", flush=True)

    # 只有初始化/恢复检查全部通过后才写实际配置，避免失败命令污染既有目录。
    (output_dir / "resolved_config.json").write_text(
        json.dumps({key: value for key, value in config.items() if key != "_config_path"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if start_epoch > epochs:
        print(f"nothing to train: start_epoch={start_epoch} > epochs={epochs}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_steps=total_steps,
            global_step=global_step,
            base_lr=float(optim_cfg.get("lr", 0.1)),
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            eta_min=eta_min,
            log_interval=int(train_cfg.get("log_interval", 50)),
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        val_metrics = evaluate_fixed_validation_batches(
            model=model,
            loader=validation_loader,
            plan=validation_plan,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        selection_metrics = checkpoint_selection_metrics(val_metrics, config)
        val_metrics.update(selection_metrics)
        current_selection_score = val_metrics["checkpoint_selection_score"]
        current_fpr_at_tpr95 = val_metrics["fpr_at_tpr95"]
        is_best = current_selection_score >= best_selection_score
        if is_best:
            best_selection_score = current_selection_score
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config,
            epoch,
            global_step,
            val_metrics,
            scaler=scaler,
            model_macs_per_patch=model_macs_per_patch,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                global_step,
                val_metrics,
                scaler=scaler,
                model_macs_per_patch=model_macs_per_patch,
            )

        # 综合分数越高越好；min_delta 表示相对分数提升比例。
        if early_best_selection_score == float("-inf"):
            early_best_selection_score = current_selection_score
            no_improve_epochs = 0
            early_stop_message = "early_stop=init"
        else:
            improvement_threshold = early_best_selection_score * (1.0 + early_stop_min_relative)
            if current_selection_score > improvement_threshold:
                early_best_selection_score = current_selection_score
                no_improve_epochs = 0
                early_stop_message = "early_stop=improved"
            else:
                no_improve_epochs += 1
                early_stop_message = f"early_stop=no_improve({no_improve_epochs}/{early_stop_patience})"

        metric_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_positive_tail_loss": train_metrics["positive_tail_loss"],
            "train_positive_p95": train_metrics["positive_p95"],
            "train_positive_p99": train_metrics["positive_p99"],
            "val_loss": val_metrics["loss"],
            "val_pos_mean": val_metrics["pos_mean"],
            "val_pos_p95": val_metrics["pos_p95"],
            "val_neg_mean": val_metrics["neg_mean"],
            "val_neg_p01": val_metrics["neg_p01"],
            "val_fpr_at_tpr95": current_fpr_at_tpr95,
            "val_tpr_at_fpr_1e_4": val_metrics["tpr_at_fpr_1e_4"],
            "val_checkpoint_selection_score": current_selection_score,
            "val_valid_anchor_count": int(val_metrics["valid_anchor_count"]),
            "val_skipped_anchor_count": int(val_metrics["skipped_anchor_count"]),
            "lr": train_metrics["lr"],
            "early_stop_best_selection_score": early_best_selection_score,
            "no_improve_epochs": no_improve_epochs,
        }
        append_metrics(output_dir / "metrics.csv", metric_row)

        print(
            f"epoch={epoch} done "
            f"train_loss={train_metrics['loss']:.4g} "
            f"train_pos_tail={train_metrics['positive_tail_loss']:.4g} "
            f"train_pos_p95={train_metrics['positive_p95']:.4g} "
            f"train_pos_p99={train_metrics['positive_p99']:.4g} "
            f"val_loss={val_metrics['loss']:.4g} "
            f"val_pos_mean={val_metrics['pos_mean']:.4g} "
            f"val_neg_mean={val_metrics['neg_mean']:.4g} "
            f"val_fpr95={current_fpr_at_tpr95:.4g} "
            f"val_pos_p95={val_metrics['pos_p95']:.4g} "
            f"valid_anchors={int(val_metrics['valid_anchor_count'])} "
            f"skipped_anchors={int(val_metrics['skipped_anchor_count'])} "
            f"best_for_stop={early_best_selection_score:.4g} "
            f"{early_stop_message}",
            flush=True,
        )
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"early stopping: matching checkpoint score has not improved by "
                f"{early_stop_min_relative * 100:.4g}% for {early_stop_patience} epochs.",
                flush=True,
            )
            break

    if not args.no_plot:
        plot_training_curves(output_dir / "metrics.csv", output_dir / "training_curves.png")
        print(f"saved training curves to {output_dir / 'training_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
