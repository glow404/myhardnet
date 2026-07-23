"""HardNet 指纹 patch 描述子训练入口。

作用：
    1. 读取 YAML 配置与命令行覆盖参数。
    2. 构造训练/验证 DataLoader。
    3. 构造 HardNet 网络、top-k hardest-in-batch triplet loss 和 SGD 优化器。
    4. 按论文设置执行线性学习率衰减训练。
    5. 每个 epoch 输出训练日志、验证指标、checkpoint 和 metrics.csv。

典型用法：
    python -m hardnet_train.train --config hardnet_train/config.yaml --device cuda

输出：
    outputs/hardnet_train/
        best.pt              验证 FPR95 最好的模型
        last.pt              最后一个 epoch 的模型
        metrics.csv          每个 epoch 的训练/验证指标
        resolved_config.json 实际使用的配置快照
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
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

from hardnet_train.data import FingerImagePairBatchSampler, FingerprintPairDataset
from hardnet_train.loss import HardNetLoss, normalize_hard_negative_strategy
from hardnet_train.metrics import RunningMean, fpr_at_recall
from hardnet_train.model import HardNet, count_parameters


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


def set_seed(seed: int) -> None:
    """固定 Python / numpy / PyTorch 随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(config: dict[str, Any], split: str, batch_size: int, steps: int, seed: int) -> DataLoader:
    """构造某个 split 的 DataLoader。

    split 可取 train/val/test。这里训练和验证都使用同一个 batch sampler，
    目的是让验证指标和训练时 hardest-in-batch 的负样本定义一致。
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    csv_path = resolve_path(config, data_cfg[f"{split}_csv"])
    max_rows = data_cfg.get(f"max_{split}_rows")
    dataset = FingerprintPairDataset(
        csv_path=csv_path,
        max_rows=max_rows,
        max_rows_per_finger=data_cfg.get(f"max_{split}_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    available_fingers = len({record.finger_id for record in dataset.records})
    # 验证集可能只有 3 根手指；如果配置的 fingers_per_batch 更大，自动降到可用手指数。
    fingers_per_batch = min(int(train_cfg.get("fingers_per_batch", 8)), available_fingers)
    hard_negative_strategy = normalize_hard_negative_strategy(train_cfg.get("hard_negative_strategy", "same_finger_allowed"))
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
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )


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
    model: HardNet,
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

        # 两个分支共享同一个 HardNet 权重，等价于 siamese/two-stream CNN。
        optimizer.zero_grad(set_to_none=True)
        anchor_desc = model(anchor)
        positive_desc = model(positive)
        # point_group 会屏蔽同一物理点，避免伪负样本参与 hardest negative。
        loss, stats = criterion(anchor_desc, positive_desc, point_group=point_group, finger_group=finger_group)
        loss.backward()
        optimizer.step()

        # 日志中的 pos/neg 是 batch 内平均距离，用于观察正负分布是否逐渐拉开。
        batch_size = anchor.size(0)
        valid = stats["valid_triplets"]
        loss_meter.update(float(loss.item()), batch_size)
        pos_meter.update(float(stats["pos_dist"].mean().item()), batch_size)
        if torch.any(valid):
            neg_meter.update(float(stats["neg_dist"][valid].mean().item()), int(valid.sum().item()))

        if log_interval > 0 and step % log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"epoch={epoch} step={step}/{len(loader)} lr={lr:.6g} "
                f"loss={loss_meter.value:.4f} pos={pos_meter.value:.4f} "
                f"neg={neg_meter.value:.4f} samples/s={loss_meter.count / elapsed:.1f}",
                flush=True,
            )
        global_step += 1

    return {"loss": loss_meter.value, "pos_dist": pos_meter.value, "neg_dist": neg_meter.value, "lr": lr}, global_step


@torch.no_grad()
def evaluate(model: HardNet, criterion: HardNetLoss, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """在验证集上评估 loss、正负距离和 FPR95。"""
    model.eval()
    loss_meter = RunningMean()
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []

    for batch in loader:
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        point_group = batch["point_group"].to(device, non_blocking=True)
        finger_group = batch["finger_group"].to(device, non_blocking=True)
        loss, stats = criterion(model(anchor), model(positive), point_group=point_group, finger_group=finger_group)
        batch_size = anchor.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        valid = stats["valid_triplets"].cpu()
        positives.append(stats["pos_dist"].cpu())
        negatives.append(stats["neg_dist"].cpu()[valid])

    pos = torch.cat(positives) if positives else torch.empty(0)
    neg = torch.cat(negatives) if negatives else torch.empty(0)
    return {
        "loss": loss_meter.value,
        "pos_dist": float(pos.mean().item()) if pos.numel() else 0.0,
        "neg_dist": float(neg.mean().item()) if neg.numel() else 0.0,
        "fpr95": fpr_at_recall(pos, neg, recall=0.95),
    }


def save_checkpoint(
    path: Path,
    model: HardNet,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    """保存模型、优化器、配置和当前验证指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": {key: value for key, value in config.items() if key != "_config_path"},
    }
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
    model: HardNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
) -> tuple[int, int]:
    """加载 checkpoint，返回下一轮 epoch 和 global_step。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    finished_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", finished_epoch * steps_per_epoch))
    return finished_epoch + 1, global_step


def best_fpr95_from_metrics(metrics_path: Path) -> tuple[float, float, int]:
    """从已有 metrics.csv 中恢复 best_fpr95、早停 best 和 no_improve 计数。"""
    if not metrics_path.exists():
        return float("inf"), float("inf"), 0
    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        return float("inf"), float("inf"), 0
    fprs = [float(row["val_fpr95"]) for row in rows if row.get("val_fpr95")]
    best_fpr95 = min(fprs) if fprs else float("inf")
    last = rows[-1]
    early_best = float(last.get("early_stop_best_fpr95") or best_fpr95)
    no_improve = int(float(last.get("no_improve_epochs") or 0))
    return best_fpr95, early_best, no_improve


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的指标追加写入 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    """根据 metrics.csv 输出训练曲线图。

    图中包含 loss、正负样本距离、val_fpr95 和学习率曲线。
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

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) not in (None, "")]

    epochs = values("epoch")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)

    axes[0, 0].plot(epochs, values("train_loss"), label="train_loss")
    axes[0, 0].plot(epochs, values("val_loss"), label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("train_pos_dist"), label="train_pos")
    axes[0, 1].plot(epochs, values("train_neg_dist"), label="train_neg")
    axes[0, 1].plot(epochs, values("val_pos_dist"), label="val_pos")
    axes[0, 1].plot(epochs, values("val_neg_dist"), label="val_neg")
    axes[0, 1].set_title("Descriptor Distances")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, values("val_fpr95"), label="val_fpr95", color="tab:red")
    axes[1, 0].set_title("FPR@95% Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    if rows and "lr" in rows[0]:
        axes[1, 1].plot(epochs, values("lr"), label="lr", color="tab:green")
    axes[1, 1].set_title("Learning Rate")
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
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fingers-per-batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--hard-negative-strategy", choices=["same_finger_allowed", "different_finger"], default=None)
    parser.add_argument(
        "--hard-negative-top-k",
        type=int,
        default=None,
        help="Number of hardest legal negatives used per positive pair (default from YAML, normally 3).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--scheduler", choices=["linear", "warmup_cosine"], default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from, or 'auto' for output_dir/last.pt.")
    parser.add_argument("--resume-auto", action="store_true", help="Resume from output_dir/last.pt if it exists.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate training_curves.png after training.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = config.get("model", {})
    optim_cfg = config.get("optimizer", {})
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_name = args.device or train_cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    epochs = int(args.epochs or train_cfg.get("epochs", 10))
    steps_per_epoch = int(args.steps_per_epoch or train_cfg.get("steps_per_epoch", 1000))
    val_steps = int(args.val_steps or train_cfg.get("val_steps", 100))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 128))
    if args.fingers_per_batch is not None:
        train_cfg["fingers_per_batch"] = int(args.fingers_per_batch)
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)
    if args.dropout is not None:
        model_cfg["dropout"] = float(args.dropout)
    if args.margin is not None:
        train_cfg["margin"] = float(args.margin)
    if args.hard_negative_strategy is not None:
        train_cfg["hard_negative_strategy"] = args.hard_negative_strategy
    if args.hard_negative_top_k is not None:
        train_cfg["hard_negative_top_k"] = int(args.hard_negative_top_k)
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

    train_cfg["hard_negative_strategy"] = normalize_hard_negative_strategy(train_cfg.get("hard_negative_strategy", "same_finger_allowed"))
    train_cfg["hard_negative_top_k"] = int(train_cfg.get("hard_negative_top_k", 3))
    if train_cfg["hard_negative_top_k"] < 1:
        raise ValueError(
            f"training.hard_negative_top_k must be >= 1, got {train_cfg['hard_negative_top_k']}."
        )

    output_dir = resolve_path(config, args.output_dir or config.get("output_dir", "../outputs/hardnet_train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存实际配置，避免训练结束后忘记当时使用的 batch_size/lr/dropout 等参数。
    (output_dir / "resolved_config.json").write_text(
        json.dumps({key: value for key, value in config.items() if key != "_config_path"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 注意：构造 loader 会扫描 CSV，但不会把 patch 图像读入内存。
    train_loader = make_loader(config, "train", batch_size=batch_size, steps=steps_per_epoch, seed=seed)
    val_loader = make_loader(config, "val", batch_size=batch_size, steps=val_steps, seed=seed + 10_000)

    model = HardNet(
        dropout=float(model_cfg.get("dropout", 0.1)),
        final_bn_affine=bool(model_cfg.get("final_bn_affine", False)),
    ).to(device)
    criterion = HardNetLoss(
        margin=float(train_cfg.get("margin", 1.0)),
        hard_negative_strategy=str(train_cfg.get("hard_negative_strategy", "same_finger_allowed")),
        hard_negative_top_k=int(train_cfg.get("hard_negative_top_k", 3)),
    )
    # 按论文默认：SGD + momentum=0.9 + weight_decay=1e-4。
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 0.1)),
        momentum=float(optim_cfg.get("momentum", 0.9)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0001)),
    )

    scheduler = str(train_cfg.get("scheduler", "warmup_cosine"))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 2.0))
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    eta_min = float(train_cfg.get("eta_min", 1e-4))
    total_steps = max(1, epochs * steps_per_epoch)
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    early_stop_min_relative = float(train_cfg.get("early_stop_min_relative_improvement", 0.01))
    print(
        f"device={device} params={count_parameters(model)} batch={batch_size} "
        f"fingers_per_batch={train_cfg.get('fingers_per_batch', 8)} "
        f"dropout={model_cfg.get('dropout', 0.1)} lr={optim_cfg.get('lr', 0.1)} "
        f"margin={train_cfg.get('margin', 1.0)} steps_per_epoch={steps_per_epoch} epochs={epochs} "
        f"hard_negative_strategy={train_cfg.get('hard_negative_strategy', 'same_finger_allowed')} "
        f"hard_negative_top_k={train_cfg.get('hard_negative_top_k', 3)} "
        f"scheduler={scheduler} warmup_epochs={warmup_epochs} eta_min={eta_min} "
        f"early_stop_patience={early_stop_patience} early_stop_min_delta={early_stop_min_relative}",
        flush=True,
    )

    best_fpr95, early_stop_best_fpr95, no_improve_epochs = best_fpr95_from_metrics(output_dir / "metrics.csv")
    start_epoch = 1
    global_step = 0
    resume_arg = "auto" if args.resume_auto else args.resume
    if resume_arg:
        resume_path = output_dir / "last.pt" if resume_arg == "auto" else Path(resume_arg).expanduser()
        if not resume_path.is_absolute():
            resume_path = (Path.cwd() / resume_path).resolve()
        if resume_path.exists():
            start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, device, steps_per_epoch)
            print(f"resumed from {resume_path} | start_epoch={start_epoch} global_step={global_step}", flush=True)
        elif args.resume_auto or resume_arg == "auto":
            print(f"resume auto skipped: checkpoint not found at {resume_path}", flush=True)
        else:
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

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
        )
        val_metrics = evaluate(model, criterion, val_loader, device)
        save_checkpoint(output_dir / "last.pt", model, optimizer, config, epoch, global_step, val_metrics)
        # FPR95 越低越好，因此用它选择 best checkpoint。
        if val_metrics["fpr95"] <= best_fpr95:
            best_fpr95 = val_metrics["fpr95"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, config, epoch, global_step, val_metrics)

        # 早停规则：
        #   只有当 val_fpr95 相比早停历史 best 至少相对下降 min_delta 时，
        #   才认为“显著改善”并重置计数；否则累计 no_improve_epochs。
        #   例如 best=0.9868、min_delta=0.01，则新值必须 < 0.976932。
        if early_stop_best_fpr95 == float("inf"):
            early_stop_best_fpr95 = val_metrics["fpr95"]
            no_improve_epochs = 0
            early_stop_message = "early_stop=init"
        else:
            improvement_threshold = early_stop_best_fpr95 * (1.0 - early_stop_min_relative)
            if val_metrics["fpr95"] < improvement_threshold:
                early_stop_best_fpr95 = val_metrics["fpr95"]
                no_improve_epochs = 0
                early_stop_message = "early_stop=improved"
            else:
                no_improve_epochs += 1
                early_stop_message = f"early_stop=no_improve({no_improve_epochs}/{early_stop_patience})"

        metric_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pos_dist": train_metrics["pos_dist"],
            "train_neg_dist": train_metrics["neg_dist"],
            "val_loss": val_metrics["loss"],
            "val_pos_dist": val_metrics["pos_dist"],
            "val_neg_dist": val_metrics["neg_dist"],
            "val_fpr95": val_metrics["fpr95"],
            "lr": train_metrics["lr"],
            "early_stop_best_fpr95": early_stop_best_fpr95,
            "no_improve_epochs": no_improve_epochs,
        }
        append_metrics(output_dir / "metrics.csv", metric_row)

        print(
            f"epoch={epoch} done train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_fpr95={val_metrics['fpr95']:.4f} "
            f"best_for_stop={early_stop_best_fpr95:.4f} {early_stop_message}",
            flush=True,
        )
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"early stopping: val_fpr95 has not improved by "
                f"{early_stop_min_relative * 100:.2f}% for {early_stop_patience} epochs.",
                flush=True,
            )
            break

    if not args.no_plot:
        plot_training_curves(output_dir / "metrics.csv", output_dir / "training_curves.png")
        print(f"saved training curves to {output_dir / 'training_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
