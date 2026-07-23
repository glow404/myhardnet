"""HardNet 正样本 patch 对 PyTorch Dataset。

作用：
- 读取 extract_patches 阶段生成的 train_pairs.csv / val_pairs.csv / test_pairs.csv。
- 加载其中的 `patch_a_path` 和 `patch_p_path`，返回 HardNet 训练需要的正样本对。
- 默认执行 per-patch mean/std normalization，对齐 HardNet 论文的输入预处理。

说明：
- 这个脚本只负责读取已经落盘的 patch，不负责在线 SIFT、几何扩展或裁剪。
- 当前没有实现 batch 采样约束；后续训练阶段可基于 metadata 中的
  `finger_id` / `image_pair_id` / `correspondence_id` 实现专门 sampler。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import read_csv_rows

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None
    Dataset = object  # type: ignore[assignment]


class HardNetPositivePairDataset(Dataset):
    """读取正样本 patch pair 的最小 Dataset 实现。"""

    def __init__(self, csv_path: str | Path, normalize: bool = True) -> None:
        """初始化数据集。

        参数：
        - csv_path：由 pair_build 生成的 pairs CSV。
        - normalize：true 时执行每个 patch 独立的 mean/std 标准化。
        """

        if torch is None:
            raise ImportError("当前环境未安装 torch，无法使用 HardNetPositivePairDataset")
        self.csv_path = Path(csv_path)
        self.rows = read_csv_rows(self.csv_path)
        self.normalize = normalize
        if not self.rows:
            raise ValueError(f"CSV 没有可用样本: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_patch(path: str) -> np.ndarray:
        """读取单通道 patch 文件。"""

        patch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(f"patch 读取失败: {path}")
        return patch.astype(np.float32)

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """按 HardNet 输入约定归一化 patch。"""

        if not self.normalize:
            return patch / 255.0
        mean = float(np.mean(patch))
        std = max(float(np.std(patch)), 1e-6)
        return (patch - mean) / std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """返回 `(anchor_patch, positive_patch, metadata)`。"""

        row = self.rows[index]
        patch_a = self._normalize_patch(self._read_patch(row["patch_a_path"]))[None, :, :]
        patch_p = self._normalize_patch(self._read_patch(row["patch_p_path"]))[None, :, :]
        return (
            torch.from_numpy(np.ascontiguousarray(patch_a.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(patch_p.astype(np.float32))),
            {**row, "index": index},
        )


def build_dataloader(csv_path: str | Path, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, normalize: bool = True) -> DataLoader:
    """构造一个普通 DataLoader，方便快速冒烟测试训练读取链路。"""

    if DataLoader is None:
        raise ImportError("当前环境未安装 torch，无法构造 DataLoader")
    dataset = HardNetPositivePairDataset(csv_path=csv_path, normalize=normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
