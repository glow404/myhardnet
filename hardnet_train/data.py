"""HardNet 指纹正样本数据集与 batch 采样器。

作用：
    1. 读取 `pair_build` 生成的 `train_pairs.csv` / `val_pairs.csv`。
    2. 使用 PIL 加载 32x32 灰度 patch，避免训练进程同时导入 OpenCV 和
       PyTorch 时触发 Windows 上的 Intel OpenMP 冲突。
    3. 按 HardNet 论文要求返回正样本对 `(anchor, positive)`。
    4. 针对小指纹图像实现特殊 batch 采样：
       一个 batch 包含多个手指；每个手指在该 batch 里只选一个
       `image_pair_id`，也就是只来自两张图，降低伪负样本概率。
    5. 使用 union-find 把跨图连通的关键点合并为同一个 `point_group`，
       训练 loss 会屏蔽同组样本，避免同一物理点被当成负样本。
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class PairRecord:
    """CSV 中一条正样本 pair 的轻量索引记录。

    注意这里不提前读图，只保存路径和采样所需元数据。这样可以避免启动训练
    时把百万级 patch 全部载入内存。
    """

    patch_a_path: str
    patch_p_path: str
    finger_id: str
    finger_group: int
    image_pair_id: str
    point_key: str
    point_group: int
    pair_id: str
    source: str
    stability_score: float


class UnionFind:
    """并查集，用于把“被正样本关系连接起来的关键点”合并为同一物理点组。"""

    def __init__(self) -> None:
        # parent[x] 存储 x 的父节点；根节点满足 parent[x] == x。
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """查找 item 所在集合的根节点，并做路径压缩。"""
        if item not in self.parent:
            self.parent[item] = item
            return item
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """把 left 和 right 所在集合合并。"""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _keypoint_node(row: dict[str, str], side: str) -> str:
    """把 CSV 中某一侧关键点转成全局节点 id。

    side="a" 表示 anchor 图上的关键点 `(image_a_id, kp_a_idx)`；
    side="p" 表示 positive 图上的关键点 `(image_b_id, kp_b_idx)`。
    同一个 finger_id 下的同一张图、同一个 kp_idx 会得到同一个节点 id。
    """
    if side == "a":
        image_id = row.get("image_a_id", "")
        kp_idx = row.get("kp_a_idx", "")
    elif side == "p":
        image_id = row.get("image_b_id", "")
        kp_idx = row.get("kp_b_idx", "")
    else:
        raise ValueError(f"Unsupported side: {side}")
    return "|".join([row.get("finger_id", ""), image_id, kp_idx])


class FingerprintPairDataset(Dataset):
    """读取 `pair_build` 输出的正样本 patch 对。"""

    def __init__(
        self,
        csv_path: str | Path,
        max_rows: int | None = None,
        max_rows_per_finger: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.normalize = bool(normalize)

        # rows 暂存被采纳的 CSV 行。先扫描并建立 union-find，再生成 PairRecord；
        # 因为 point_group 需要等所有正样本边都合并完才能确定。
        rows: list[dict[str, str]] = []
        union_find = UnionFind()
        per_finger_counts: dict[str, int] = defaultdict(int)

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # max_rows 用于 smoke test 或快速小实验；正式训练保持 None。
                if max_rows is not None and len(rows) >= int(max_rows):
                    break
                finger_id = row["finger_id"]
                # CSV 往往按手指聚集，max_rows_per_finger 可以做均衡小样本抽查。
                if max_rows_per_finger is not None and per_finger_counts[finger_id] >= int(max_rows_per_finger):
                    continue

                # 一条正样本边说明 A 图关键点和 P 图关键点应属于同一个物理点。
                union_find.union(_keypoint_node(row, "a"), _keypoint_node(row, "p"))
                rows.append(row)
                per_finger_counts[finger_id] += 1

        if not rows:
            raise ValueError(f"No usable rows found in {self.csv_path}")

        root_to_group: dict[str, int] = {}
        finger_to_group: dict[str, int] = {}
        self.records = []
        for row_index, row in enumerate(rows):
            # union-find 的 root 是字符串，不适合直接搬到 GPU；这里映射成连续 int。
            root = union_find.find(_keypoint_node(row, "a"))
            point_group = root_to_group.setdefault(root, len(root_to_group))
            finger_group = finger_to_group.setdefault(row["finger_id"], len(finger_to_group))
            self.records.append(
                PairRecord(
                    patch_a_path=row["patch_a_path"],
                    patch_p_path=row["patch_p_path"],
                    finger_id=row["finger_id"],
                    finger_group=finger_group,
                    image_pair_id=row["image_pair_id"],
                    point_key=root,
                    point_group=point_group,
                    pair_id=row.get("pair_id", str(row_index)),
                    source=row.get("source", ""),
                    stability_score=float(row.get("stability_score", "0") or 0.0),
                )
            )

    def __len__(self) -> int:
        """返回可训练正样本对数量。"""
        return len(self.records)

    @staticmethod
    def _read_patch(path: str) -> np.ndarray:
        """用 PIL 读取灰度 patch。

        不使用 cv2.imread，是为了绕开当前 Windows 环境里 torch/cv2 同进程导入
        可能出现的 OpenMP runtime 重复初始化问题。
        """
        image = Image.open(path).convert("L")
        return np.asarray(image, dtype=np.float32)

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """按论文方式做单 patch 标准化。

        HardNet 论文中输入 patch 会减去自身均值、除以自身标准差。
        这比全局均值方差更适合局部 patch 描述子，能弱化曝光/整体灰度差异。
        """
        if not self.normalize:
            return patch / 255.0
        mean = float(patch.mean())
        std = max(float(patch.std()), 1e-6)
        return (patch - mean) / std

    def __getitem__(self, index: int) -> dict[str, object]:
        """返回 DataLoader 需要的一条训练样本。"""
        record = self.records[index]
        # 增加 channel 维度，形成 `[1, 32, 32]`。
        anchor = self._normalize_patch(self._read_patch(record.patch_a_path))[None, :, :]
        positive = self._normalize_patch(self._read_patch(record.patch_p_path))[None, :, :]
        return {
            "anchor": torch.from_numpy(np.ascontiguousarray(anchor, dtype=np.float32)),
            "positive": torch.from_numpy(np.ascontiguousarray(positive, dtype=np.float32)),
            "point_group": torch.tensor(record.point_group, dtype=torch.long),
            "finger_group": torch.tensor(record.finger_group, dtype=torch.long),
            "finger_id": record.finger_id,
            "image_pair_id": record.image_pair_id,
            "pair_id": record.pair_id,
            "source": record.source,
        }


class FingerImagePairBatchSampler(Sampler[list[int]]):
    """按“多手指、每手指单两图组合”构造 batch。

    采样策略：
        1. 先随机选 `fingers_per_batch` 个手指。
        2. 对每个手指，只随机选一个 `image_pair_id`。
        3. 从这个两图组合中抽若干个正样本 patch 对。
        4. 同一个 batch 内尽量不重复同一个 `point_group`。

    这样做是为了配合 HardNet 的 hardest-in-batch 负样本挖掘，尽量让负样本
    来自不同手指，而不是同一手指跨多张图的同一真实位置。
    """

    def __init__(
        self,
        records: list[PairRecord],
        batch_size: int,
        fingers_per_batch: int,
        batches_per_epoch: int,
        seed: int = 42,
        drop_incomplete: bool = True,
    ) -> None:
        if fingers_per_batch < 1:
            raise ValueError("fingers_per_batch must be >= 1")
        if batch_size < fingers_per_batch:
            raise ValueError("batch_size must be >= fingers_per_batch")
        self.records = records
        self.batch_size = int(batch_size)
        self.fingers_per_batch = int(fingers_per_batch)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.drop_incomplete = bool(drop_incomplete)
        self._iteration = 0

        # 建立二级索引：
        #   finger_id -> image_pair_id -> [dataset index]
        # 采样时可以快速限制“每个手指只用一个两图组合”。
        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            groups[record.finger_id][record.image_pair_id].append(index)
        self.groups = {finger: dict(image_pairs) for finger, image_pairs in groups.items()}
        self.finger_ids = [finger for finger, image_pairs in self.groups.items() if image_pairs]
        if len(self.finger_ids) < self.fingers_per_batch:
            raise ValueError(
                f"Need at least {self.fingers_per_batch} fingers, found {len(self.finger_ids)} in dataset."
            )

    def __len__(self) -> int:
        """一个 epoch 产生多少个 batch。"""
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        """逐 batch 产生 dataset index 列表。"""
        # 同一个 sampler 会被 DataLoader 在每个 epoch 重新迭代。
        # seed + iteration 能保证可复现，同时不同 epoch 采样不同 batch。
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        for _ in range(self.batches_per_epoch):
            selected_fingers = rng.sample(self.finger_ids, self.fingers_per_batch)
            per_finger_counts = self._counts_for_batch()
            rng.shuffle(per_finger_counts)
            batch: list[int] = []
            for finger_id, pair_count in zip(selected_fingers, per_finger_counts):
                image_pair_id = rng.choice(list(self.groups[finger_id].keys()))
                candidates = self.groups[finger_id][image_pair_id]
                batch.extend(self._sample_unique_points(candidates, pair_count, rng))

            # 默认 drop_incomplete=True。若某些图组可用点太少，会丢掉不完整 batch；
            # smoke/调试场景也可以改成补采样。
            if len(batch) < self.batch_size and not self.drop_incomplete:
                batch.extend(rng.choices(batch, k=self.batch_size - len(batch)))
            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]
            if len(batch) == self.batch_size or not self.drop_incomplete:
                rng.shuffle(batch)
                yield batch

    def _counts_for_batch(self) -> list[int]:
        """计算每个手指在当前 batch 中贡献多少正样本对。"""
        base = self.batch_size // self.fingers_per_batch
        remainder = self.batch_size % self.fingers_per_batch
        return [base + (1 if index < remainder else 0) for index in range(self.fingers_per_batch)]

    def _sample_unique_points(self, candidates: list[int], count: int, rng: random.Random) -> list[int]:
        """从一个两图组合中抽取尽量不重复物理点的样本。"""
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        selected: list[int] = []
        seen: set[int] = set()
        for index in shuffled:
            point_group = self.records[index].point_group
            if point_group in seen:
                continue
            selected.append(index)
            seen.add(point_group)
            if len(selected) >= count:
                break
        # 如果某个 image_pair 内独立物理点不足，为了保持 batch_size，只从已选样本补齐。
        # 这种补齐样本会在 loss 中被 point_group mask 保护，不会互相当负样本。
        if len(selected) < count and selected:
            selected.extend(rng.choices(selected, k=count - len(selected)))
        return selected
