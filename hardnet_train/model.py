"""HardNet 网络结构定义。

作用：
    1. 按论文中的 L2Net/HardNet 结构构建 32x32 灰度 patch 描述子网络。
    2. 输入形状为 `[B, 1, 32, 32]`，输出形状为 `[B, 128]`。
    3. 输出描述子会做 L2 归一化，因此可以直接用欧氏距离比较 patch 相似度。

设计依据：
    HardNet 论文采用无池化卷积网络，通过 stride=2 卷积降低空间尺寸，
    最后用 8x8 卷积把 8x8 特征图变成 128 维描述子。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HardNet(nn.Module):
    """HardNet/L2Net 描述子网络。

    参数：
        dropout:
            论文原始设置为 0.1；post-NIPS 版本曾使用 0.3。
            小指纹数据量和噪声情况不确定时，建议先用 0.1 做基线。
        descriptor_dim:
            HardNet/SIFT 对齐为 128 维。这里保留参数但限制为 128，
            避免不小心改坏论文结构。
        final_bn_affine:
            最后一层 BN 默认不学习 affine 参数，更接近常见 HardNet 实现。
    """

    def __init__(self, dropout: float = 0.1, descriptor_dim: int = 128, final_bn_affine: bool = False) -> None:
        super().__init__()
        if descriptor_dim != 128:
            raise ValueError("The paper architecture expects descriptor_dim=128.")

        # 尺寸变化：
        #   32x32 -> 32x32 -> 32x32 -> 16x16 -> 16x16 -> 8x8 -> 8x8 -> 1x1
        # 通道变化：
        #   1 -> 32 -> 32 -> 64 -> 64 -> 128 -> 128 -> 128
        self.features = nn.Sequential(
            self._conv_block(1, 32, stride=1),
            self._conv_block(32, 32, stride=1),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 64, stride=1),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 128, stride=1),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(128, descriptor_dim, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(descriptor_dim, affine=final_bn_affine),
        )
        self.reset_parameters()

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """构造论文中重复出现的 `3x3 Conv + BN + ReLU` 块。"""
        return nn.Sequential(
            # bias=False 是因为后面接 BatchNorm，卷积 bias 会被 BN 平移项吸收。
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def reset_parameters(self) -> None:
        """按论文描述初始化参数。

        论文写明卷积权重使用 orthogonal 初始化，gain=0.6，bias=0.01。
        当前卷积层 bias=False，因此 bias 分支主要是给未来改结构时兜底。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=0.6)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """前向计算 128 维单位长度描述子。"""
        descriptors = self.features(patches)
        # 最后一层卷积输出 `[B, 128, 1, 1]`，压平成 `[B, 128]`。
        descriptors = descriptors.view(descriptors.size(0), -1)
        # HardNet 距离公式假设描述子是单位向量；训练和评估都依赖这一点。
        return F.normalize(descriptors, p=2, dim=1)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，训练启动时用于打印模型规模。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
