"""指纹 patch 描述子训练包。

本目录承接 `pair_build` 生成的正样本 patch 对，复现 HardNet 论文中的
L2Net/HardNet 网络结构与 hardest-in-batch triplet loss，并加入适合小指纹
图像的 batch 采样策略，尽量减少伪负样本对。
"""
