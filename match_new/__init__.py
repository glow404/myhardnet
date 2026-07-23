"""match_new 包初始化。

作用：
    这个目录实现 HardNet L2/top-k/RANSAC 版本的手机解锁式指纹验证实验。
    与旧 `match/` 目录不同，这里不再把 HardNet 描述子二值化为 Hamming，
    而是在连续 L2 描述子空间中生成候选，再做几何验证和 identity 级融合。
"""
