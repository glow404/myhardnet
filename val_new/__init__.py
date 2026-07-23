"""新的 HardNet/RootSIFT 公平验证项目。

目标：
    1. 先找出 RootSIFT/SIFT 表现差、内点数少的图像对。
    2. 对这些困难图像对，在完全相同的匹配参数下比较 RootSIFT 和 HardNet。
    3. 输出困难图像对原图与匹配预览，方便人工分析失败原因。
"""

