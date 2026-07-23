# finger

`finger` 目录保存已有的指纹匹配/C++ 相关代码。

## 目录定位

当前 HardNet 训练代码不会直接修改这里的文件。后续如果要把训练好的
HardNet 描述子接入现有匹配流程，可以在这里增加：

- patch 提取与网络推理接口；
- HardNet 描述子替换 SIFT 描述子的匹配逻辑；
- RANSAC 内点数对比；
- SIFT 与 HardNet 的可视化评估。

## 与 hardnet_train 的关系

`hardnet_train` 负责产出模型 checkpoint，例如：

```text
outputs/hardnet_train/best.pt
```

`finger` 后续可以加载该 checkpoint 或导出的推理模型，将其用于实际指纹匹配流程。

