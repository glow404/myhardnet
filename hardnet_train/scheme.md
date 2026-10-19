网络构建训练方案：

1.网络架构，根据hardnet论文，L2网络架构

top-k 难负样本挖掘（默认 k=3）



2.batch 模式,采样方式

2.1

```
按手指分层的 P×K 采样方式。

含义是：

- 每个 batch 先选若干不同 `finger_id`；
- 每个 `finger_id` 在当前 batch 中只贡献一对图像；
- 从这对图像里取 `K` 条正样本对；
- `K` 由 `pairs_per_image_pair` 指定，`P` 由 `batch_size // K` 自动推导。
```

2.2 分组分簇，将同一个物理点合并为并查集



3.数据增强

亮度、对比度、高斯噪声、模糊。

随机180°旋转



4.学习率调度策略

```
论文是学习率在 10 个 epoch 内线性衰减到 0。
可以换成余弦热重启：
或者warmup + cosine decay + 非零最小学习率
```





5.参数更新算法：

```
带动量的SGD（Stochastic Gradient Descent，随机梯度下降）
亮度、对比度、高斯噪声、模糊。

随机180°旋转
```





6.训练停止条件

论文为训练10轮停止。

  val_fpr_at_tpr95 验证集上的 FPR@95% 召回率（关键匹配指标）连续3轮没有下降1%则停止。



7.损失函数

对每条正样本，从双向合法候选中取距离最小的 k 个负样本：

单个负样本损失 = max(0, margin + 正样本距离 - 负样本距离)

先对该正样本的 k 个负样本损失求均值，再对 batch 内有效正样本求均值。



# 训练流程

下面按当前 [`config.yaml`](C:\Users\ZYH\Desktop\harnet32\myhardnet\hardnet_train\config.yaml) 的实际参数说明。

---

# 2. 训练集 batch 的构建

训练采样器是 [`FingerImagePairBatchSampler`](C:\Users\ZYH\Desktop\harnet32\myhardnet\hardnet_train\data.py:300)。

当前参数：

```yaml
batch_size: 256
fingers_per_batch: 32
steps_per_epoch: 1000
```

这里的 `batch_size=256` 表示：

> 一个 batch 有 256 个正样本对，即 256 张 anchor patch 加 256 张 positive patch，总共读取 512 张 patch。

## 具体例子

每个 batch 首先随机选择 32 根不同的手指：

```text
F01, F02, F03, ... F32
```

因为：

\[
256 \div 32 = 8
\]

所以每根手指贡献 8 个正样本对。

对于手指 `F01`：

1. 随机选择该手指的一组原图组合，例如：

```text
image_pair_id = image_12 + image_37
```

2. 只从这一对原图产生的对应点中选择 8 个正样本：

```text
A1 ↔ P1
A2 ↔ P2
A3 ↔ P3
...
A8 ↔ P8
```

3. 尽量保证这 8 对属于不同的 `point_group`，也就是不同物理位置。

所有手指都执行相同操作：

```text
32 根手指
× 每根手指 1 个 image_pair
× 每个 image_pair 8 个正样本对
= 256 个正样本对
```

最后将这 256 对打乱，形成训练 batch：

```text
anchor tensor:   [256, 1, 32, 32]
positive tensor: [256, 1, 32, 32]
```

### 如果某个图像对不足 8 个独立物理点

采样器会从已经选中的正样本中重复抽样补齐。

例如只有 6 个独立点：

```text
A1-P1 ... A6-P6
```

可能补成：

```text
A1-P1 ... A6-P6, A2-P2, A5-P5
```

重复记录拥有相同的 `point_group`，loss 会**屏蔽它们之间的负样本关系**，不会把同一个物理点错误地推远。

如果一个 image pair 连一个可用样本都没有，最终 batch 不完整时会丢弃该 batch。

---

# 3. 一个训练 step 如何执行

## 第一步：读取并标准化 patch

每张 patch 单独执行：

\[
x' = \frac{x-\operatorname{mean}(x)}
{\max(\operatorname{std}(x),10^{-6})}
\]

也就是每张 patch 独立减均值、除标准差。

当前没有在线亮度、噪声、模糊、旋转等数据增强；随机性主要来自 batch 采样，模型内部有 `dropout=0.12`。

## 第二步：共享网络前向

anchor 和 positive 使用同一个 HardNet Strong V2：

```text
256 anchor patches   ─┐
                      ├─ 同一个 HardNet → 256维 L2 单位描述子
256 positive patches ─┘
```

输出：

```text
anchor_desc:   [256, 256]
positive_desc: [256, 256]
```

两个分支共享全部权重，不是两个独立网络。

## 第三步：计算距离矩阵

计算所有 anchor 与所有 positive 的 L2 距离：

```text
distances[i, j] = distance(anchor_i, positive_j)
```

得到：

```text
[256, 256] 距离矩阵
```

对角线是正样本：

```text
distances[i, i] = 正样本距离
```

非对角线是潜在负样本。

## 第四步：屏蔽非法负样本

当前策略是：

```yaml
hard_negative_strategy: same_finger_allowed
```

同一手指的不同物理点允许成为负样本，但以下候选会被屏蔽：

1. `i == j`，即正样本自身；
2. 相同 `point_group`，即同一真实物理点；
3. 同一手指、同一原图坐标系内距离过近的点。

空间规则是：

\[
\max(|dx|,|dy|) \ge 16
\]

也就是位于 anchor 中心 `32×32` 方形邻域内的点不能作为负样本。

双向寻找负样本：

```text
anchor_i   → 所有 positive_j
positive_i → 所有 anchor_j
```

两个方向分别使用对应原图坐标系判断空间距离。

## 第五步：选择 top-3 最难负样本

当前配置：

```yaml
hard_negative_top_k: 3
margin: 0.8
```

对每个正样本，从所有合法候选中选择距离最小的 3 个负样本。

例如：

```text
正样本距离：0.35

三个最难负样本距离：
0.42
0.48
0.61
```

分别计算：

\[
L_k=\max(0,\ 0.8+d_{pos}-d_{neg,k})
\]

然后：

1. 对这 3 个负样本损失求平均；
2. 再对 batch 中所有有效正样本求平均。

距离矩阵和 loss 使用 FP32，卷积前向使用 FP16，避免困难负样本距离非常接近时产生半精度排序误差。

## 第六步：反向传播

```text
loss
  ↓
GradScaler 缩放
  ↓
backward
  ↓
SGD + momentum=0.9 + Nesterov
  ↓
更新模型参数
```

---

# 4. “一个 epoch”实际代表什么

当前一个 epoch 不是完整遍历一次 `train_pairs.csv`，而是固定随机生成：

```text
1000 个 batch
× 256 个正样本对
= 256,000 次正样本对采样
```

每轮采样器使用不同但可复现的随机序列：

```text
epoch 1 → seed 42
epoch 2 → seed 43
epoch 3 → seed 44
...
```

因此：

- 某些 CSV 记录一轮内可能被抽中多次；
- 某些记录可能一轮内没有被抽中；
- `epoch` 本质上是固定 `1000` 次参数更新的逻辑周期。

当前共训练：

```text
100 epochs × 1000 steps = 100,000 次参数更新
```

学习率：

```text
前 3 个 epoch：从 0 warmup 到 0.08
之后：余弦下降到 0.0001
```

---

# 5. 验证集如何构建

验证集现在使用 `fixed_candidate_pool_v3`。固定 batch 和合法候选身份每轮不变；完整候选池负责总体 ROC 和距离分布，动态 top-k 只负责 margin loss。

示例配置：

```yaml
validation:
  protocol: fixed_candidate_pool_v3
  seed: 10042
  finger_count: auto
  batch_count: 256
  batch_size: 24
```

训练启动时只构建一次固定计划：

```text
统计 val 中可用手指数 N
    ↓
auto 解析为 min(N, 24)
    ↓
每根手指的 image_pair_id 按固定 seed 洗牌
    ↓
第 i 个验证 batch:
    每根手指使用自己调度序列中的第 i 个 image_pair_id
    将 24 对正样本尽量均匀分配到所有选中手指
    ↓
一共固定生成 256 个 batch
```

当前验证 CSV 有 3 根手指，因此 `auto` 解析为 3，该固定计划总共提供：

```text
256 batch × 24 对正样本 = 6,144 对正样本
```

`batch_count` 与训练的 `steps_per_epoch` 类似，控制每轮执行多少个 batch；区别是验证不反向传播。某根手指的 image pair 全部使用一轮后会重新洗牌循环，因此 `batch_count` 大于 image pair 数量不会报错。

同一个 `seed`、同一份验证 CSV 和同一配置始终生成相同的手指、`image_pair_id`、样本索引及 batch 顺序。每个 epoch 都复用该计划，指标可以直接横向比较。

如果某个 image pair 中独立 `point_group` 不足以填满分配数量，会像训练 sampler 一样从已选正样本中重复补齐；重复项仍会被 `point_group` mask 屏蔽，不能互相充当负样本。

---

# 6. 验证 batch_size=24 的真实含义

验证的 `batch_size=24` 现在与训练 batch 含义一致：它表示一个 batch 中有 `24` 对正样本，而不是唯一 patch 编码数量。

每个验证 batch 的数据流是：

```text
24 对 anchor / positive patch
    ↓
共享网络前向
    ↓
24×24 anchor-positive 距离矩阵
    ↓
与训练相同的非法候选屏蔽：
  正样本自身
  相同 point_group
  同指同原图中的 16px 近邻
    ↓
与训练相同的双向 top-k hardest negatives
```

候选处理规则：

- 合法候选不少于 `hard_negative_top_k`：使用最难的 k 个；
- 合法候选少于 k：只使用实际存在的候选，不补固定负样本配额；
- 单个 anchor 没有合法候选：跳过该 anchor，并计入 `val_skipped_anchor_count`；
- 整个固定计划都没有有效 anchor：给出明确配置错误。

验证仍然满足：

- `model.eval()`，关闭 dropout；
- 使用 `torch.inference_mode()`；
- 不做反向传播，不更新参数；
- 浮点验证复用训练的 `HardNetLoss` 候选掩码和 top-k 实现；
- 二值验证复用相同候选索引，并用真实 Hamming 距离计算指标。

---

# 7. 每轮完整训练流程

```text
开始 epoch
    ↓
随机生成 1000 个训练 batch
    ↓
每个 batch:
    32 根手指
    × 每根手指一个 image pair
    × 每根手指 8 个正样本
    = 256 个正样本对
    ↓
HardNet 前向
    ↓
256×256 距离矩阵
    ↓
屏蔽同物理点和近邻伪负样本
    ↓
双向 top-3 hardest negatives
    ↓
Triplet margin loss
    ↓
反向传播、SGD 更新
    ↓
完成 1000 step
    ↓
执行固定 in-batch 验证计划
    ↓
计算 val_loss、平均正负距离、FPR@TPR95 和正样本 p95
    ↓
保存 last.pt
    ↓
按 matching_composite_v1 四项综合分保存 best.pt
    ↓
写入 metrics.csv
    ↓
下一个 epoch
```

当前 `early_stop_patience: 0`，表示关闭早停，所以正常情况下会完成全部 100 个 epoch。最终训练产物中：

- `last.pt`：最后一轮模型；
- `best.pt`：验证集 `val_checkpoint_selection_score` 最高的模型；
- `metrics.csv`：每轮训练和验证指标；
- `training_curves.png`：训练曲线。

---

# 8. 训练与验证风险边界

启动前会直接拒绝能够确定的配置错误：

- 训练或验证 `batch_size < 2`、`batch_count < 1`、`steps_per_epoch < 1`；
- 显式 `validation.finger_count` 超过验证集可用手指数或验证 `batch_size`；
- `different_finger` 策略实际不足两根训练或验证手指；
- CSV 缺失、为空、缺少坐标字段，或旧验证字段/协议仍在使用；
- checkpoint 的模型、优化器、负采样或验证协议与当前实验不兼容；
- 固定验证计划超过一千万个正样本槽位。

仍需在运行时观察的风险：

- 训练显存主要按 `training.batch_size²` 增长；CUDA OOM 时优先减小训练 `batch_size`；
- CSV 中的 patch 路径可能失效，或图片可能损坏，这类问题在 DataLoader 实际读图时暴露；
- Windows 多进程读取异常时可降低 `data.num_workers`，必要时设为 `0`；
- 某个 anchor 无合法负样本时会跳过，整个验证计划无有效 anchor 才会报错；
- 学习率过高、输入异常或优化器状态损坏可能产生 NaN/Inf；
- checkpoint、指标和曲线写入可能因磁盘空间或权限不足失败。

`batch_count` 大于 image pair 数量不会因为数据不足报错，image pair 会确定性循环。每批验证指标会立即转存 CPU，所以 `batch_count` 不直接提高单步 GPU 显存峰值；但验证耗时、固定索引计划和 CPU 指标缓存会随 `batch_count × batch_size` 近似线性增长。超过十万个正样本槽位会警告，超过一千万会在构建前拒绝。
