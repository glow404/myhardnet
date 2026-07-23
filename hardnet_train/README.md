# hardnet_train

`hardnet_train` 是指纹 patch 版 HardNet 训练包。它不负责生成 patch，而是读取
`pair_build` 已经生成好的 `train_pairs.csv` / `val_pairs.csv` 和 patch PNG 文件。

## 文件说明

- `model.py`：HardNet/L2Net 网络结构。输入 `[B, 1, 32, 32]`，输出 `[B, 128]`。
- `loss.py`：HardNet 论文中的 hardest-in-batch triplet margin loss。
- `data.py`：CSV 数据集、PIL patch 读取、union-find 物理点分组、按手指/两图组合采样的 batch sampler。
- `metrics.py`：运行均值和 `FPR@95TPR` 指标。
- `train.py`：训练主入口，负责配置解析、训练、验证、checkpoint、metrics 输出。
- `config.yaml`：正式训练默认配置。
- `smoke_config.yaml`：快速自检配置，只跑极少 step。

## 输出文件

训练结果默认写入 `../outputs/hardnet_train/`：

- `best.pt`：验证集 `val_fpr95` 最低的 checkpoint。
- `last.pt`：最后一个 epoch 的 checkpoint。
- `metrics.csv`：每个 epoch 的训练/验证指标。
- `resolved_config.json`：本次训练实际使用的配置快照。

## 早停规则

训练默认监控 `val_fpr95`，即验证集上的 `FPR@95%` 召回率。该指标越低越好。

默认终止条件：

```text
连续 3 个 epoch 没有让 val_fpr95 相对历史 best 下降至少 1%，则停止训练。
```

例如历史 best 为 `0.9868`，那么下一次必须低于：

```text
0.9868 * (1 - 0.01) = 0.976932
```

才算一次显著改善。否则累计 `no_improve_epochs`。

可在配置中调整：

```yaml
training:
  early_stop_patience: 3
  early_stop_min_relative_improvement: 0.01
```

也可以用命令行覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --early-stop-patience 3 `
  --early-stop-min-delta 0.01
```

## Margin

HardNet 论文原始 triplet margin 为 `1.0`。由于小指纹 patch 的类间差异可能更小，
当前默认把 margin 调为 `0.5`，让训练目标稍微温和一些。

命令行覆盖示例：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --margin 0.5
```

## 采样策略

HardNet 的 top-k hardest-in-batch loss 会从 batch 内选择最相似的若干非正样本。
指纹数据里同一物理点可能跨多张图重复出现，因此普通随机 batch 容易产生伪负样本。

负样本候选策略由 `training.hard_negative_strategy` 控制：

- `same_finger_allowed`：允许同一 `finger_id` 内的样本成为负样本，但会屏蔽同一 `point_group`。如果 CSV 中存在 A-B、B-C 两条正样本关系，即使没有显式 A-C，union-find 也会把 A、B、C 合并为同一物理点组，loss 不会把 A-C 当负样本。
- `different_finger`：最难负样本只能来自不同 `finger_id`。使用这个策略时，每个 batch 至少需要 2 根手指，因此 `fingers_per_batch` 也必须大于等于 2。

难负样本数量由 `training.hard_negative_top_k` 控制，默认是 `3`。对每条正样本，loss 会合并
anchor→positive 和 positive→anchor 两个方向的合法负样本候选，选距离最小的 k 个，分别计算
triplet margin loss 后求均值。合法候选不足 k 个时只使用现有候选；设为 `1` 可复现原来的
单一 hardest-negative 行为。日志中的 `neg_dist` 是实际选中 top-k 负样本的平均距离。

本包还采用两层采样保护：

1. 每个 batch 选多个 `finger_id`。
2. 每个 `finger_id` 在当前 batch 中只使用一个 `image_pair_id`，也就是只来自两张图。
3. 通过 union-find 把正样本关系连起来的关键点合并为 `point_group`。
4. loss 计算时屏蔽同一 `point_group` 的候选负样本。

命令行可临时覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --hard-negative-strategy different_finger `
  --hard-negative-top-k 3 `
  --fingers-per-batch 8
```

## 常用命令

快速自检：

```powershell
python -m hardnet_train.train --config hardnet_train/smoke_config.yaml
```

正式训练：

```powershell
python -m hardnet_train.train --config hardnet_train/config.yaml
```

CUDA 大 batch 训练：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512
```

## 续训

长训练可以从上一次 `last.pt` 继续：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/post_nips_config.yaml `
  --device cuda `
  --output-dir ../outputs/hardnet_train_post_nips `
  --resume auto
```

也可以指定 checkpoint：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/post_nips_config.yaml `
  --resume ../outputs/hardnet_train_post_nips/last.pt
```

续训会恢复：

- 模型权重；
- optimizer 状态；
- epoch；
- global step；
- 既有 `metrics.csv` 中的 best FPR95 和早停计数。

## 学习率调度

当前默认策略是：

```text
warmup + cosine decay
```

配置项：

```yaml
training:
  scheduler: warmup_cosine
  warmup_epochs: 2
  eta_min: 0.0001
```

含义：

1. 前 `warmup_epochs` 个 epoch 从 0 线性升到初始学习率；
2. 后续用余弦退火缓慢下降到 `eta_min`；
3. 不会像原来的线性衰减一样最后直接变成 0。

如果要复现论文原始线性衰减，可改为：

```yaml
training:
  scheduler: linear
```

## 训练曲线

训练结束后会自动生成：

```text
training_curves.png
```

图中包含：

- train/val loss；
- train/val 正负样本距离；
- val_fpr95；
- 学习率曲线。

如不需要绘图，可加：

```powershell
--no-plot
```
