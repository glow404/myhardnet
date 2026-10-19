# hardnet_train

`hardnet_train` 是指纹 patch 版 HardNet 训练包。它不负责生成 patch，而是读取
`pair_build` 已经生成好的 `train_pairs.csv` / `val_pairs.csv` 和 patch PNG 文件。

## 文件说明

- `model.py`：保留原 HardNet、MobileHardNet 与 HardNet Strong V2，统一接收 `[B, 1, 32, 32]` patch。
- `model_report.py`：实际执行一次前向，生成逐层输入/输出、作用、参数量与 MACs/FLOPs 表。
- `loss.py`：top-k hardest-in-batch triplet margin loss、top-1 加权和正样本 p95/p99 尾部约束。
- `data.py`：CSV 数据集、轻量在线增强、坐标契约、PIL patch 读取、union-find 物理点分组和训练 batch sampler。
- `negative_sampling.py`：空间近邻规则，供训练和验证共同使用；验证指标不按同指/跨指拆分。
- `validation.py`：固定验证 batch 计划；batch 构造、候选掩码和 top-k 规则与训练共用。
- `checkpoint_selection.py`：把验证 loss、平均距离间隔、误接受和正样本 p95 合成为 best checkpoint 分数。
- `binary_model.py`：浮点 teacher、residual hash head、STE 量化和 bit packing 契约。
- `binary_loss.py`：二值排序、teacher 关系保持、正样本一致性、量化、bit balance 和去相关组合损失。
- `binary_validation.py`：固定验证计划上的归一化 Hamming 距离和 bit 分布指标。
- `train_binary.py`：二值网络训练、续训、验证和 checkpoint 入口。
- `config_binary_256.yaml`：带逐项注释的 256-bit 二值网络正式训练配置。
- `optim.py`：SGD/AdamW 工厂与 decay/no-decay 参数分组。
- `train.py`：训练主入口，负责配置解析、训练、固定验证、checkpoint 和 metrics 输出。
- `config.yaml`：浮点描述子的正式训练配置，默认训练 Strong V2。
- `smoke_config.yaml`：快速自检配置，只跑极少 step。

## 输出文件

训练结果默认写入 `../outputs/models/hardnet_train_strong_v2_256_fixed_candidate_pool_v3/`：

- `best.pt`：固定候选池协议上 `matching_composite_v1` 四项综合分数最高的 checkpoint；综合全部验证候选，不拆分同指/跨指分项。
- `last.pt`：最后一个 epoch 的 checkpoint。
- `metrics.csv`：每个 epoch 只记录与最终匹配最相关的精简指标，浮点值最多保留 4 位有效数字。
- `resolved_config.json`：包含命令行覆盖后的实际配置快照。
- `model_structure.md`：当前模型的 PyTorch 模块树、汇总参数量，以及按实际前向顺序整理的逐层输入/输出、作用、参数量和计算量表。
- `model_structure.csv`：与 Markdown 逐层表相同的机器可读版本，使用 UTF-8 BOM，便于直接用 Excel 打开。

每次启动浮点或二值训练都会在 checkpoint 的同一 `output_dir` 重新生成上述结构报告。
计算量以单个 `[1,1,32,32]` patch 为单位：MACs 统计 Conv/Linear 乘加，FLOPs
按 `1 MAC ≈ 2 FLOPs` 约算；归一化、激活、池化、插值、拼接和逐元素运算不计入
MACs，因此该统计适合做模型间统一比较，但不代表设备上的实际推理延迟。

### 精简训练指标

新的 `metrics.csv` 每个 epoch 只写入以下字段：

- `epoch`
- `train_loss`
- `train_positive_tail_loss`
- `train_positive_p95`、`train_positive_p99`
- `val_loss`
- `val_pos_mean`、`val_neg_mean`
- `val_pos_p95`、`val_neg_p01`
- `val_fpr_at_tpr95`
- `val_tpr_at_fpr_1e_4`
- `val_checkpoint_selection_score`：只综合 `val_loss`、`val_neg_mean - val_pos_mean`、`val_fpr_at_tpr95` 和 `val_pos_p95`，越高越好。
- `val_valid_anchor_count`：当前固定计划中至少存在一个合法负样本的 anchor 数量。
- `val_skipped_anchor_count`：当前 batch 内完全没有合法负样本、因此未参与 loss/ROC 的 anchor 数量。
- `lr`
- `early_stop_best_selection_score`
- `no_improve_epochs`

浮点指标以 4 位有效数字写入 CSV；epoch 和早停计数保持整数。验证集只记录总体指标，不按同指/跨指拆分。

训练曲线同步展示四项选模信号：训练/验证 loss、正样本 p95、FPR@TPR95 和平均正负距离间隔。旧 schema 的 `metrics.csv` 不能与新 schema 混写，继续训练必须使用匹配新策略的新输出目录。

## 网络架构切换

`config.yaml` 使用 `descriptor_dim: auto` 和 `output_dir: auto`，会按架构选择兼容维度及独立输出目录：

| `model.architecture` | 自动维度 | 自动输出目录 | 网络状态 |
| --- | ---: | --- | --- |
| `hardnet_strong_v2` | 256 | `outputs/models/hardnet_train_strong_v2_256_fixed_candidate_pool_v3` | 默认高精度主干 |
| `mobile_hardnet` | 128 | `outputs/models/hardnet_train_mobile_128_fixed_candidate_pool_v3` | 带 Coordinate Attention 的轻量主干 |
| `hardnet` | 128 | `outputs/models/hardnet_train_hardnet_128_fixed_candidate_pool_v3` | 论文版 HardNet 主干 |

三种架构都支持把 `model.descriptor_dim` 设置为任意正整数；表中只是 `auto` 的
默认值。自定义维度只改变各模型的描述子输出投影和末端归一化层，不改变前面的
特征提取主干。`output_dir: auto` 会把实际维度写进目录名，避免不同维度的
checkpoint 混放。原 HardNet 配置为 128 时仍与旧版本保持 checkpoint 兼容。

当前 MobileHardNet 在 96 通道 adapter 与全图 depthwise 描述子投影之间加入一个
Coordinate Attention，通过横、纵两个方向的池化保留位置敏感的脊线信息。
架构标识仍为 `mobile_hardnet`，但结构已经更新，旧 MobileHardNet checkpoint
不再兼容，必须重新训练；原 HardNet checkpoint 兼容性不受影响。

当前 Strong V2 是针对 32×32 patch 收缩后的轻量高精度版本：Stem 只保留
3×3 局部纹理和局部对比度两条分支；三个 stage 使用 32/64/128 通道及
1/2/2 个普通残差块。逐块 SE 已移除，包含下采样块在内的 7 个残差块按照深度
把 Stochastic Depth 从 0 线性增加到 `model.drop_path_rate`。原四分支上下文被
单路扩张 depthwise ContextMixer 替代；8×8 特征通过 depthwise 8×8 空间投影和
GeM 全局投影得到两个同维描述子，分别 L2 单位化后由一个可学习标量融合。

Strong V2 的架构标识仍为 `hardnet_strong_v2`，但参数形状已经改变，旧 Strong V2
checkpoint 不兼容。必须使用新的输出目录从头训练，并在浮点训练完成后把二值配置
的 `backbone.checkpoint` 更新到新生成的 `best.pt`。

因此，只把下面一行改为 `mobile_hardnet`，即可使用当前带 Coordinate Attention
的 MobileHardNet 网络训练：

```yaml
model:
  architecture: mobile_hardnet
  descriptor_dim: auto
```

该切换会自动改变网络结构、描述子维度和输出目录。当前 `dropout: 0.12`、`warmup_epochs: 3`、`lr: 0.08` 是 Strong V2 默认训练参数；若要严格复现旧 Mobile 配置，还应改回 `dropout: 0.1`、`warmup_epochs: 2`、`lr: 0.1`。

## 训练在线增强

浮点训练的三种架构和二值训练共用 `FingerprintPairDataset`，默认只对训练集启用
轻量在线增强；固定验证集不增强。anchor 与 positive 独立采样扰动，使模型学习对
关键点定位和成像误差的稳定性，而不是记忆固定 patch。默认参数为：

| 扰动 | 默认强度 |
| --- | ---: |
| 整体增强概率 | 0.80 |
| 旋转 | ±5° |
| 平移 | ±1 px |
| 缩放 | ±3% |
| 对比度 | ±10% |
| Gamma | ±10% |
| 高斯模糊 | 概率 0.15，半径不超过 0.40 |
| 高斯噪声 | 概率 0.25，标准差不超过 2/255 满量程 |

增强发生在单 patch 标准化之前，不使用水平/垂直翻转和大角度旋转，避免破坏已经
完成的关键点方向对齐。设置 `data.train_augmentation.enabled: false` 可关闭。
实际增强参数会写入 `resolved_config.json` 和 checkpoint 契约；启用增强后不能把旧的
无增强 checkpoint 作为同一实验恢复优化器状态，应使用新输出目录重新训练。

`matching_composite_v1` 是描述子阶段的代理目标，不替代最终 `match_new` 全量模板、双向验证和 Lowe 比率评估；正式部署前仍应在独立匹配实验中确认阈值和误报行为。

## 二值网络训练

二值网络不是把浮点描述子在推理时直接执行一次 `sign`。`train_binary.py` 会加载已经训练好的浮点/L2 checkpoint 作为 teacher，在其后增加 residual hash head，通过可退火的 `tanh + straight-through sign` 学习固定长度的 0/1 bit code。

组合损失包含：

- **Hamming 排序代理损失**：保持正样本比 hardest-in-batch 负样本更近；
- **teacher 关系保持**：让 student 的 batch 内两两相似度接近浮点 teacher；
- **正样本一致性**：降低同一物理点二值码的不一致；
- **量化约束**：推动连续输出接近 `-1/+1`，减少部署时 sign 误差；
- **bit balance / decorrelation**：减少恒定 bit 和重复 bit。

### 训练前提

开始二值训练前必须满足以下条件：

1. 已从 `pair_build` 生成 `outputs/butieping/train_pairs.csv` 和 `outputs/butieping/val_pairs.csv`，CSV 引用的 patch PNG 路径可读取；
2. train/val 已按 `finger_id` 隔离，同一根手指不会跨集合；
3. 已完成当前轻量 Strong V2 浮点网络训练，并存在 float/L2 teacher，例如 `outputs/models/hardnet_train_strong_v2_256_fixed_candidate_pool_v3/best.pt`；
4. `config_binary_256.yaml` 中的 `backbone.checkpoint` 指向该 teacher；程序会校验 checkpoint 类型、架构和描述子维度，并拒绝 binary/Hamming checkpoint；
5. 正式训练使用新的 `output_dir`。目录中已有 `metrics.csv` 时，必须显式续训，程序不会覆盖旧实验；
6. `training.device: cuda` 时，当前 PyTorch 必须能够使用 CUDA。

可在项目根目录执行以下检查：

```powershell
Test-Path outputs\butieping\train_pairs.csv
Test-Path outputs\butieping\val_pairs.csv
Test-Path outputs\models\hardnet_train_strong_v2_256_fixed_candidate_pool_v3\best.pt
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_count=', torch.cuda.device_count())"
```

前三个 `Test-Path` 应输出 `True`；正式 CUDA 训练时 `cuda_available` 也应为 `True`。

如果还没有浮点 teacher，先执行：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
python -m hardnet_train.train --config hardnet_train/config.yaml
```

浮点训练完成后，确认其 `best.pt` 路径与二值配置中的 `backbone.checkpoint` 一致。

### 配置文件

正式二值配置是 `hardnet_train/config_binary_256.yaml`。文件内已经逐项注释以下内容：

- 数据 CSV、DataLoader worker、预取和调试截断参数；
- 浮点 teacher checkpoint、架构和维度校验；
- bit 数、hash head 宽度、dropout、量化温度和 bit 打包位序；
- 六项组合损失的职责和权重；
- batch 结构、hard negative、学习率调度、梯度裁剪和早停；
- 固定 Hamming 验证计划及 `best.pt` 综合选择权重。

默认采用 `hash_bits: 256` 和 `backbone_trainable: false`，即冻结 teacher、只训练 hash head。这是更稳定的第一阶段。若后续需要联合微调，应复制配置到新的输出目录，将 `backbone_trainable` 改为 `true`，并显著降低学习率，避免破坏已经训练好的浮点描述子。

二值训练虽然复用 `HardNetLoss` 作为 metric loss，但不会再依赖其中隐藏的默认值。
`training.hard_negative_top1_weight` 和五个 `positive_tail_*` 参数会写入
`resolved_config.json`、checkpoint 续训契约，并被显式传给实际损失。默认尾部权重
为 `0.05`，低于浮点训练的 `0.10`，因为二值组合损失已经另有权重 `0.10` 的
`positive_consistency_loss`。二值 p95/p99 L2 目标 `0.60/0.80` 分别约对应
`9%/16%` 的正样本 bit 差异比例。

### 小规模自检

先用极小训练/验证预算确认 CSV、patch、teacher、CUDA 和 checkpoint 写入链路都正常：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --device cuda `
  --epochs 1 `
  --steps-per-epoch 2 `
  --batch-size 32 `
  --val-batch-count 2 `
  --val-batch-size 12 `
  --output-dir ../outputs/models/hardnet_binary_256_smoke
```

若该 smoke 输出目录已经存在 `metrics.csv`，请换一个新目录名；不要删除或覆盖仍需保留的实验。

### 正式训练命令

配置中的正式预算为 150 epoch、每 epoch 1000 step，直接执行：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --device cuda
```

也可以显式覆盖常用参数：

```powershell
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --backbone-checkpoint ../outputs/models/hardnet_train_strong_v2_256_fixed_candidate_pool_v3/best.pt `
  --hash-bits 256 `
  --batch-size 256 `
  --val-finger-count auto `
  --val-batch-count 256 `
  --val-batch-size 24 `
  --lr 0.001 `
  --output-dir ../outputs/models/hardnet_binary_256_fixed_candidate_pool_v3
```

`--backbone-checkpoint` 和 `--output-dir` 会作为配置值解析，相对路径以 `hardnet_train/config_binary_256.yaml` 所在目录为基准；`--resume` 的显式相对路径则以当前工作目录为基准。为避免歧义，建议优先在 YAML 中固定数据、teacher 和输出路径，命令行只覆盖临时实验参数。

### 续训命令

从当前输出目录的 `last.pt` 恢复模型、optimizer、epoch、global step 和 AMP scaler：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:OMP_NUM_THREADS='1'
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --resume auto
```

也可以显式指定 checkpoint：

```powershell
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --resume outputs/models/hardnet_binary_256_fixed_candidate_pool_v3/last.pt
```

续训时不能修改 backbone、hash bit 数、hash head 结构、bitorder、优化器或固定负样本协议；这些实验变量变化时必须使用新的输出目录重新训练。

### 训练产物和关键指标

默认输出目录为 `outputs/models/hardnet_binary_256_fixed_candidate_pool_v3/`：

- `best.pt`：`val_checkpoint_selection_score` 最高的二值 checkpoint；
- `last.pt`：最后完成 epoch 的 checkpoint，用于续训；
- `metrics.csv`：训练损失、Hamming 验证、bit 分布和 checkpoint 选择指标；
- `resolved_config.json`：应用命令行覆盖后的最终配置。

每轮二值验证复用固定 batch 计划、合法负样本 mask 和双向 top-k，并统一写入 `val_` 前缀。重点观察：

- `val_checkpoint_selection_score`：综合 best 选择分数，越高越好；
- `val_pos_mean` / `val_neg_mean`：正负样本平均归一化 Hamming 距离；
- `val_pos_p95`：正样本 Hamming 距离 p95，越低越好；
- `val_fpr_at_tpr95` / `val_tpr_at_fpr_1e_4`：总体 ROC 指标；
- `val_bit_balance_error`：各 bit 偏离 50% 取 1 的平均程度，越低越好；
- `val_constant_bit_ratio`：恒 0 或恒 1 bit 比例，理想值为 0；
- `val_valid_anchor_count` / `val_skipped_anchor_count`：确认固定计划产生了足够合法负样本。

验证 Hamming 距离范围是 `[0, 1]`。`validation.batch_count` 超过 image pair 数时会确定性循环使用，主要增加验证时间和 CPU 指标缓存，不直接增加单批 GPU 峰值。

二值 checkpoint 当前不能直接交给只支持浮点/L2 描述子的 `match_new`。投入匹配流程前，需要另行实现二值模板存储、bitorder 契约和 Hamming 距离分支。


## 早停规则

训练监控固定验证 batch 计划上的 `matching_composite_v1` 四项综合分数。它只使用四项总体指标：动态 hard top-k 的 `val_loss`、完整合法候选池的 `val_neg_mean - val_pos_mean` 和 `val_fpr_at_tpr95`，以及总体正样本的 `val_pos_p95`。

当前正式配置将 `early_stop_patience` 设为 `0`，即关闭早停。启用后，连续 patience 个 epoch 没有让 `val_checkpoint_selection_score` 相对历史 best 提升配置比例，就停止训练：

```text
历史 best * (1 + early_stop_min_relative_improvement)
```

例如历史 best 为 `0.80`、比例为 `0.01`，下一次必须超过 `0.808` 才算显著改善。`val_fpr_at_tpr95` 仍保留为核心观测指标，但不再单独决定 `best.pt`。

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

HardNet 论文原始 triplet margin 为 `1.0`。

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

- `same_finger_allowed`：允许同一 `finger_id` 内的样本成为负样本，但会先屏蔽同一 `point_group`。如果 CSV 中存在 A-B、B-C 两条正样本关系，即使没有显式 A-C，union-find 也会把 A、B、C 合并为同一物理点组，loss 不会把 A-C 当负样本。对仍然合法的同指候选，仅在同一原图坐标系内比较坐标，并要求 `max(|dx|, |dy|) >= same_finger_min_coordinate_separation_px`；默认阈值为 16px，等于 16 时允许。
- `different_finger`：最难负样本只能来自不同 `finger_id`。使用这个策略时，每个 batch 至少需要 2 根手指，因此 `fingers_per_batch` 也必须大于等于 2；同指空间规则不会影响跨指候选。

难负样本数量由 `training.hard_negative_top_k` 控制，默认是 `3`。对每条正样本，loss 会合并
anchor→positive 和 positive→anchor 两个方向的合法负样本候选，选距离最小的 k 个，分别计算
triplet margin loss。`training.hard_negative_top1_weight` 默认是 `0.6`，最近的 top-1 负样本
获得 60% 权重，其余有效候选均分剩余权重；合法候选不足 k 个时会按实际候选重新归一化。
设 `hard_negative_top_k=1` 可复现单一 hardest-negative 行为。日志中的 `neg_dist` 是实际选中
top-k 负样本的平均距离。

为避免训练后期只优化负样本而让正样本长尾恶化，loss 还对有效正样本距离的 p95/p99 加入平方
hinge 约束：

```yaml
training:
  positive_tail_loss_weight: 0.10
  positive_tail_p95_weight: 0.7
  positive_tail_p99_weight: 0.3
  positive_tail_p95_target: 0.60
  positive_tail_p99_target: 0.80
```

目标是单位描述子的 L2 距离。p99 在 batch=256 时只有少量样本参与分位点梯度，因此默认低于
p95 的权重。`train_positive_tail_loss` 记录该项加权后的损失；如果正样本 p95/p99 已低于目标，
该项为 0，不会继续压缩已经合格的正样本。

本包还采用两层采样保护：

1. 每个 batch 选多个 `finger_id`。
2. 每个 `finger_id` 在当前 batch 中只使用一个 `image_pair_id`，也就是只来自两张图。
3. 通过 union-find 把正样本关系连起来的关键点合并为 `point_group`。
4. loss 计算时屏蔽同一 `point_group` 的候选负样本。
5. 对同指候选，anchor→positive 方向使用 A 图坐标，positive→anchor 方向使用 B 图坐标；只有同一坐标参考图内、位于锚点中心 32x32 方形邻域之外的点才合法。

固定验证使用 `fixed_candidate_pool_v3`。训练启动时按固定 `seed` 固化正样本 batch 和所有合法负样本身份：

1. `validation.finger_count: auto` 时使用 `min(验证集可用手指数, validation.batch_size)`，并把实际值写入 `resolved_config.json` 和 checkpoint；
2. 构造 `validation.batch_count` 个固定验证 batch；
3. 每根手指的 `image_pair_id` 固定洗牌后依次使用，全部用完时重新洗牌循环；
4. 同一 batch 中每根手指仍只使用一个 `image_pair_id`；
5. 每轮验证复用完全相同的 batch 顺序和合法候选 mask；
6. 完整合法候选池计算 ROC、FPR、TPR、EER 和负样本距离分布；每轮动态 top-k 只计算 active-triplet 和 margin loss。

例如：

```yaml
validation:
  protocol: fixed_candidate_pool_v3
  seed: 10042
  finger_count: auto
  batch_count: 256
  batch_size: 24
```

当前验证 CSV 有 3 根手指，因此 `auto` 会解析为 3，生成 `256` 个固定 batch，每个 batch 共 `24` 对正样本，平均每根手指提供 `8` 对。若验证集手指数超过 `batch_size`，`auto` 最多使用 `batch_size` 根手指，以保证每根手指至少贡献一对样本。

`batch_count` 与训练的 `steps_per_epoch` 类似，负责控制每轮执行多少个 batch，但验证不反向传播。某根手指拥有的 image pair 少于 `batch_count` 时会循环使用，不会因为 image pair 数量不足而报错。

合法候选少于 `hard_negative_top_k` 时只使用现有候选；某个 anchor 完全没有合法候选时只跳过该 anchor 并计入 `val_skipped_anchor_count`。只有整个固定计划都没有有效 anchor 时才报错。

新协议改变了验证样本与指标语义，旧验证计划 checkpoint 不能恢复优化器状态，也不能与新 `metrics.csv` 混写。

命令行可临时覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --hard-negative-strategy different_finger `
  --hard-negative-top-k 3 `
  --fingers-per-batch 8 `
  --val-finger-count auto `
  --val-batch-count 256 `
  --val-batch-size 24
```

## 训练与验证风险检查

程序会在启动阶段提前拒绝以下确定性错误：

- `training.epochs <= 0`、`steps_per_epoch <= 0` 或 `batch_size < 2`；
- `training.fingers_per_batch < 1`，或实际每 batch 手指数超过 `batch_size`；
- `validation.batch_count < 1`、`batch_size < 2`，或显式手指数超过验证集/`batch_size`；
- `different_finger` 策略实际少于 2 根训练或验证手指；
- 验证计划超过一千万个正样本槽位；
- CSV 缺失、为空、坐标字段不完整，或配置要求的 CUDA/BF16 不可用；
- checkpoint、模型结构、优化器或训练/验证协议与当前配置不兼容；
- 非续训模式下输出目录已有 `metrics.csv`，避免覆盖既有实验。

以下风险依赖实际运行环境，无法仅靠 YAML 完全排除：

- **训练显存不足**：HardNet 距离矩阵按 `training.batch_size²` 增长；出现 CUDA OOM 时优先减小训练 `batch_size`。
- **patch 文件不存在或损坏**：CSV 可以正常读取，但 DataLoader 真正打开图片时仍会失败；错误路径中会包含具体 patch 文件。
- **没有合法负样本**：空间过滤、`point_group` 屏蔽和手指数量可能让部分 anchor 被跳过；若整个验证计划都没有有效 anchor，验证会明确报错。
- **验证预算过大**：image pair 不足只会循环使用，不会报错；每批指标计算完会立即转存 CPU，因此 `batch_count` 不直接提高单步 GPU 显存峰值，但固定计划、验证耗时和 CPU 指标缓存仍会随 `batch_count × batch_size` 近似线性增长。超过十万个正样本槽位会发出警告，超过一千万会提前拒绝。
- **自动手指数过多**：若 `auto` 最终解析为 `finger_count == batch_size`，每根手指每 batch 只有一对样本，合法负样本数量会减少；程序会发出警告。

启动日志会显示实际验证手指数、固定 batch 数、每根手指使用的唯一 image pair 范围，以及发生 image pair 循环的手指数。

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

续训必须继续使用原训练的有效配置和输出目录。命令行覆盖后的完整配置可在输出目录的 `resolved_config.json` 中确认。

当前 `hardnet_train_cuda_b512` 模型从 `last.pt` 续训：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512 `
  --resume auto
```

也可以显式指定同一个 checkpoint：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512 `
  --resume ../outputs/hardnet_train_cuda_b512/last.pt
```

`--resume auto` 表示必须找到当前输出目录的 `last.pt`，否则直接报错。`--resume-auto` 适合脚本化启动：checkpoint 存在时恢复，不存在时仅在空输出目录中从头训练。

续训会恢复：

- 模型权重；
- optimizer 状态；
- epoch；
- global step；
- 既有 `metrics.csv` 中的 best `val_fpr_at_tpr95` 和早停计数。

只有模型架构、优化器名称、参数分组以及训练/验证候选协议完全兼容时才允许恢复训练。旧模型 checkpoint 仍可用于推理或作为预训练权重；旧 fixed-pair checkpoint 和旧指标 CSV 不能在当前输出目录续写。

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
- `val_fpr_at_tpr95`；
- 学习率曲线。

如不需要绘图，可加：

```powershell
--no-plot
```
