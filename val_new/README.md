# val_new: SIFT 困难样本上的公平验证

这个目录是一套新的验证项目，目标是回答一个更具体的问题：

```text
在 SIFT/RootSIFT 表现差、内点数少的图像对上，
HardNet 是否能在相同匹配参数下取得更多内点？
```

## 核心原则

公平对比时保持：

```text
同一批图像对
同一批 SIFT keypoints
同一种匹配策略
同一组匹配参数
```

唯一变化是：

```text
RootSIFT descriptor
HardNet descriptor
```

## 第一步：选择 SIFT 困难图像对

默认从 `outputs/hardnet_dataset/test_pairs.csv` 中扫描图像对，用 RootSIFT 的经典策略：

```text
ratio=0.85
mutual=true
RANSAC reproj_thresh=1.0
```

然后按 RootSIFT RANSAC 内点数从少到多选择 50 对。

正式运行：

```powershell
python -m val_new.select_sift_hard_cases --config val_new/config.yaml
```

快速调试：

```powershell
python -m val_new.select_sift_hard_cases `
  --config val_new/config.yaml `
  --bottom-k 5 `
  --max-image-pairs 20
```

输出：

```text
outputs/val_new_eval/hard_cases/all_sift_case_scores.csv
outputs/val_new_eval/hard_cases/hard_cases.csv
outputs/val_new_eval/hard_cases/hard_case_selection_summary.json
```

每个困难样本还会输出一个目录：

```text
outputs/val_new_eval/hard_cases/case_001_sift_inliers_xxx_.../
  image_a.bmp
  image_b.bmp
  rootsift_inlier_preview.png
```

其中 `rootsift_inlier_preview.png` 是 RootSIFT 的 RANSAC 内点连线图，用来人工分析为什么 SIFT 表现差。

## 第二步：在困难图像对上做公平对比

正式运行：

```powershell
python -m val_new.eval_fair_matching --config val_new/config.yaml
```

输出：

```text
outputs/val_new_eval/fair_matching/fair_matching_details.csv
outputs/val_new_eval/fair_matching/fair_matching_summary.csv
outputs/val_new_eval/fair_matching/fair_matching_summary.json
outputs/val_new_eval/fair_matching/fair_matching_summary.png
```

## 当前默认比较策略

配置在 `val_new/config.yaml` 的 `strategies` 里：

```text
ratio_0.85_two_way
ratio_0.95_no_two_way
top2_mutual
top5_mutual
```

这些策略会同时用于 RootSIFT 和 HardNet。

## 关键指标

`fair_matching_summary.csv` 中建议优先看：

```text
sift_one_to_one_inliers_mean
hardnet_one_to_one_inliers_mean
mean_one_to_one_delta
hardnet_better_one_to_one_rate
```

原因：

```text
top-k 策略可能产生一对多候选。
one-to-one inliers 会同时限制一个 A 点最多匹配一个 B 点、一个 B 点最多匹配一个 A 点，
比普通 inliers 更严格，更适合作为正式报告指标。
```

普通字段含义：

```text
*_raw_matches
  RANSAC 前的粗匹配数量。

*_inliers
  RANSAC 判断为内点的 correspondence 数量。

*_unique_query_inliers
  RANSAC 内点中，去重后的 A 图 keypoint 数。

*_one_to_one_inliers
  RANSAC 内点中，按描述子距离贪心保留的一对一匹配数量。
```

## 附加工具：检查训练 batch 和最难负样本

这个脚本直接复用 `hardnet_train` 里的 batch 构建逻辑，输出每个 batch 中每个样本的：

```text
anchor patch
positive patch
hardest negative patch
```

快速运行：

```powershell
python -m val_new.inspect_hardnet_batches `
  --config val_new/config.yaml `
  --pairs-csv outputs/hardnet_dataset/train_pairs.csv `
  --batches 4 `
  --batch-size 128 `
  --fingers-per-batch 8
```

如果只是调试脚本，可限制读取行数：

```powershell
python -m val_new.inspect_hardnet_batches `
  --config val_new/config.yaml `
  --pairs-csv outputs/hardnet_dataset/test_pairs.csv `
  --batches 1 `
  --batch-size 4 `
  --fingers-per-batch 1 `
  --max-rows 100
```

输出：

```text
outputs/val_new_eval/batch_inspect/batch_manifest.csv
outputs/val_new_eval/batch_inspect/batch_inspect_summary.json
outputs/val_new_eval/batch_inspect/batch_000/batch_records.csv
outputs/val_new_eval/batch_inspect/batch_000/sample_000_.../
  anchor.png
  positive.png
  hard_negative.png
  triplet_panel.png
```

`triplet_panel.png` 是三联图，方便直接观察训练时 hardest negative 是否合理。
