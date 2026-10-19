# FAR / FRR 匹配参数调优指南

面向 `match_new` 离线评估（`run_hardnet_matching.py`）。常调项优先改 `config_match_tuning.yaml`，其余在 `config_match_new.yaml`。

本文回答两件事：

1. 哪些参数**更可能同时压低 FAR 和 FRR**（改善区分能力）；
2. 哪些参数**基本是此消彼长**（一个降、另一个升）。

---

## 1. 先分清两类效果

### 1.1 此消彼长（固定最终分数阈值时）

把门槛拧紧 → 更难通过 → **FAR 降、FRR 升**；放宽则相反。  
典型代表：`identification.match_score_threshold`。  
这类参数**不提高**本人/非本人分数的分离度，只是在同一条曲线上换工作点。

### 1.2 可能两边一起变好（先改善分数分布，再重标定阈值）

目标是让本人分数整体更高、非本人分数整体更低（两条分布拉开）。  
拉开之后，在**同一 FAR 目标**下重选阈值，FRR 往往能下降——这才叫“同时改善”。  
若只改参数、**不重标定阈值**，仍可能看起来像此消彼长。

| 类型 | 含义 | 调参后怎么看 |
| --- | --- | --- |
| 纯取舍 | 分数分布几乎不变，只换接受门槛 | 看固定阈值下的 FAR/FRR，或沿曲线滑动 |
| 改善分离 | genuine / impostor 分数分布拉开 | 看 `FRR @ 目标 FAR`、EER、AUC；必须重标定阈值 |
| 方向不定 | 数据/场景依赖大（如湿手指） | 必须做对照实验，不能凭经验定论 |

下文“增大/放宽”均指**其他条件不变时的常见趋势**；实际可能非单调，以全量实验为准。

---

## 2. 哪些可能同时降低 FAR 与 FRR

这些项的价值在于**拉开分数分离**，而不是单纯拧门槛。调完后务必看曲线工作点（例如 `FRR @ FAR≈2e-5`），不要只盯一个固定阈值。

| 参数 / 手段 | 为什么可能两边都降 | 注意 |
| --- | --- | --- |
| `model.checkpoint`（换更好模型） | 描述子更稳，本人更像、非本人更不像 | 等于换系统：须重建模板，并分别重标定距离阈值与最终分数阈值 |
| `model.hadamard_binarization.enabled` | 换浮点 L2 / 二值 Hamming 契约，可能改变可分性 | 不是“越开越好”；模板与距离阈值不能混用 |
| `texture_verification.enabled` 及融合权重 | 几何碰巧对齐、脊线不对的非本人可被压分；本人双通过可保持高分 | 湿手指、低对比图可能伤本人 → 方向不定，必须开关对照 |
| 候选质量（不是一味加候选） | 去掉明显错误的描述子候选，减少非本人“撞上”RANSAC 内点 | 过松 → 噪声抬 FAR；过严 → 本人候选不足抬 FRR。甜区才可能两边都好 |
| `bidirectional_ratio_test`、`allow_many_to_one_before_ransac` | 抑制一对多噪声或找回被误杀的真候选 | 与 `candidate_policy`、双向开关强耦合；效果看场景 |
| `reproj_error_weight` | 一对一去重时更偏向几何一致的点 | 影响常较小且非单调 |
| SIFT / 关键点上限（与训练分布一致时） | 点更稳、覆盖更好，可同时减假匹配与漏匹配 | **须重建模板**；乱改会偏离训练分布，两边都可能变差 |
| 注册模板数量与覆盖（配合合理融合） | 覆盖更多按压姿态，本人更容易命中 | 在 `fusion_method=max` 下，模板增多也可能抬非本人最高分 → FAR 有上行风险，需重标定后看工作点 |

**实践口诀**：想两边一起降，优先换模型 / 开对纹理融合 / 把候选从“更多”调到“更干净”，再用阈值曲线选工作点；不要指望只拧 `match_score_threshold`。

---

## 3. 哪些基本是一个升、一个降

固定最终分阈值时，下表方向最常见。若你改完后又沿曲线重选阈值，表中趋势仍可参考“容错变宽还是变严”。

### 3.1 最终判定（纯工作点滑动）

| 参数 | 拧紧时 | 放宽时 |
| --- | --- | --- |
| `identification.match_score_threshold` | FAR↓ FRR↑ | FAR↑ FRR↓ |
| `identification.fusion_method`：`max` → `top3_mean` / `mean` | 通常 FAR↓ FRR↑（更保守） | 回到 `max` 通常 FRR↓、FAR 风险升 |

### 3.2 纹理与几何硬门槛（更严 → FAR↓ FRR↑）

| 参数 | 增大 / 更严 |
| --- | --- |
| `geometry_saturation_inliers` | 同样内点数几何分更低 → FAR↓ FRR↑ |
| `low_unique_inliers` | 硬门槛更高 → FAR↓ FRR↑ |
| `texture_weight` 升高（相对 `geometry_weight`） | 常压非本人，但弱纹理本人易受伤 → 多为 FAR↓、FRR↑ 或不定 |
| `min_overlap_fraction` / `min_valid_blocks` / `min_block_std` / `min_block_valid_fraction` | 纹理更易判无效 → FAR↓ FRR↑ |
| `patch.min_overlap_ratio` | 丢弃更多边界点 → FAR↓ FRR↑（须重建模板） |

### 3.3 候选与几何容差（更松 → FAR↑ FRR↓）

| 参数 | 放宽时（增大阈值、放宽策略） |
| --- | --- |
| `matching.ratio_threshold`（及 Hamming 下 `hamming.ratio_threshold`） | FAR↑ FRR↓ |
| `abs_distance_threshold` / `distance_margin` | FAR↑ FRR↓（在对应策略生效时） |
| `candidate_policy`：`ratio_only` → `topk_or_ratio` / `topk_only` | 常 FRR↓、FAR↑ |
| `top_k`（仅 top-k 类策略） | FAR↑ FRR↓ |
| `bidirectional_ratio_test: true` → `false` | 放宽 → FAR↑ FRR↓ |
| `ransac_reproj_threshold` | FAR↑ FRR↓ |
| `min_scale`↓ / `max_scale`↑（尺度窗变宽） | FAR↑ FRR↓ |
| `max_candidates_for_ransac` | 过大易噪声 → FAR 风险升；过小可能伤本人 |

### 3.4 注册规模（在 `max` 融合下常呈弱取舍）

| 参数 | 常见现象 |
| --- | --- |
| `enrollment.enrollment_images_per_identity` 增大 | FRR 常降；非本人也更易碰到某一张高分模板 → FAR 可能升。看重标定后的工作点，不要只看固定阈值 |

---

## 4. 按调参优先级整理（怎么动手）

建议：**一次只动一组**，每组结束后用完整阈值曲线重标定，再比 `FRR @ 目标 FAR`。

### 第 1 级：工作点（只取舍，不改善模型）

| 参数 | 当前常见值 | 说明 |
| --- | ---: | --- |
| `identification.match_score_threshold` | `0.80`（以配置为准） | 先跑曲线，再选点；不要手工盲猜 |
| `evaluation.auto_thresholds` / `score_threshold_step` | 扫描用 | 不改变匹配分数，只影响曲线密度 |

### 第 2 级：融合与注册覆盖

| 参数 | 当前常见值 | 类型 |
| --- | ---: | --- |
| `identification.fusion_method` | `max` | 取舍（`max` vs `top3_mean`） |
| `enrollment.enrollment_images_per_identity` | `40` | 弱取舍 / 覆盖；须留至少 1 张 query |
| `enrollment.selection_strategy` | `first` | `first` 取前 N 张；`stride` 等步长取样；`random` 用 `random_seed` 随机抽取 |
| `enrollment.random_seed` | `42` | 注册选取不再随机；仍可用于 impostor 抽样等 |

### 第 3 级：纹理融合（最有希望改善分离的一组后端参数）

| 参数 | 当前常见值 | 类型 | 首轮尝试 |
| --- | ---: | --- | --- |
| `texture_verification.enabled` | `true` | 可能改善分离 | `false` / `true` 全量对照 |
| `texture_weight` / `geometry_weight` | `0.30` / `0.70` | 多为取舍，甜区可能改善 | 纹理占比 `0、0.20、0.30、0.40` |
| `geometry_saturation_inliers` | `12` | 取舍 | `10、12、14、16` |
| `low_unique_inliers` | `3` | 取舍 | `2、3、4` |
| `min_overlap_fraction` 等块阈值 | 见 tuning | 取舍 | 小步微调 |

开启纹理融合后模板须含 `overlap_image`；旧模板不兼容时须重建。

### 第 4 级：描述子候选

L2 与 Hamming **阈值不能混用**；二值时以 `matching.hamming.*` 为准。

| 参数 | 当前常见值 | 类型 | 首轮尝试 |
| --- | ---: | --- | --- |
| `ratio_threshold` | L2 `0.85`；Hamming `0.90` | 取舍；甜区或可改善分离 | L2：`0.82~0.90`；Hamming：`0.88~0.92` |
| `bidirectional_ratio_test` | `false`（tuning） | 取舍 | `true` / `false` |
| `candidate_policy` | `ratio_only` | 取舍 | 对比 `topk_or_ratio` |
| `top_k` | `1` | `ratio_only` 下 >1 基本无效 | 仅 top-k 策略时试 `1~5` |
| `abs_distance_threshold` | L2 `1.5`；Hamming `0.50` | 取舍 | 小步扫描 |
| `distance_margin` | L2 `0.15`；Hamming `0.05` | 主要影响 top-k 分支 | 同上 |
| `allow_many_to_one_before_ransac` | `true`（tuning） | 不定 | 与双向 / top-k 联调 |

耦合：`ratio_only` 时 `top_k`、`distance_margin` 几乎不起作用；放宽一项不够时往往要看整组候选是否仍被另一门槛卡住。

### 第 5 级：RANSAC 与几何

| 参数 | 当前常见值 | 类型 | 首轮尝试 |
| --- | ---: | --- | --- |
| `ransac_reproj_threshold` | `0.5` | 取舍 | `0.4、0.5、0.75、1.0` |
| `min_scale` / `max_scale` | `0.80` / `1.20` | 取舍 | 收紧/放宽对照 |
| `reproj_error_weight` | `0.05` | 不定 / 弱 | `0、0.05、0.10` |
| `max_candidates_for_ransac` | `250` | 弱取舍 + 耗时 | `200~400` |
| `ransac_max_iters` / `ransac_confidence` | `3000` / `0.995` | 饱和后影响小 | 失败率异常再动 |

方向一致性相关配置仅诊断，**不进分数**。

### 第 6 级：关键点与 patch（须重建模板）

与训练预处理强耦合，**不要**和后端参数同一轮大范围乱改。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `sift.*`、`keypoint_filter.max_keypoints` | 不定；可能改善分离也可能两边变差 | 优先小步，并与训练设置对照 |
| `patch.min_overlap_ratio` | 取舍 | 更严 → FAR↓ FRR↑ |
| `patch.crop_size` / `out_size` / `normalize` | **契约，不调** | 必须与训练一致 |
| `matching.distance`、描述子维度与存储 | **契约，不调** | 跟随 checkpoint / Hadamard |

---

## 5. 不影响离线 FAR/FRR（或只影响统计可信度）

| 参数 | 结论 |
| --- | --- |
| `runtime.max_impostor_identities_per_query` | 正式实验必须为 `-1`；`0` 不产 FAR；正整数抽样会降低 FAR 可信度 |
| `runtime.limit_identities` / `limit_images_per_identity` | 仅调试；正式为 `0` |
| `identification.early_stop_*` | 主要影响耗时；开启后曲线上的 identity 分可能低于完整遍历最大值 |
| `template_management.*` | 离线标定不更新模板库，不影响本次 FAR/FRR |
| 推理 batch、半精度、`hamming.backend` | 速度为主 |
| `evaluation.failure_export.*`、`output.*` | 不改匹配分数 |

`data.image_root`、划分方式属于**评估协议**，对比实验必须固定。

---

## 6. 推荐实验顺序

1. **固定模型与数据，跑一条完整基线曲线**  
   记录：`FRR @ FAR=2e-5`、`FRR @ FAR=1e-4`、EER、AUC、错误接受/拒绝绝对数、分数分位数、各阶段失败原因。
2. **纹理融合消融**（最可能改善分离）  
   开关 + 权重 + `geometry_saturation_inliers` + `low_unique_inliers`。
3. **候选质量**  
   ratio、双向、`candidate_policy`；追求“干净”而不是“最多”。
4. **几何容差**  
   RANSAC 阈值、尺度窗。
5. **注册规模与关键点**（后者须重建模板）  
   模板数、关键点上限、SIFT 对比度等。
6. **最后才用 `match_score_threshold` 选产品工作点**  
   在已选参数组合的曲线上取点，而不是反过来用阈值掩盖后端问题。

每个实验至少报告：

- `FRR @ 目标 FAR`（主指标）
- EER、AUC
- 错误接受 / 错误拒绝绝对数量（小样本 FAR 极敏感）
- genuine / impostor 分数分位数
- 候选数、RANSAC 成功率、唯一内点、纹理可用率
- 单次匹配耗时（若关心解锁体验）

---

## 7. 一句话对照表

| 你想做的事 | 优先动什么 | 别指望什么 |
| --- | --- | --- |
| 只换产品松紧（FAR↔FRR） | `match_score_threshold`、融合方式 | 自动两边都降 |
| 尽量两边都降 | 更好模型、纹理融合甜区、更干净的候选 | 只拧最终阈值 |
| 湿手指特化 | 纹理权重、ratio、内点门槛的对照 | 直接照搬正常手指阈值 |
| 换浮点 / Hadamard | checkpoint + Hadamard 开关，重建模板 | 复用旧距离阈值与旧模板 |

---

## 8. 证据与限制

- 实现与默认值以 `config_match_new.yaml`、`config_match_tuning.yaml` 及匹配代码为准；本文“当前常见值”会随 tuning 文件变化。
- 历史某次 `normal_pic` + strongv2 实验曾表明：仅滑动最终阈值无法同时满足极低 FAR 与很低 FRR；要用后端拉开分离度。
- 训练 patch 级 EER 与身份级 FAR/FRR 口径不同，不能互相替代。
- impostor 样本少时，1 次错误接受就会让 FAR 跳变，结论需更大库复核。
- 改 SIFT / patch / 模型 / Hadamard 后必须重建模板；只改匹配、RANSAC、纹理融合、最终阈值时，在模板契约兼容前提下可 `--skip-template-build`。
