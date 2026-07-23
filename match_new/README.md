# match_new 命令说明

在项目根目录执行，例如：

```powershell
cd D:\新建文件夹\research\指纹识别\code\myhardnet
conda activate hardnet-cuda
```

默认配置文件：`match_new/config_match_new.yaml`

---

## 1. `run_hardnet_matching.py`

**作用**：HardNet 主实验。只检测 SIFT 关键点位置/方向，计算 HardNet 描述子并匹配；**不计算、不保存 SIFT/RootSIFT 描述子**。模板还会无损保存原始 `uint8` 灰度图，供仿射对齐后的脊线纹理验证使用。

**运行参数**默认写在 `match_new/config_match_new.yaml`：

| 配置项 | 作用 |
|--------|------|
| `output.output_dir` | 实验结果输出目录 |
| `runtime.skip_template_build` | 是否复用已有模板 |
| `runtime.max_impostor_identities_per_query` | 每个 query 最多测几个非本人身份，`0`=全量 |
| `runtime.limit_identities` / `limit_images_per_identity` | 调试裁剪，正式实验保持 `0` |
| `data.image_root` | 原始指纹图像根目录 |
| `data.identity_depth` | 组成一个手指 identity 的目录层级数 |
| `model.checkpoint` | HardNet 权重 |
| `enrollment.random_seed` | 注册/query 划分种子 |
| `texture_verification.*` | 局部脊线纹理二次筛选及灰区提升参数 |
| `template_management.*` | 在线模板学习、LRU排序和固定容量替换 |
| `evaluation.target_far` / `target_frr` | 目标 FAR/FRR |
| `evaluation.failure_export.*` | 失败样本导出 |

优先级：`命令行覆盖 > 配置文件 > 程序默认值`。

**推荐命令**（全部走配置）：

```powershell
python match_new\run_hardnet_matching.py
```

也可以直接覆盖原图目录及身份目录层级，无需预先运行 `pair_build`：

```powershell
python match_new\run_hardnet_matching.py `
  --image-root D:\datasets\tiny_v12 `
  --identity-depth 2
```

**临时覆盖示例**：

```powershell
python match_new\run_hardnet_matching.py `
  --output_dir match_new\butieping `
  --skip-template-build
```

**调试**（也可直接改配置里的 `runtime.limit_*`）：

```powershell
python match_new\run_hardnet_matching.py `
  --limit_identities 2 `
  --limit_images_per_identity 22
```

**输出模板目录**：`<output_dir>/image_templates/`，每个 `.npz` 包含关键点字段、`hardnet_descriptors` 和 `overlap_image`。灰度图保持原始尺寸和关键点坐标系，由 `np.savez_compressed` 无损压缩，`np.load` 时自动解压。

**常用命令行覆盖**（均可不传，改用配置）：

| 参数 | 作用 |
|------|------|
| `--config` | 配置文件路径，默认 `match_new/config_match_new.yaml` |
| `--image-root` | 覆盖 `data.image_root` |
| `--identity-depth` | 覆盖 `data.identity_depth` |
| `--model_path` | 覆盖 `model.checkpoint` |
| `--output_dir` | 覆盖 `output.output_dir` |
| `--skip-template-build` / `--no-skip-template-build` | 覆盖是否跳过模板构建 |
| `--max_impostor_identities_per_query` | 覆盖 impostor 上限 |
| `--random_seed` | 覆盖注册随机种子 |
| `--target_far` / `--target_frr` | 覆盖目标 FAR/FRR |
| `--failure-export` / `--no-failure-export` | 覆盖是否导出失败样本 |
| `--max_failure_cases_per_type` | 覆盖每类失败样本导出上限 |
| `--limit_identities` | 调试：限制 identity 数量 |
| `--limit_images_per_identity` | 调试：限制每个 identity 的图像数量 |

**解锁提前停止**：

`config_match_new.yaml` 中默认启用：

```yaml
identification:
  match_score_threshold: 0.55
  early_stop_on_unlock_threshold: true
  early_stop_threshold:
```

含义：对某个 query 匹配某个 identity 的注册模板库时，只要某张注册模板的图像级连续匹配分数 `score >= early_stop_threshold`，就立即返回，不再继续跑剩余模板。`early_stop_threshold` 留空时使用 `match_score_threshold`。这里的阈值范围是 `[0,1]`，不再表示内点数量。

输出 CSV 中：

| 字段 | 含义 |
|------|------|
| `num_templates` | 该 identity 的注册模板总数 |
| `num_templates_evaluated` | 本次实际匹配了几张模板 |
| `early_stopped` | 是否因为达到阈值提前停止 |
| `early_stop_threshold` | 本次使用的提前停止阈值 |

该策略适合实际手机解锁，但不能直接用于完整阈值扫描：在 `0.55` 提前停止后，无法知道剩余模板是否能达到更高分数。正式 FAR/FRR 实验默认使用：

```yaml
evaluation:
  disable_early_stop_for_threshold_sweep: true
```

评估时会临时关闭提前停止并计算完整最高分；`identification.early_stop_on_unlock_threshold` 仍保留给实际在线解锁。此时评估生成的 `unlock_timing` 是全模板匹配耗时，不代表线上提前停止耗时。

**换 checkpoint 示例**：

```powershell
python match_new\run_hardnet_matching.py `
  --model_path outputs\hardnet_train_xxx\best.pt `
  --output_dir match_new\outputs_xxx
```

**复用已有模板**（只改了匹配参数、未改模型/patch/SIFT 检测参数和纹理模板格式时）：

```powershell
python match_new\run_hardnet_matching.py `
  --output_dir match_new\outputs `
  --skip_template_build
```

注意：旧版双描述子模板（同时含 HardNet 和 RootSIFT）**不再支持复用**，需要重新生成 HardNet 单描述子模板。

启用 `texture_verification.enabled: true` 后，模板还必须包含 `overlap_image`。旧 HardNet 模板没有该字段，不能使用 `--skip-template-build`，需要重新生成一次模板。

### 几何与脊线纹理融合（C 方案）

纹理分数依赖 RANSAC 得到的 query→template 仿射矩阵，因此仍保留一个仅用于保证仿射可靠性的低内点门槛。达到门槛后，不再用纹理做硬通过/拒绝，而是计算连续融合分数：

```math
geometry\_similarity = clip(unique\_inliers / 12, 0, 1)
```

```math
score = 0.70 \times geometry\_similarity + 0.30 \times texture\_similarity
```

```text
unique_inliers < low_unique_inliers
    -> score = 0，不计算纹理

unique_inliers >= low_unique_inliers
    -> 计算 geometry_similarity 和 texture_similarity
    -> 加权得到 [0,1] match score
```

纹理分数使用分块 ZNCC：先将 query 灰度图仿射变换到模板坐标系，只在有效重叠区域计算；低对比块被排除，最终取有效块正相关系数的中位数。相关配置位于：

```yaml
texture_verification:
  enabled: true
  low_unique_inliers: 3
  geometry_saturation_inliers: 12.0
  geometry_weight: 0.70
  texture_weight: 0.30

identification:
  match_score_threshold: 0.55
```

`unique_inliers` 继续作为诊断字段输出，但不再直接充当 `score` 或解锁阈值。以上权重和 `0.55` 阈值只是首轮实验值，必须通过独立验证集按目标 FAR/FRR 重新标定。`verification_scores.csv` 会额外输出 `geometry_similarity`、`texture_similarity`、有效重叠比例、有效块数和实际融合权重。

FAR/FRR 阈值曲线写入 `match_score_threshold_curve.csv`。默认按 `0.01` 在 `[0,1]` 范围扫描，不再生成整数内点阈值曲线。

`metrics.json` 的 `texture_verification_config` 会记录实际融合参数，`score_component_summary` 会分别汇总 genuine/impostor 中纹理参与次数，以及相对纯几何分数改变了多少次阈值判定。`matching_backend` 名称带有 `texture_fusion` 时，表示本次结果确实启用了 C 方案。

### 动态模板学习、替换与LRU排序

动态模板库由以下模块组成：

| 模块 | 职责 |
|------|------|
| `template_learning.py` | 身份可信确认、真实纹理公共区域和模板内容去重 |
| `template_ranking.py` | 成功模板移到首位、新学习模板插入首位 |
| `template_replacement.py` | 满载时从LRU末尾选择非保护模板 |
| `template_library.py` | 模板文件复制、索引持久化和完整在线更新流程 |

当前配置为20张初始模板、最多40张活动模板。20张初始模板全部是受保护seed，永远不会被自动替换：

```yaml
template_management:
  enabled: true
  apply_during_evaluation: true
  reset_library_on_start: true

  max_active_templates: 40
  protected_seed_templates: 20

  learn_score_threshold: 0.85
  learn_min_unique_inliers: 12
  learn_min_texture_similarity: 0.75
  confirm_score_threshold: 0.70
  confirmation_templates: 2
  require_seed_confirmation: true

  min_common_area_ratio: 0.35
  min_common_area_pixels: 256

  persist_replace_retries: 20
  persist_retry_delay_ms: 100
  persist_strict: false
```

一次query只有同时满足以下条件才会写入模板库：

```text
线上LRU匹配达到解锁阈值
AND 至少一张模板满足严格学习阈值
AND 至少两张可信模板确认
AND 确认模板中至少有一张初始seed
AND 仿射对齐后的有效脊线公共区域达标
AND 与活动模板不存在完全相同的特征内容
```

这里不再设置按压位置、旋转角度和新增覆盖量阈值。高置信度且具有足够公共区域的非重复query会直接学习。

LRU规则：

```text
成功命中的现有模板 -> 移到第一个
匹配失败的模板       -> 不改变顺序
新学习模板           -> 插入第一个
达到40张             -> 删除末尾第一个非保护学习模板
```

评估中的静态FAR/FRR分数仍先按完整模板库计算。每个query的静态记录完成后，再使用线上提前停止配置对本人模板库执行动态更新；本次更新只影响后续query。学习确认耗时不计入前台解锁耗时。

主要输出：

| 输出 | 含义 |
|------|------|
| `template_library.json` | 当前活动模板、保护状态和持久化LRU顺序 |
| `learned_templates/` | 已加入模板库的query NPZ副本 |
| `retired_templates/` | 被LRU替换的学习模板，便于实验回滚 |
| `eval_hardnet_l2/template_learning_events.csv` | 每次query的学习、替换和耗时摘要 |
| `eval_hardnet_l2/template_learning_events.json` | 包含每张确认模板详细证据的完整事件 |

`reset_library_on_start: true` 适合可重复实验，每次从相同20张seed开始。长期在线运行时改为 `false`，程序会继续读取已有 `template_library.json`。

Windows 下杀毒软件、文件索引器或同步程序可能短暂占用 `template_library.json`，使原子替换返回 `WinError 5`。程序会使用唯一临时文件并按退避间隔重试；默认重试后仍无法替换时，不中断整批评估，而是把最新完整状态保存到 `template_library.json.pending`。后续写盘会再次尝试更新主索引；使用 `reset_library_on_start: false` 启动时，也会在主索引、`.pending` 和完整临时文件中加载修改时间最新且结构有效的状态。若工程部署要求索引写盘失败必须立即终止，可设置 `persist_strict: true`。

---

## 2. `ablation.py`

**作用**：在已有模板基础上，扫描不同匹配参数组合（`top_k`、`ratio_threshold`、`candidate_policy`），快速比较调参效果。不重新提特征。

**前置条件**：先运行 `run_hardnet_matching.py` 生成 `image_templates/`、`metadata_with_split_20.csv`、`identity_templates_20.json`。

**命令**：

```powershell
python match_new\ablation.py `
  --base_output_dir match_new\outputs `
  --output_dir match_new\outputs_ablation `
  --topk_values 1 2 5 10 `
  --ratio_thresholds 0.85 0.90 0.95 0.98
```

**常用参数**：

| 参数 | 作用 |
|------|------|
| `--base_output_dir` | 已有主实验输出目录 |
| `--output_dir` | 消融结果输出目录 |
| `--topk_values` | 要扫描的 top-k 列表 |
| `--ratio_thresholds` | 要扫描的 ratio 阈值列表 |
| `--candidate_policies` | 候选策略：`ratio_only` / `topk_only` / `topk_or_ratio` |
| `--max_impostor_identities_per_query` | 每个 query 最多测试几个非本人 identity |

---

## 推荐执行顺序

**HardNet 验证**：

```text
run_hardnet_matching.py
```

**只调匹配参数**：

```text
run_hardnet_matching.py          # 首次构建模板
  -> ablation.py                 # 复用模板扫参数
```

---

## 3. `visualize_hardnet_inliers.py`

**作用**：单独查看 HardNet 的内点匹配可视化。脚本从一个原始图像目录扫描指纹图像，构建 HardNet 模板，然后按 `run_hardnet_matching.py` / FAR-FRR 评估中相同的 genuine 匹配逻辑，为每个 query 匹配本人注册模板库中的最佳模板，最后导出：

- `unique_inliers` 最高的 top N，默认 10
- `unique_inliers` 最低且不低于阈值的 bottom N，默认 10，最低阈值默认 3

匹配参数与 `config_match_new.yaml` 共用，脚本内部复用 `match_templates_descriptor_l2(..., descriptor_source="hardnet", include_debug=True)`，因此 Lowe ratio、top-k、RANSAC、unique inlier 去重等逻辑和正式 HardNet 评估一致。

双向 Lowe 验证由以下配置控制：

```yaml
matching:
  ratio_threshold: 0.85
  bidirectional_ratio_test: true
```

启用后，候选必须在 `query -> gallery` 和 `gallery -> query` 两个方向都通过同一个 Lowe 比值阈值，并且双方互为最近邻。关闭后恢复原来的单向比值验证。

**推荐命令**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals
```

**复用已构建模板**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals `
  --skip_template_build
```

**如果每个 identity 注册 20 张导致 query 太少，可以调小注册数**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals_enroll10 `
  --enrollment_count 10
```

**常用参数**：

| 参数 | 作用 |
|------|------|
| `--config` | 配置文件路径，默认 `match_new/config_match_new.yaml` |
| `--image_root` | 输入原始图像目录，默认 `pair_build/select_top500` |
| `--output_dir` | 可视化输出目录，默认 `match_new/hardnet_inlier_visuals` |
| `--model_path` | 覆盖配置中的 HardNet checkpoint |
| `--skip_template_build` | 复用 `<output_dir>/image_templates` 和 `metadata_success.csv` |
| `--enrollment_count` | 覆盖每个 identity 的注册模板数量，默认读取 config |
| `--random_seed` | 覆盖注册模板随机种子，默认读取 config |
| `--top_k` | 导出内点数最高的 case 数量，默认 10 |
| `--bottom_k` | 导出内点数最低的 case 数量，默认 10 |
| `--bottom_min_unique_inliers` | bottom case 的最低 unique 内点数阈值，默认 3 |
| `--identity_depth` | 输入目录下几层路径组成 identity，默认 1 |
| `--max_lines` | 每张连线图最多画多少条线，默认 200 |
| `--include_impostor` | 额外计算 query 对非本人 identity 的匹配，默认关闭 |

**主要输出**：

```text
match_new/hardnet_inlier_visuals/
  image_templates/
  metadata_all.csv
  metadata_success.csv
  metadata_with_split_20.csv
  identity_templates_20.json
  top_unique_inliers/
  bottom_unique_inliers_nonzero/
  selected_cases.csv
  summary.json
```

每个 case 目录包含：

| 文件 | 含义 |
|------|------|
| `match_lines_unique.png` | HardNet unique inliers 左右连线图 |
| `match_lines_raw.png` | RANSAC raw inliers 左右连线图 |
| `query_warped_to_gallery.png` | 按 affine matrix 把 query warp 到 gallery 坐标系后的图 |
| `overlap_color.png` | warp 后 query 和 gallery 的彩色重叠图 |
| `overlap_and.png` | 两张图二值化后重叠区域的 AND |
| `overlap_xor.png` | 两张图二值化后差异区域的 XOR |
| `match_result.json` | 匹配指标、affine matrix、输出路径和 debug 计数 |
