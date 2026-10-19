# match_new 命令说明

在项目根目录执行，例如：

```powershell
cd D:\新建文件夹\research\指纹识别\code\myhardnet
conda activate hardnet-cuda
```

默认配置文件：`match_new/config_match_new.yaml`

常调实验参数集中在 `match_new/config_match_tuning.yaml`；主配置通过
`extends` 自动继承，运行命令和 `--config` 路径无需修改。

---

## 1. `run_hardnet_matching.py`

**作用**：HardNet 全量离线阈值标定实验。只检测 SIFT 关键点位置/方向，计算 HardNet 描述子并匹配；支持 float32/L2 与 packed binary/Hamming 两种 checkpoint 契约，后续候选生成、RANSAC、纹理融合和 identity 级评估共用。所有未注册 query 都会遍历本人和全部非本人 identity 的全部注册模板，不再拆分 validation/test；阈值曲线和最终 FAR/FRR 均基于同一份全量匹配结果。

**运行参数**默认写在 `match_new/config_match_new.yaml`：

| 配置项 | 作用 |
|--------|------|
| `output.output_dir` | 实验结果输出目录 |
| `runtime.skip_template_build` | 是否复用已有模板 |
| `runtime.max_impostor_identities_per_query` | 每个 query 最多测几个非本人身份：`-1`=全量，`0`=不比对，正整数=抽样上限 |
| `runtime.limit_identities` / `limit_images_per_identity` | 调试裁剪，正式实验保持 `0` |
| `data.image_root` | 原始指纹根目录；固定为 `image_root/<手指目录>/图像文件` |
| `model.checkpoint` | 浮点或二值 HardNet 权重 |
| `model.descriptor_kind` / `model.binary_storage` / `model.binary_bitorder` | 描述子类型与二值模板存储契约，默认 `auto`/`packed_uint8`/`auto` |
| `model.hadamard_binarization.*` | 启用时把浮点 checkpoint 的描述子按 `Hf -> sign -> packed_uint8` 二值化；不使用 P、D 或额外 state 文件 |
| `matching.distance` | `auto` 跟随 checkpoint，也可显式写 `l2` 或 `hamming` |
| `matching.hamming.backend` | packed-Hamming 候选后端；默认 `cpu` 使用 OpenCV SIMD/POPCNT，`cuda` 仅用于显式实验 |
| `matching.hamming.*` | 二值 Hamming 的 ratio、绝对距离和自适应 margin，需按验证集重新标定 |
| `enrollment.random_seed` | 注册策略为 `random` 时控制抽样；其它用途仍可复用 |
| `enrollment.selection_strategy` | `first`（默认，取前 N 张）、`stride`（等步长取样）或 `random`（随机抽样） |
| `texture_verification.*` | 局部脊线纹理二次筛选及灰区提升参数 |
| `template_management.*` | 离线评估中的模板学习、LRU排序和固定容量替换 |

优先级：`命令行覆盖 > 配置文件 > 程序默认值`。

**推荐命令**（全部走配置）：

```powershell
python match_new\run_hardnet_matching.py
```

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\new_data_V3 `
  --output-dir outputs\newdataV3-8-float256+模板数30 `
```

**输出模板目录**：`<output_dir>/image_templates/`，每个 `.npz` 包含关键点字段、`hardnet_descriptors` 和 `overlap_image`。浮点模板使用 `float32` 列；二值模板使用 `packed_uint8` 列，`hardnet_descriptor_dim` 表示 bit 数，`hardnet_descriptor_bitorder` 保存位序。Hadamard 后处理还会保存 `hardnet_descriptor_transform_name` 和 `hardnet_descriptor_transform_id`，防止旧的 P/D 模板与当前直接 `Hf` 模板混用。灰度图保持原始尺寸和关键点坐标系，由 `np.savez_compressed` 无损压缩，`np.load` 时自动解压。

主要输出结构：

```text
<output_dir>/
  metadata_all.csv
  image_templates/
  identity_templates_<N>.json
  template_build_timings.csv
  eval_hardnet_<l2|hamming>/
    verification_scores.csv
    metrics.json
    effective_config.yaml
    match_score_threshold_curve.csv
    far_frr_thresholds_<min>_<max>.csv
    per_finger_far_frr_at_global_zero_far.csv
    unlock_timing.json
    plots/
      score_distribution.png
      far_frr_vs_threshold.png
    failure_cases/
```

`template_build_timings.csv` 只有一行，统计全部成功模板在读图、SIFT 检测、
关键点过滤、patch 裁切旋转、HardNet 推理、模板组装和保存等阶段的平均耗时，
不再输出逐图耗时；使用 `--skip-template-build` 时本次没有模板构建计时，因此
不生成该文件。离线评估只保留 `unlock_timing.json` 总体解锁耗时汇总，
不再生成逐次 `unlock_timing.csv`。`far_frr_vs_threshold.png` 会直接标出 EER
阈值/EER，以及配置匹配阈值对应的 FAR、FRR。

**常用命令行覆盖**（均可不传，改用配置）：

| 参数 | 作用 |
|------|------|
| `--config` | 配置文件路径，默认 `match_new/config_match_new.yaml` |
| `--image-root` | 覆盖 `data.image_root` |
| `--image-root` | 覆盖 `data.image_root` |
| `--model-path` / `--model_path` | 覆盖 `model.checkpoint` |
| `--output-dir` / `--output_dir` | 覆盖 `output.output_dir` |
| `--hadamard-binarization` / `--no-hadamard-binarization` | 开启/关闭 `model.hadamard_binarization.enabled` |
| `--ratio-threshold` | 覆盖浮点/L2 的 `matching.ratio_threshold` |
| `--hamming-ratio-threshold` | 覆盖二值/Hamming 的 `matching.hamming.ratio_threshold` |
| `--skip-template-build` / `--no-skip-template-build` | 覆盖是否跳过模板构建 |
| `--max_impostor_identities_per_query` | 覆盖 impostor 上限 |
| `--random_seed` | 覆盖注册随机种子 |
| `--failure-export` / `--no-failure-export` | 覆盖是否导出失败样本 |
| `--max_failure_cases_per_type` | 覆盖每类失败样本导出上限 |
| `--limit_identities` | 调试：限制 identity 数量 |
| `--limit_images_per_identity` | 调试：限制每个 identity 的图像数量 |

浮点/L2 实验示例：

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\wet_select `
  --model-path outputs\models\3\m0\m0_256_v2\best.pt `
  --output-dir outputs\wet_select_float `
  --no-hadamard-binarization `
  --ratio-threshold 0.85
```

Hadamard 二值/Hamming 实验示例：

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\wet_select `
  --model-path outputs\models\3\m0\m0_256_v2\best.pt `
  --output-dir outputs\wet_select_hadamard `
  --hadamard-binarization `
  --hamming-ratio-threshold 0.90
```

`matching.hamming.*` 会覆盖同名顶层参数，所以二值/Hamming 实验应使用
`--hamming-ratio-threshold`。切换 checkpoint 或 Hadamard 开关会改变描述子契约，
必须重新构建模板，不能同时使用 `--skip-template-build`；只修改 ratio 阈值时可以复用模板。

**离线标定与早停**：

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

该策略由 `identification.early_stop_on_unlock_threshold` 控制离线 identity 级匹配。开启后，`run_hardnet_matching.py` 在本人模板分数达到阈值时可提前停止；关闭后遍历全部注册模板。`metrics.json` 中的 `offline_full_template_scoring` 会标记本次评估是否使用完整模板遍历。

开启早停可降低本人匹配耗时；若需要完整的 identity 最大分数用于阈值曲线、AUC 或 EER，请将 `early_stop_on_unlock_threshold` 设为 `false`。

本目录仅保留离线模板构建、匹配评估和推理后端测速入口；单次在线解锁与在线延迟基准代码已移除。

**换 checkpoint 示例**：

```powershell
python match_new\run_hardnet_matching.py `
  --model-path outputs\hardnet_train_xxx\best.pt `
  --output-dir match_new\outputs_xxx
```

**复用已有模板**（只改了匹配参数、未改模型/patch/SIFT 检测参数和纹理模板格式时）：

```powershell
python match_new\run_hardnet_matching.py `
  --output-dir match_new\outputs `
  --skip-template-build
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

`unique_inliers` 继续作为诊断字段输出，但不再直接充当 `score` 或解锁阈值。离线评估会直接使用全部 query 的 identity 级分数扫描 FAR/FRR，并在 `identification.match_score_threshold` 下报告最终结果。`verification_scores.csv` 会额外输出 `geometry_similarity`、`texture_similarity`、有效重叠比例、有效块数和实际融合权重。

FAR/FRR 阈值曲线写入 `match_score_threshold_curve.csv`。默认按 `0.01` 在 `[0,1]` 范围扫描，不再生成整数内点阈值曲线。

固定阈值区间的 FAR/FRR 对照表由 `evaluation.far_frr_table` 控制，默认
`0.50--0.85`（步长 `0.01`），写入 `far_frr_thresholds_0.50_0.85.csv`。此外，
`per_finger_far_frr_at_global_zero_far.csv` 使用所有手指共同的
“全局 FAR=0 且 FRR 最小”阈值：第一行是全体统计，后续各行按
`owner_identity` 汇总单个注册手指的 FAR/FRR 及接受、拒绝计数。
只有一个 identity、没有 impostor 尝试时，表中会将选择状态标记为
`unavailable_no_impostor_attempts`，此时 FAR 不可估计。

`metrics.json` 的 `texture_verification_config` 会记录实际融合参数，`score_component_summary` 会分别汇总 genuine/impostor 中纹理参与次数，以及相对纯几何分数改变了多少次阈值判定。`matching_backend` 名称带有 `texture_fusion` 时，表示本次结果确实启用了 C 方案。

### 离线动态模板学习、替换与 LRU 排序

离线评估中每次运行都会基于本次注册模板构建 `identity_templates_40.json`，不会继承上一次运行的学习结果。

动态模板学习复用以下模块：

| 模块 | 职责 |
|------|------|
| `template_learning.py` | 计算仿射对齐后的公共纹理区域率，并判断学习准入 |
| `template_ranking.py` | 成功命中的模板移到首位，新模板插入首位 |
| `template_replacement.py` | 模板库满载时从末尾选择非保护模板 |
| `template_library.py` | 更新活动模板列表并原子写回 `identity_templates_40.json` |

默认配置：

```yaml
template_management:
  enabled: true

  # 活动模板库最大数量；初始受保护模板数量由 enrollment.enrollment_images_per_identity 决定。
  max_active_templates: 40

  # 本人匹配分数必须达到该值。
  learn_score_threshold: 0.85
  # 仿射对齐后的公共纹理区域率必须不高于该值，可按实验调节。
  max_common_area_ratio: 0.50

  coverage_window_size: 9
  coverage_min_std: 5.0
  coverage_morph_kernel: 3
```

每个 query 的处理顺序固定为：

```text
1. 使用当前活动模板库完成本人匹配
2. 本人分数达到 match_score_threshold 后，检查模板学习条件
3. 满足 learn_score_threshold 且公共纹理区域率 <= max_common_area_ratio 时加入模板库
4. 学习结果写回 identity_templates_40.json，仅影响后续 query
5. 使用更新后的模板库执行非本人匹配
```

学习不要求多个模板确认，也不要求初始模板确认；仍会拒绝完全重复的模板内容。初始注册模板数量由 `enrollment_images_per_identity` 决定，并全部受保护。`max_active_templates` 是活动模板库上限；当前两者都为 40 时，模板库已经满载，不能再加入学习模板。如需在保留 40 张初始模板的同时学习新模板，应将 `max_active_templates` 设置为大于 40 的值，例如 60。模板库满载后，才会从未受保护的学习模板中按 LRU 规则替换。写回 JSON 的重试次数和等待时间固定在代码内部，分别为 2 次和 1 毫秒，不放在 YAML 配置中。

主要输出仍保留原有评估文件，模板库只有一个动态索引：

| 输出 | 含义 |
|------|------|
| `identity_templates_40.json` | 当前活动模板路径、模板来源、保护状态、LRU顺序和更新计数 |
| `eval_hardnet_<metric>/metrics.json` | FAR、FRR及模板学习统计 |
| `eval_hardnet_<metric>/verification_scores.csv` | 每次本人/非本人验证分数 |

`metrics.json` 中的 `template_management` 会记录处理 query 数、通过本人匹配数、学习模板数和替换模板数。模板学习更新发生在当前 query 的本人结果记录之后，因此新模板不会改变当前 query 已经写入的本人分数；它只参与后续 query。

---

## 3. `ablation.py`

**作用**：在已有模板基础上，扫描不同匹配参数组合（`top_k`、`ratio_threshold`、`candidate_policy`），快速比较调参效果。不重新提特征。

**前置条件**：先运行 `run_hardnet_matching.py` 生成 `image_templates/`、`metadata_all.csv`、`identity_templates_20.json`；query 划分由注册模板索引在内存中推导。

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

## 4. `visualize_hardnet_inliers.py`

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
| `--skip_template_build` | 按当前原图 metadata 复用 `<output_dir>/image_templates` 中已存在的模板 |
| `--enrollment_count` | 覆盖每个 identity 的注册模板数量，默认读取 config |
| `--random_seed` | 覆盖注册模板随机种子，默认读取 config |
| `--top_k` | 导出内点数最高的 case 数量，默认 10 |
| `--bottom_k` | 导出内点数最低的 case 数量，默认 10 |
| `--bottom_min_unique_inliers` | bottom case 的最低 unique 内点数阈值，默认 3 |
| `--max_lines` | 每张连线图最多画多少条线，默认 200 |
| `--include_impostor` | 额外计算 query 对非本人 identity 的匹配，默认关闭 |

**主要输出**：

```text
match_new/hardnet_inlier_visuals/
  image_templates/
  metadata_all.csv
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

---

## 5. HardNet 推理时间优化与可选 CPU 后端

本次只优化 HardNet 描述子生成，不接入参考项目的前端预处理，也不合并预处理与匹配工程。
模型权重、模型架构、32×32 patch、SIFT、关键点上限、Hadamard 配置、位序、零值约定、
匹配阈值、RANSAC 和纹理验证参数保持不变。

### 5.1 后端和描述子边界

- 默认 `config_match_new.yaml` 使用 ONNX Runtime、FP32、NCHW、16 个主干线程和 Conv–BN 融合；PyTorch 仅用于加载权重和后处理。
- ONNX Runtime 只导出并执行 `forward_features(patches).q`。
  浮点模型的最终 L2 使用原来的 `torch.nn.functional.normalize(p=2, dim=1, eps=1e-12)`；
  Hadamard、sign、zero_is_one 和 packbits 继续调用原 Python 实现。
- 原生二值 checkpoint 只导出 `model.backbone.forward_features(...).q`，
  residual hash head、`logits >= 0` 和 `pack_binary_codes` 保留在 PyTorch/Python 中。
- ONNX Runtime 只支持 CPU、FP32、NCHW 和动态 batch；显式配置 CUDA 或低精度会报错。
- 初始化或推理失败不会静默回退；ONNX Runtime 只使用 CPUExecutionProvider。

Conv–BN 融合通过 `fuse_sequential_conv_bn_eval()` 对内存中的 eval 模型副本执行。
只融合 Sequential 内相邻 Conv2d/BatchNorm2d，不跨分支，也不融合 GroupNorm/LayerNorm。
磁盘 checkpoint 和训练模型不变。融合在独立实验配置中启用，默认关闭；浮点数学等价不代表二值 bit 必然一致。

### 5.2 后端配置与环境

所有后端参数统一写在 `config_match_new.yaml` 的 `model` 段，不再使用独立的 CPU 配置文件。
默认配置为 ONNX Runtime CPU，使用 ONNX Runtime 执行 CNN 主干，PyTorch 只负责加载权重和描述子后处理。

```yaml
model:
  inference_backend: onnxruntime
  device: cpu
  inference_precision: fp32
  channels_last: false
  cpu_num_threads: 1
  backend_num_threads: 28
  backend_inter_op_num_threads: 1
  backend_graph_optimization: all
  fuse_conv_bn: true
```

参考目录差异核对结果：主干后端、28 个 ONNX Runtime 线程、inter-op=1、CPU FP32、NCHW、
动态 batch、Conv–BN 融合、SIFT 500 和关键点上限 500 已对齐。当前实现额外保留了更严格的
raw-q 图契约、缓存校验、原子发布和后端输出校验，因此初始化代码不与参考目录逐行相同，
但这些额外检查不应计入稳态 `hardnet_inference_ms`。

修改 `model.inference_backend` 即可切换推理方式：

```yaml
# ONNX Runtime CPU（唯一支持的 CNN 推理方式）
model:
  inference_backend: onnxruntime
  device: cpu
  inference_precision: fp32
  fixed_inference_batch_size: 0
  cpu_num_threads: 1
  backend_num_threads: 16
  backend_inter_op_num_threads: 1
  backend_graph_optimization: all
  fuse_conv_bn: true
```

说明：

- `inference_backend` 固定为 `onnxruntime`，不再提供 PyTorch CNN 推理配置。
- PyTorch 仍用于加载模型权重、处理 L2/hash/Hadamard 和打包描述子；`cpu_num_threads` 控制这些后处理线程。
- `backend_num_threads`、`backend_inter_op_num_threads` 和 `backend_graph_optimization` 控制 ONNX Runtime。
- ONNX Runtime 固定使用 CPU、FP32、NCHW 和动态 batch；初始化或推理失败会直接报错，不会回退。
- `onnx_path` 为空时使用系统临时目录的安全缓存；`fuse_conv_bn` 默认开启。

运行时直接使用主配置：

```powershell
D:\Anaconda\envs\hardnet\python.exe -m match_new.run_hardnet_matching `
  --config match_new/config_match_new.yaml
```

如果只是小规模速度对照，benchmark 默认也读取 `config_match_new.yaml`，通过变体选择后端：

```powershell
D:\Anaconda\envs\hardnet\python.exe -m match_new.benchmark_cpu_inference `
  --config match_new/config_match_new.yaml `
  --variants onnxruntime_16 onnxruntime_28_fused `
  --limit-images 4 --warmup 2 --repeats 7
```

本次验证环境为 `D:\Anaconda\envs\hardnet\python.exe`，PyTorch 2.13.0+cpu、
ONNX 1.23.2、ONNX Runtime 1.23.2（含 CPUExecutionProvider）。系统默认
`D:\Anaconda\python.exe` 的 Torch DLL 被 Windows 应用控制策略阻止加载，
因此请使用 `hardnet` 环境运行。

安装依赖：

```powershell
D:\Anaconda\envs\hardnet\python.exe -m pip install onnx onnxruntime
```

### 5.3 安全导出缓存与初始化计时

ONNX 使用动态 batch 和常量折叠，仅在缓存失效时导出。默认缓存位于系统临时目录
`hardnet_onnx_cpu_cache`；`model.onnx_path` 可指定目标路径，但仍会验证对应元数据和文件校验和，
不能用来绕过当前 checkpoint/模型的契约。

缓存标识包含 checkpoint SHA256、有效推理模型权重和 buffer、架构/模块选项、源码签名、
融合状态、patch 尺寸、opset、PyTorch 版本和 raw-q 图协议版本。图损坏、manifest 不完整或
参数变化均会重导；发布使用独立临时文件与文件锁，失败保留旧缓存。

启动信息独立记录：

- `backend_export_ms`：实际 ONNX tracing/export 耗时；缓存命中时严格为 0。
- `backend_cache_prepare_ms`：模型复制、哈希、缓存验证、图校验及发布等准备时间，不含 tracing/export。
- `backend_cache_hit`：是否命中有效缓存。
- `backend_initialization_ms`：ONNX Runtime Session 初始化。
- `model_initialization_ms`、`model_warmup_ms`：模型初始化和预热。

离线构建保存 `inference_startup.json`。
上述时间均不计入单次 `hardnet_inference_ms`。首次初始化和首次某形状的推理可能更慢，
正式测速必须在同一常驻实例上预热真实形状。

### 5.4 分阶段推理时间

- CPU 在显式 `describe_with_timing()` 时记录分阶段耗时；正式 `describe()` 默认不插入阶段计时，避免同步开销。

`hardnet_inference_ms` 包含以下互不重叠的子项：

- `hardnet_input_prepare_ms`：FP32 数组/Tensor 准备和输入校验。
- `hardnet_batch_prepare_ms`：分批、可选 padding、pinning 和设备传输。
- `hardnet_layout_conversion_ms`：ONNX Runtime NCHW 输入准备。
- `hardnet_cnn_ms`：模型/外部图执行；适配器的结果校验和输出拷贝也在此范围内。
- `hardnet_normalization_ms`：外部 raw-q 输出的原 Python/Torch L2 归一化。
- `hardnet_hash_head_ms`：原生二值 checkpoint 的 hash head。
- `hardnet_output_transfer_ms`：拼接和最终回传/NumPy 转换。
- `hardnet_binarization_ms`：原生 sign/packing，或 Hadamard/sign/packbits。
- `hardnet_inference_overhead_ms`：包装、输出校验等剩余开销。

PyTorch 的 `forward()/forward_features()` 原本就计算 L2，保留该调用以免改变训练模型接口；
因此其 L2 包含在 `hardnet_cnn_ms` 中、`hardnet_normalization_ms` 为 0。
跨后端对照 CNN 时应比较 `hardnet_cnn_ms + hardnet_normalization_ms`，
整体性能优先比较 `hardnet_inference_ms`。Strong V2 等架构内部固有的归一化不删除。

子项作为 HardNet 总时间的细分，不能再与 `hardnet_inference_ms` 一起求和。
离线模板平均耗时和 `hardnet_inference.stages` 均传播这些字段。
关闭 profile 时子项为 0，并通过 `hardnet_inference_profiled` 标记，避免将未测量误认成零开销。

### 5.5 固定真实 patch 的后端回归与耗时

`benchmark_cpu_inference.py` 可以用 `--images` 显式指定图像；这次选用了参考项目基准中的同四张图，
patch 数分别为 373、390、396、388，patch SHA256 与已有 patch 清单一致。每个后端在独立进程内
预热 2 次、测量 7 次，比较相同 patch。主配置、外层 SIFT/patch/匹配参数不参与后端差异。

本次融合状态、后端线程和格式都与参考配置对齐：

- ONNX Runtime 是唯一的 CNN 推理后端，28 个主干线程、inter-op=1、图优化 all、Conv–BN 融合：**27.53 ms**。

这些数值是四张图各测七次所得 `describe()` 耗时的中位数，不是四张图总耗时。参考目录曾报告单图约 71.15/48.77/64.13 ms；其输入批次、测试轮数及计时运行条件
并非当前这四图七轮的同一统计样本，因此应以相同机器负载、相同输入清单、相同预热/重复次数重跑后再做严格数值比较。

四张图合计 396,032 个输出 bit 均未变化，packed byte 完全一致；raw `q` 在 `rtol=1e-4, atol=1e-5`
内。6 个图像对和 4 个自匹配样本的候选及判定结果一致。自匹配只用于覆盖通过分支，不构成识别准确率证明。
这不是全量 bit 一致性或 FAR/FRR 认证；正式部署仍需独立输出目录完整评估。

复现完全相同的图像：

```powershell
D:\Anaconda\envs\hardnet\python.exe -m match_new.benchmark_cpu_inference `
  --config match_new/config_match_new.yaml `
  --images data\normal_pic\crp_L0\pair_45.bmp data\normal_pic\crp_L0\pair_46.bmp `
           data\normal_pic\crp_L0\pair_47.bmp data\normal_pic\crp_L0\pair_48.bmp `
  --variants onnxruntime_16 onnxruntime_28_fused `
  --limit-images 4 --warmup 2 --repeats 7 `
  --output-dir outputs/cpu_reference_repeat
```

默认 `benchmark_cpu_inference.py` 会启用分阶段计时；主配置中的 `profile_inference: false` 则让正常模板构建使用不插入阶段计时的 `describe()` 路径。
测试全部代码回归：

```powershell
D:\Anaconda\envs\hardnet\python.exe -m pytest -q hardnet_train
```

可选后端直接由 `benchmark_cpu_inference.py` 的变体参数控制，不再维护单独的后端 YAML。

```powershell
D:\Anaconda\envs\hardnet\python.exe -m match_new.run_hardnet_matching `
  --config match_new/config_match_new.yaml
```

正式 FAR 评估还需要显式启用非本人比较；当前继承的
`runtime.max_impostor_identities_per_query: 0` 不会计算非本人 FAR，
本次为保持匹配参数不变没有修改此值。

