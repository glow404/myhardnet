# pair_build

`pair_build` 是 HardNet 指纹训练数据构建目录，负责把原始小指纹图像转换成
HardNet 可训练的正样本 patch 对。

## 主要文件

- `config.yaml`：数据构建配置，包括原始数据路径、输出路径、SIFT 参数、匹配过滤参数、patch 裁剪参数。
- `main.py`：数据构建入口，支持 `index`、`split`、`build_positive`、`extract_patches`、`qa`、`all` 阶段。
- `index_builder.py`：扫描原始图像，建立样本索引。
- `split_builder.py`：按 `finger_id` 划分 train/val/test，避免同一手指泄漏到不同 split。
- `positive_builder.py`：使用 SIFT、RANSAC 和几何扩展构建正样本对应点。
- `patch_extractor.py`：根据正样本对应点裁剪并保存 32x32 patch。
- `qa_tools.py`：生成 QA 报告和可视化预览图。
- `dataset_hardnet.py`：早期/辅助的 HardNet 正样本 DataLoader。
- `utils.py`：路径、CSV、JSON、图像读写等通用工具。
- `scheme.md`：数据构建思路记录。

## 常用命令

完整构建：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage all
```

只生成 QA：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage qa
```

## 输出关系

默认输出到：

```text
outputs/hardnet_dataset/
```

`hardnet_train` 会读取其中的：

- `train_pairs.csv`
- `val_pairs.csv`
- `test_pairs.csv`
- `patches/`

