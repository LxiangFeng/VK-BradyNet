# Islam et al. 2023 FT severity reproduction
## PyCharm 工程（适配你的 FT 0/1/2 数据）

论文：
**Islam MS et al. Using AI to measure Parkinson's disease severity at home.
npj Digital Medicine 6, 156 (2023).**

官方代码：
https://github.com/ROC-HCI/finger-tapping-severity

本工程的目标不是重写论文算法，而是：

1. 从官方 GitHub 拉取 `feature_extraction.py`；
2. 对你的 FT 视频使用论文的 MediaPipe/angle-based feature extraction；
3. 保留论文的核心 `LightGBM regressor` 形式；
4. 使用你已经固定的 train / validation / test patient-level split；
5. 在 0/1/2 三个等级上将连续预测舍入为离散等级；
6. 在同一个 held-out FT test set 上计算 **exact Accuracy**、Macro-F1、QWK、MAE；
7. 自动生成 Table 5 的结果行。

---

## 一、与你论文比较时的实验定义

最终建议写：

```text
Evaluation setting:
Published-method reproduction

Method:
Islam et al. (2023)

Training data:
Our FT cohort

Test data:
Our held-out FT test set

Task:
FT

N videos / participants:
由脚本自动统计

Accuracy (%):
由实际预测得到
```

这里不是拿 Islam 论文原始 MAE 与 VK-BradyNet 的 Accuracy 横向比较。
Islam 原论文使用 LightGBM 回归连续 0--4 score，并通过舍入得到离散类别。
本工程在你的 0/1/2 数据上训练同类回归模型，然后将预测限制在 0--2 后舍入。

---

## 二、你的数据目录

推荐保持：

```text
data/
  train/
    0/*.mp4
    1/*.mp4
    2/*.mp4
  val/
    0/*.mp4
    1/*.mp4
    2/*.mp4
  test/
    0/*.mp4
    1/*.mp4
    2/*.mp4
```

你之前上传的 150 FT ZIP 虽然根目录名称是 `train/0, train/1, train/2`，
但如果这批数据是论文中的 held-out FT test set，请用：

```bash
python scripts/00_import_split.py --zip "你的150FT.zip" --split test
```

脚本会读取类别文件夹 0/1/2，然后导入到 `data/test/`，
不会把 ZIP 内部的 `train` 名称误认为训练集。

文件名如：

```text
00242_0003.mp4
```

默认解析：

```text
patient_id = 00242
```

---

## 三、环境

推荐：

- Windows 10/11
- PyCharm
- Python 3.10
- CPU 可以运行；LightGBM 不要求 GPU
- MediaPipe 特征提取是最耗时的部分

创建环境后：

```bash
pip install -r requirements.txt
```

然后：

```bash
python scripts/01_setup_official_repo.py
```

这一步会下载作者官方仓库到：

```text
third_party/finger-tapping-severity/
```

官方仓库提供：
- `feature_extraction.py`
- `model_training.py`
- `severity_dataset_dropped_correlated_columns.csv`

本工程直接调用官方 feature extractor，而不是自行杜撰特征定义。

---

## 四、关于 Left / Right

官方 `extract_features(filename, output_path, hand)` 明确要求 `hand=left/right`。

如果你有真实 left/right metadata，优先把它写入 manifest。

如果没有，可以运行：

```bash
python scripts/03_autodetect_side.py --split test
```

它使用与官方代码相同的 MediaPipe handedness 逻辑，并采用官方代码的
“非镜像视频需要反向 handedness 标签”映射：

```text
MediaPipe Right -> requested hand = left
MediaPipe Left  -> requested hand = right
```

自动推断只作为兼容方案。脚本会保存 `side_confidence`；
若 confidence 很低，建议人工检查。

---

## 五、推荐执行顺序

### 1. 导入数据

如果是 ZIP：

```bash
python scripts/00_import_split.py --zip "D:\FT_test_150.zip" --split test
python scripts/00_import_split.py --zip "D:\FT_train.zip" --split train
python scripts/00_import_split.py --zip "D:\FT_val.zip" --split val
```

如果你已经手工放入 `data/train|val|test/0|1|2`，跳过此步。

### 2. 下载官方代码

```bash
python scripts/01_setup_official_repo.py
```

### 3. 建 manifest

```bash
python scripts/02_build_manifest.py --split train
python scripts/02_build_manifest.py --split val
python scripts/02_build_manifest.py --split test
```

### 4. 补左右手

优先人工/临床 metadata。

没有 metadata 时：

```bash
python scripts/03_autodetect_side.py --split train
python scripts/03_autodetect_side.py --split val
python scripts/03_autodetect_side.py --split test
```

### 5. 提取 Islam 官方特征

```bash
python scripts/04_extract_official_features.py --split train
python scripts/04_extract_official_features.py --split val
python scripts/04_extract_official_features.py --split test
```

每个视频失败时不会被悄悄填补，而是写入：

```text
results/extraction_failures_<split>.csv
```

这点对你的短 clips 很重要。Islam 原方法依赖 tap peaks、period、
frequency、amplitude decrement 等统计；过短视频可能没有足够 peaks。

### 6. 训练并测试

推荐：

```bash
python scripts/05_train_evaluate_lightgbm.py --mode val_tune
```

输出：

```text
results/metrics.json
results/test_predictions.csv
results/confusion_matrix.csv
results/selected_features.txt
results/model.joblib
```

最终 exact Accuracy：

```text
pred_class == true_label
```

### 7. Table 5 行

```bash
python scripts/06_make_table5_row.py
```

---

## 六、为什么训练脚本与 Islam 原论文不完全采用 LOPO

Islam 原论文在自己的 250-participant cohort 上采用
leave-one-patient-out cross-validation。

你的论文已经预先固定了 patient-level train/val/test split。
为了与 VK-BradyNet 在 **同一 test set** 公平比较，本工程不重新做 LOPO，
而是：

```text
your train -> train baseline
your val   -> select baseline hyperparameters
your test  -> evaluate once
```

这是 **evaluation protocol adaptation**，不是算法结构创新。
论文中应明确写：

> The publicly released feature-extraction pipeline and LightGBM severity
> regression framework of Islam et al. were adapted to the three-class
> setting and retrained using the same patient-level data partition as
> VK-BradyNet.

---

## 七、重要：短视频兼容性

Islam 原研究使用完整 finger-tapping task video，并从整个序列计算：
- speed
- period/frequency
- amplitude
- rhythm/aperiodicity
- freezing/interruption
- amplitude decrement
- wrist movement

你当前上传的 150 clips 很短，因此部分 clips 可能没有足够峰值供官方
`get_final_features()` 使用。

本工程 **不会人为复制帧、循环视频或补造 taps**。

如果大量 test clips extraction failure，应优先考虑：
1. 用产生这些 clips 的原始完整 FT task-side videos 复现 Islam 方法；或
2. 在论文中明确说明该 published method 与 2-s clip evaluation unit
   存在输入时长不兼容，不能进行直接公平复现。

不要为了得到数字而修改原始测试动作内容。

---

## 八、模式说明

`05_train_evaluate_lightgbm.py` 支持：

```text
--mode fixed
```

LightGBM 使用固定通用设置，不利用 validation 做调参。

```text
--mode val_tune
```

在一个预先定义的小型参数网格上，以 validation MAE 选择参数，
test set 只使用一次。推荐用于最终 Table 5。

---

## 九、引用和方法依据

官方仓库 README 明确说明：
- `feature_extraction.py` 用于指定文件和 left/right hand 的特征提取；
- `model_training.py` 用于 severity model；
- ground truth `Rating` 来自 expert consensus。

论文报告：
- 250 participants；
- 489 usable left/right videos；
- 47 finger-tapping features + 18 wrist-movement features；
- LightGBM regressor；
- leave-one-patient-out validation；
- continuous severity prediction rounded to the nearest integer for
  classification analysis。

本工程在此基础上适配你的 0/1/2 fixed-split comparison。
