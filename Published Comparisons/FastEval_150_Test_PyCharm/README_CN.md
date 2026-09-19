# FastEval Parkinsonism × Our FT Test Set
## PyCharm 可直接打开的测试工程

这个工程用于把 **FastEval Parkinsonism 官方 Hand Predictor API** 运行在我们自己的 FT 测试集上，
并计算与 VK-BradyNet 一致的 **exact Accuracy**。

> 重要：如果你只导入 150 个测试 clips，这个实验属于
> **Zero-shot published-model transfer**：
> FastEval 使用作者原始训练数据/官方权重，我们只在自己的测试集上做推理。
>
> 如果论文 Table 5 要做“同一训练数据、同一测试数据”的公平方法比较，
> 则还需要用我们的 FT train/val 数据重新训练 FastEval 方法。
> 这个压缩包首先解决“150 个测试样本直接跑官方模型”的问题，不会伪装成重新训练结果。

---

## 1. 为什么可以这样测试

FastEval 官方仓库公开了 Hand Predictor API，并给出命令行推理方法。
论文/官方代码来源：

- Paper: Yang et al., *npj Digital Medicine*, 2024
- Official repository:
  https://github.com/yuyuan871111/fast_eval_Parkinsonism
- License: Apache-2.0

FastEval 原模型输出的是 MDS-UPDRS 0 / 1 / 2 / 3+。
我们的 FT 测试集只有 0 / 1 / 2。
本工程保留官方预测结果，并用严格相等：
`pred == label`
计算 exact Accuracy。

**不要使用 FastEval 论文中的 AAC 作为 Accuracy。**
FastEval 的 AAC 允许预测与真实标签相差 1 分仍算 acceptable，
与我们论文的 exact Accuracy 不同。

---

## 2. 数据准备

把 150 个 FT 测试 clips 放入：

```text
data/test_videos/
```

然后编辑：

```text
data/test_manifest.csv
```

格式：

```csv
filename,label,patient_id,side
FT_0001.mp4,0,P001,Left
FT_0002.mp4,1,P002,Right
FT_0003.mp4,2,P003,Left
```

要求：

- `filename`：视频文件名
- `label`：只能是 0、1、2
- `patient_id`：患者 ID，仅用于统计，不传给模型
- `side`：必须为 Left 或 Right

---

## 3. PyCharm 使用顺序

### Step A — 打开工程

PyCharm -> Open -> 选择整个 `FastEval_150_Test_PyCharm` 文件夹。

推荐 Python 3.8，因为 FastEval 官方论文/代码使用 Python 3.8.x。

### Step B — 安装本工程辅助依赖

```bash
pip install -r requirements_wrapper.txt
```

### Step C — 下载 FastEval 官方代码

运行：

```bash
python scripts/01_setup_official_repo.py
```

它会执行：

```bash
git clone https://github.com/yuyuan871111/fast_eval_Parkinsonism.git
```

到：

```text
third_party/fast_eval_Parkinsonism/
```

随后请按照官方仓库的 `environment.yml` 创建 FastEval 环境。
官方 README 推荐：

```bash
conda env create -f third_party/fast_eval_Parkinsonism/environment.yml
conda activate mediapipe
```

### Step D — 检查 150 个样本

```bash
python scripts/02_check_testset.py
```

应显示：

- 总视频数
- 标签 0/1/2 分布
- unique patient 数
- 左/右手分布
- 缺失文件
- 视频时长统计

FastEval 官方建议上传视频 >5 s、720p、60 fps。
我们的样本是 2-s clips，因此脚本会把 `<5 s` 标为 warning。
**不会自动拉长、复制或改变视频内容**，避免人为篡改测试数据。

### Step E — 跑官方 FastEval 推理

```bash
python scripts/03_run_official_fasteval.py
```

脚本逐个调用官方：

```text
src/lib/hand_predictor/hand_predictor.py
```

并将每个样本的 stdout/stderr 以及输出目录保存到：

```text
results/raw/
```

结果汇总到：

```text
results/predictions_raw.csv
```

如果官方输出格式可被自动解析，会自动填写 `pred_score`。
如果某个版本的官方代码改变了输出文件名/字段，
`pred_score` 可能为空；此时把 `results/raw/` 中任意 1 个成功样本的输出发给我，
我可以把 parser 精确改到该版本。

### Step F — 计算 exact Accuracy

```bash
python scripts/04_evaluate_exact_accuracy.py
```

输出：

```text
results/metrics.json
results/predictions_scored.csv
```

其中主指标：

```text
exact_accuracy = number(pred_score == label) / N
```

同时给出 Macro-F1、QWK、混淆矩阵数据，方便后续分析，
但你的 Table 5 当前只需要 Accuracy (%)。

### Step G — 生成 Table 5 行

```bash
python scripts/05_make_table5_row.py
```

会打印类似：

```text
Zero-shot published-model transfer | FastEval Parkinsonism |
FastEval original cohort | Our held-out FT test set | FT |
150 clips / XX participants | XX.XX
```

---

## 4. 关于“只用 150 个测试数据”的论文表述

如果使用这个工程直接跑作者官方权重，Table 5 应写：

```text
Evaluation setting:
Zero-shot published-model transfer

Method:
FastEval Parkinsonism

Training data:
FastEval original cohort

Test data:
Our held-out FT test set

Task:
FT

N videos / participants:
150 clips / X participants

Accuracy (%):
实际运行结果
```

不要写成：

```text
Published-method reproduction
Training data: Our FT training set
```

因为那需要重新训练 FastEval。

---

## 5. 公平重训练实验

如果后续要做更强的 Table 5 比较，应使用：

- Our FT train: 1200 clips
- Our FT val: 150 clips
- Our FT test: 150 clips

并在相同 patient-level split 上重新训练 FastEval/PDHandNet，
最后只在同一 150 test clips 上计算 exact Accuracy。

那一版我建议单独做成 `FastEval_Fair_Retrain` 工程，
不要把 zero-shot 与 retraining 混在一起。

---

## 6. 重要科学说明

1. FastEval 原研究的视频约 10 s、60 fps、side view。
2. 我们当前 FT 测试样本是 2-s clips。
3. 因此直接使用官方 pretrained model 属于明显的 acquisition/domain shift。
4. 如果官方 Hand Predictor 对 2-s 输入无法稳定推理，不建议人工复制帧或循环视频来满足长度要求。
5. 此时应转为“在我们的 train/val 上重新训练 FastEval architecture，再测试 150 clips”。
