# -*- coding: utf-8 -*-
"""
Independent RGB + 10D Keypoint Fusion Video-level Test Metrics

Final SCI-style version with HOC labels:
1. 自动创建 outputs/test_metrics。
2. 修复 cfg freeze 导致的 CfgNode immutable 问题。
3. 从 C:\\Users\\Administrator\\Desktop\\data\\test.csv 读取测试列表。
4. 从 C:\\Users\\Administrator\\Desktop\\npy\\test 自动加载对应 .npy 关键点。
5. forward 时传入 keypoints，避免 keypoints=None。
6. 保存完整 SCI 常用评估指标与图像：
   - confusion_matrix.npy
   - confusion_matrix.csv
   - confusion_matrix.png
   - confusion_matrix_normalized.png
   - predictions.csv
   - metrics.json
   - classification_report.txt
   - classification_report.json
   - per_class_metrics.csv
   - roc_curve.png
   - pr_curve.png
   - roc_auc.csv
   - pr_ap.csv
   - roc_curve_points.csv
   - pr_curve_points.csv
   - missing_keypoints.csv

SCI-style confusion matrix:
- 白色背景
- 蓝色单色渐变
- 600 dpi
- 细网格
- serif / Times New Roman 风格
- 图片类别名自动加 HOC_ 前缀
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Project path
# =============================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Optional sklearn metrics
# =============================================================================

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# =============================================================================
# SlowFast imports
# =============================================================================

try:
    from slowfast.config.defaults import get_cfg
    from slowfast.models import build_model
    import slowfast.datasets.loader as loader
except Exception as e:
    print("Failed to import SlowFast modules.")
    print("Please make sure this script is inside SlowFast-main/pd_mvitv2_s_repro_pack/")
    print("Error:", repr(e))
    raise


# =============================================================================
# Default paths
# =============================================================================

DEFAULT_CFG = (
    PROJECT_ROOT
    / "pd_mvitv2_s_repro_pack"
    / "MVITv2_S_PD_3cls_rgb_keypoint_fusion.yaml"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "PD_MVITv2_S_RGB_Keypoint_Fusion"
    / "checkpoints"
    / "checkpoint_epoch_00020.pyth"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_metrics"

DEFAULT_TEST_CSV = Path(r"C:\Users\Administrator\Desktop\data\data\test.csv")
DEFAULT_KEYPOINT_DIR = Path(r"C:\Users\Administrator\Desktop\data\npy\test")


# =============================================================================
# Basic utils
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB + 10D Keypoint Fusion video-level test metrics."
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=str(DEFAULT_CFG),
        help="Path to SlowFast config yaml.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to checkpoint .pyth file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save metrics.",
    )

    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help="Path to test.csv.",
    )

    parser.add_argument(
        "--keypoint-dir",
        type=str,
        default=str(DEFAULT_KEYPOINT_DIR),
        help="Directory containing test keypoint .npy files.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split.",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp autocast during evaluation.",
    )

    parser.add_argument(
        "--print-freq",
        type=int,
        default=20,
        help="Print frequency.",
    )

    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional config overrides.",
    )

    return parser.parse_args()


def setup_cfg(args):
    cfg_path = Path(args.cfg)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = get_cfg()
    cfg.defrost()

    cfg.merge_from_file(str(cfg_path))

    if args.opts:
        opts = args.opts
        if len(opts) > 0 and opts[0] == "--":
            opts = opts[1:]
        if len(opts) > 0:
            cfg.merge_from_list(opts)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        try:
            cfg.TEST.CHECKPOINT_FILE_PATH = str(checkpoint_path)
        except Exception:
            pass

    # 关键：这里不要 freeze。
    # MViT 构建过程中可能会修改 cfg.MVIT.POOL_KV_STRIDE。
    cfg.defrost()

    return cfg


def to_device(x, device, non_blocking=True):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)

    if isinstance(x, list):
        return [to_device(v, device, non_blocking=non_blocking) for v in x]

    if isinstance(x, tuple):
        return tuple(to_device(v, device, non_blocking=non_blocking) for v in x)

    if isinstance(x, dict):
        return {
            k: to_device(v, device, non_blocking=non_blocking)
            for k, v in x.items()
        }

    return x


def tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_dict_first_existing(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def safe_softmax(preds):
    if isinstance(preds, dict):
        preds = get_dict_first_existing(
            preds,
            ["logits", "preds", "prediction", "output", "outputs"],
        )
        if preds is None:
            raise ValueError("Model output is dict, but no valid logits key found.")

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    if not torch.is_tensor(preds):
        preds = torch.as_tensor(preds)

    return torch.softmax(preds, dim=1)


def unpack_batch(batch):
    """
    兼容 SlowFast 常见 batch 格式：
    1. inputs, labels, video_idx, meta
    2. inputs, labels, video_idx
    3. inputs, labels
    4. dict 格式
    """
    inputs = None
    labels = None
    video_idx = None
    meta = {}

    if isinstance(batch, dict):
        inputs = get_dict_first_existing(
            batch,
            ["inputs", "video", "frames", "imgs", "data"],
        )
        labels = get_dict_first_existing(
            batch,
            ["labels", "label", "targets", "target"],
        )
        video_idx = get_dict_first_existing(
            batch,
            ["video_idx", "video_index", "index", "idx"],
        )

        meta = batch.get("meta", batch)

    elif isinstance(batch, (list, tuple)):
        if len(batch) >= 1:
            inputs = batch[0]
        if len(batch) >= 2:
            labels = batch[1]
        if len(batch) >= 3:
            video_idx = batch[2]
        if len(batch) >= 4 and isinstance(batch[3], dict):
            meta = batch[3]
        elif len(batch) >= 3 and isinstance(batch[2], dict):
            meta = batch[2]
            video_idx = None
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if inputs is None:
        raise ValueError("Cannot find inputs from batch.")

    if labels is None:
        raise ValueError("Cannot find labels from batch.")

    return inputs, labels, video_idx, meta


def get_batch_size_from_labels(labels):
    labels_np = tensor_to_numpy(labels).reshape(-1)
    return len(labels_np)


def get_video_idx_list(video_idx, batch_size):
    if video_idx is None:
        return [None] * batch_size

    if torch.is_tensor(video_idx):
        arr = tensor_to_numpy(video_idx).reshape(-1).tolist()
    elif isinstance(video_idx, np.ndarray):
        arr = video_idx.reshape(-1).tolist()
    elif isinstance(video_idx, (list, tuple)):
        arr = list(video_idx)
    else:
        arr = [video_idx]

    if len(arr) == batch_size:
        return arr

    if len(arr) == 1 and batch_size > 1:
        return arr * batch_size

    return [None] * batch_size


def get_meta_item(meta, key_candidates, i):
    if not isinstance(meta, dict):
        return None

    for key in key_candidates:
        if key not in meta:
            continue

        value = meta[key]

        if torch.is_tensor(value):
            value = tensor_to_numpy(value)

        if isinstance(value, np.ndarray):
            value = value.reshape(-1).tolist()

        if isinstance(value, (list, tuple)):
            if i < len(value):
                return value[i]
        else:
            return value

    return None


def infer_temporal_len_from_inputs(inputs):
    """
    从 RGB 输入中推断 T。
    SlowFast/MViT 常见输入：
    - Tensor: [B, C, T, H, W]
    - List[Tensor]: [[B, C, T, H, W]]
    """
    x = inputs

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]

    if torch.is_tensor(x):
        if x.dim() == 5:
            return int(x.shape[2])
        if x.dim() == 4:
            return int(x.shape[1])

    return None


def normalize_keypoint_array(arr, target_len=None, feature_dim=10):
    """
    把各种 .npy 格式统一成 [T, D]。
    """
    arr = np.asarray(arr)

    if arr.dtype == np.object_:
        arr = arr.astype(np.float32)

    arr = arr.astype(np.float32)

    if arr.ndim == 0:
        arr = np.zeros((1, feature_dim), dtype=np.float32)

    elif arr.ndim == 1:
        if arr.size % feature_dim == 0:
            arr = arr.reshape(-1, feature_dim)
        else:
            tmp = np.zeros((1, feature_dim), dtype=np.float32)
            n = min(feature_dim, arr.shape[0])
            tmp[0, :n] = arr[:n]
            arr = tmp

    elif arr.ndim == 2:
        pass

    elif arr.ndim >= 3:
        if arr.shape[-1] == feature_dim:
            arr = arr.reshape(-1, feature_dim)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[-1] >= feature_dim:
                arr = arr[:, :feature_dim]
            else:
                pad = np.zeros(
                    (arr.shape[0], feature_dim - arr.shape[-1]),
                    dtype=np.float32,
                )
                arr = np.concatenate([arr, pad], axis=1)

    if arr.ndim != 2:
        arr = arr.reshape(-1, feature_dim)

    if arr.shape[1] > feature_dim:
        arr = arr[:, :feature_dim]
    elif arr.shape[1] < feature_dim:
        pad = np.zeros((arr.shape[0], feature_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    if arr.shape[0] <= 0:
        arr = np.zeros((1, feature_dim), dtype=np.float32)

    if target_len is not None and target_len > 0 and arr.shape[0] != target_len:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, target_len, axis=0)
        else:
            idx = np.linspace(0, arr.shape[0] - 1, target_len)
            idx = np.round(idx).astype(np.int64)
            arr = arr[idx]

    return arr.astype(np.float32)


# =============================================================================
# Save functions
# =============================================================================

def save_json(path, data):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_predictions_csv(path, rows, class_names=None):
    path = Path(path)

    if not rows:
        return

    num_classes = len(rows[0]["probs"])

    prob_headers = []
    for i in range(num_classes):
        if class_names is not None and i < len(class_names):
            prob_headers.append(f"prob_{i}_{class_names[i]}")
        else:
            prob_headers.append(f"prob_{i}")

    headers = [
        "video_id",
        "label",
        "pred",
        "correct",
        "npy_path",
    ] + prob_headers

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in rows:
            writer.writerow(
                [
                    r["video_id"],
                    r["label"],
                    r["pred"],
                    int(r["correct"]),
                    r.get("npy_path", ""),
                    *[float(x) for x in r["probs"]],
                ]
            )


def save_confusion_matrix_csv(path, conf_mat, class_names=None):
    path = Path(path)

    num_classes = conf_mat.shape[0]

    if class_names is None or len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(class_names))

        for i in range(num_classes):
            writer.writerow([class_names[i]] + conf_mat[i].tolist())


def save_confusion_matrix_png(path, conf_mat, class_names=None, normalize=False):
    """
    保存 SCI 顶刊风格混淆矩阵 PNG 图像。

    图片中的类别名自动加 HOC_ 前缀：
    class_0 -> HM_class_0
    class_1 -> HM_class_1
    class_2 -> HM_class_2
    """
    path = Path(path)

    conf_mat = np.asarray(conf_mat)
    num_classes = conf_mat.shape[0]

    if class_names is None or len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    display_class_names = []
    for name in class_names:
        name = str(name)
        if name.startswith("HM_"):
            display_class_names.append(name)
        else:
            display_class_names.append(f"HM_{name}")

    if normalize:
        cm = conf_mat.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
        title = "HM Normalized Confusion Matrix"
        colorbar_label = "Proportion"
    else:
        cm = conf_mat.astype(np.int64)
        title = "HM Confusion Matrix"
        colorbar_label = "Count"

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
        aspect="equal",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=15)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=3)

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted HM label", labelpad=8)
    ax.set_ylabel("True HM label", labelpad=8)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(
        display_class_names,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticklabels(display_class_names)

    ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    max_value = float(np.max(cm)) if cm.size > 0 else 1.0
    threshold = max_value * 0.55

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]

            if normalize:
                text_str = f"{value:.2f}"
            else:
                text_str = f"{int(value)}"

            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if value > threshold else "normal",
                color="white" if value > threshold else "#222222",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_report_txt(path, text):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0,
    }

    if HAS_SKLEARN:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        metrics.update(
            {
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
            }
        )

    return metrics


def save_per_class_metrics_csv(path, y_true, y_pred, conf_mat, class_names):
    """
    保存每类 Precision / Recall / F1 / Support / Specificity。
    Specificity = TN / (TN + FP)
    """
    path = Path(path)

    num_classes = len(class_names)

    if HAS_SKLEARN:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
        )
    else:
        precision = np.zeros(num_classes, dtype=np.float32)
        recall = np.zeros(num_classes, dtype=np.float32)
        f1 = np.zeros(num_classes, dtype=np.float32)
        support = conf_mat.sum(axis=1)

        for i in range(num_classes):
            tp = conf_mat[i, i]
            fp = conf_mat[:, i].sum() - tp
            fn = conf_mat[i, :].sum() - tp

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[i] = (
                2 * precision[i] * recall[i] / (precision[i] + recall[i])
                if (precision[i] + recall[i]) > 0
                else 0.0
            )

    total = conf_mat.sum()

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class_index",
                "class_name",
                "precision",
                "recall_sensitivity",
                "specificity",
                "f1",
                "support",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )

        for i, cname in enumerate(class_names):
            tp = int(conf_mat[i, i])
            fp = int(conf_mat[:, i].sum() - tp)
            fn = int(conf_mat[i, :].sum() - tp)
            tn = int(total - tp - fp - fn)

            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            writer.writerow(
                [
                    i,
                    cname,
                    float(precision[i]),
                    float(recall[i]),
                    specificity,
                    float(f1[i]),
                    int(support[i]),
                    tp,
                    fp,
                    tn,
                    fn,
                ]
            )


def save_multiclass_roc_pr(y_true, y_prob, class_names, out_dir):
    """
    保存多分类 one-vs-rest ROC / PR 曲线。

    输出：
    - roc_curve.png
    - pr_curve.png
    - roc_auc.csv
    - pr_ap.csv
    - roc_curve_points.csv
    - pr_curve_points.csv
    """
    out_dir = Path(out_dir)
    num_classes = len(class_names)

    summary = {
        "roc_auc_macro": None,
        "roc_auc_micro": None,
        "pr_ap_macro": None,
        "pr_ap_micro": None,
    }

    if not HAS_SKLEARN:
        return summary

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_prob.ndim != 2:
        print("Warning: y_prob is not 2D. Skip ROC/PR curves.")
        return summary

    if y_prob.shape[1] < num_classes:
        print(
            f"Warning: y_prob has {y_prob.shape[1]} columns, "
            f"but num_classes is {num_classes}. Skip ROC/PR curves."
        )
        return summary

    y_prob = y_prob[:, :num_classes]

    classes = list(range(num_classes))
    y_true_bin = label_binarize(y_true, classes=classes)

    if num_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)

    roc_rows = []
    pr_rows = []
    roc_auc_rows = []
    pr_ap_rows = []

    valid_auc_values = []
    valid_ap_values = []

    # -------------------------------------------------------------------------
    # ROC
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        binary_true = y_true_bin[:, i]

        if len(np.unique(binary_true)) < 2:
            roc_auc_rows.append([i, class_names[i], "nan", "skipped_no_positive_or_negative"])
            continue

        fpr, tpr, thresholds = roc_curve(binary_true, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        valid_auc_values.append(float(roc_auc))

        roc_auc_rows.append([i, class_names[i], float(roc_auc), "ok"])

        for k in range(len(fpr)):
            roc_rows.append(
                [
                    i,
                    class_names[i],
                    float(fpr[k]),
                    float(tpr[k]),
                    float(thresholds[k]) if k < len(thresholds) else "",
                ]
            )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_names[i]} AUC={roc_auc:.3f}",
        )

    try:
        if len(np.unique(y_true_bin.ravel())) >= 2:
            fpr_micro, tpr_micro, _ = roc_curve(
                y_true_bin.ravel(),
                y_prob.ravel(),
            )
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            summary["roc_auc_micro"] = float(roc_auc_micro)

            plt.plot(
                fpr_micro,
                tpr_micro,
                linewidth=2,
                linestyle="--",
                label=f"micro-average AUC={roc_auc_micro:.3f}",
            )
    except Exception as e:
        print(f"Warning: failed to compute micro ROC AUC: {repr(e)}")

    if valid_auc_values:
        summary["roc_auc_macro"] = float(np.mean(valid_auc_values))

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("HM ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # PR
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        binary_true = y_true_bin[:, i]

        if binary_true.sum() == 0:
            pr_ap_rows.append([i, class_names[i], "nan", "skipped_no_positive"])
            continue

        precision, recall, thresholds = precision_recall_curve(
            binary_true,
            y_prob[:, i],
        )
        ap = average_precision_score(binary_true, y_prob[:, i])
        valid_ap_values.append(float(ap))

        pr_ap_rows.append([i, class_names[i], float(ap), "ok"])

        for k in range(len(precision)):
            pr_rows.append(
                [
                    i,
                    class_names[i],
                    float(recall[k]),
                    float(precision[k]),
                    float(thresholds[k]) if k < len(thresholds) else "",
                ]
            )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{class_names[i]} AP={ap:.3f}",
        )

    try:
        if y_true_bin.sum() > 0:
            precision_micro, recall_micro, _ = precision_recall_curve(
                y_true_bin.ravel(),
                y_prob.ravel(),
            )
            ap_micro = average_precision_score(
                y_true_bin,
                y_prob,
                average="micro",
            )
            summary["pr_ap_micro"] = float(ap_micro)

            plt.plot(
                recall_micro,
                precision_micro,
                linewidth=2,
                linestyle="--",
                label=f"micro-average AP={ap_micro:.3f}",
            )
    except Exception as e:
        print(f"Warning: failed to compute micro PR AP: {repr(e)}")

    if valid_ap_values:
        summary["pr_ap_macro"] = float(np.mean(valid_ap_values))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    with open(out_dir / "roc_auc.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "auc", "status"])
        writer.writerows(roc_auc_rows)
        writer.writerow([])
        writer.writerow(["macro_auc", summary["roc_auc_macro"], "", ""])
        writer.writerow(["micro_auc", summary["roc_auc_micro"], "", ""])

    with open(out_dir / "pr_ap.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "average_precision", "status"])
        writer.writerows(pr_ap_rows)
        writer.writerow([])
        writer.writerow(["macro_ap", summary["pr_ap_macro"], "", ""])
        writer.writerow(["micro_ap", summary["pr_ap_micro"], "", ""])

    with open(out_dir / "roc_curve_points.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "fpr", "tpr", "threshold"])
        writer.writerows(roc_rows)

    with open(out_dir / "pr_curve_points.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "recall", "precision", "threshold"])
        writer.writerows(pr_rows)

    return summary


# =============================================================================
# Keypoint provider
# =============================================================================

class KeypointProvider:
    def __init__(self, test_csv, keypoint_dir, feature_dim=10):
        self.test_csv = Path(test_csv)
        self.keypoint_dir = Path(keypoint_dir)
        self.feature_dim = feature_dim

        if not self.test_csv.exists():
            raise FileNotFoundError(f"test.csv not found: {self.test_csv}")

        if not self.keypoint_dir.exists():
            raise FileNotFoundError(f"keypoint dir not found: {self.keypoint_dir}")

        self.csv_entries = self._load_csv_entries(self.test_csv)
        self.npy_files = sorted(self.keypoint_dir.rglob("*.npy"))
        self.npy_index = self._build_npy_index(self.npy_files)
        self.cache = {}
        self.missing_records = []

        print(f"Test CSV:      {self.test_csv}")
        print(f"Keypoint dir:  {self.keypoint_dir}")
        print(f"CSV rows:      {len(self.csv_entries)}")
        print(f"NPY files:     {len(self.npy_files)}")

    def _split_line(self, line):
        line = line.strip()
        if not line:
            return []

        if "," in line:
            return [x.strip() for x in line.split(",")]

        if "\t" in line:
            return [x.strip() for x in line.split("\t")]

        return [x.strip() for x in line.split()]

    def _load_csv_entries(self, csv_path):
        entries = []

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]

        if not lines:
            return entries

        first_parts = self._split_line(lines[0])
        has_header = False

        if first_parts:
            lower = [x.lower() for x in first_parts]
            header_words = ["path", "video", "filename", "file", "label", "class"]
            has_header = any(any(w in col for w in header_words) for col in lower)

        header = None
        data_lines = lines

        if has_header:
            header = [x.lower() for x in first_parts]
            data_lines = lines[1:]

        for idx, line in enumerate(data_lines):
            parts = self._split_line(line)
            if not parts:
                continue

            video_path = None
            label = None

            if header is not None:
                path_keys = ["path", "video", "video_path", "filename", "file"]
                label_keys = ["label", "class", "target"]

                for k in path_keys:
                    if k in header and header.index(k) < len(parts):
                        video_path = parts[header.index(k)]
                        break

                for k in label_keys:
                    if k in header and header.index(k) < len(parts):
                        try:
                            label = int(float(parts[header.index(k)]))
                        except Exception:
                            label = None
                        break

            if video_path is None:
                video_path = parts[0]

            if label is None and len(parts) >= 2:
                try:
                    label = int(float(parts[-1]))
                except Exception:
                    label = None

            p = Path(video_path)

            entries.append(
                {
                    "index": idx,
                    "raw": line,
                    "video_path": video_path,
                    "name": p.name,
                    "stem": p.stem,
                    "label": label,
                }
            )

        return entries

    def _build_npy_index(self, npy_files):
        index = {}

        for p in npy_files:
            name_key = p.name.lower()
            stem_key = p.stem.lower()

            index[name_key] = p
            index[stem_key] = p

            for suffix in [
                "_keypoints",
                "_kpts",
                "_pose",
                "_10d",
                "_feat",
                "_features",
            ]:
                if stem_key.endswith(suffix):
                    index[stem_key[: -len(suffix)]] = p

        return index

    def _candidate_keys_from_path(self, video_path):
        if video_path is None:
            return []

        p = Path(str(video_path).replace("\\", "/"))

        candidates = [
            p.name,
            p.stem,
            p.name + ".npy",
            p.stem + ".npy",
            p.stem + "_keypoints",
            p.stem + "_keypoints.npy",
            p.stem + "_kpts",
            p.stem + "_kpts.npy",
            p.stem + "_10d",
            p.stem + "_10d.npy",
            p.stem + "_feat",
            p.stem + "_feat.npy",
            p.stem + "_features",
            p.stem + "_features.npy",
        ]

        return [str(x).lower() for x in candidates]

    def _find_npy(self, video_idx=None, meta_path=None, fallback_pos=None):
        for key in self._candidate_keys_from_path(meta_path):
            if key in self.npy_index:
                return self.npy_index[key]

        if video_idx is not None:
            try:
                idx = int(video_idx)
                if 0 <= idx < len(self.csv_entries):
                    entry = self.csv_entries[idx]
                    for key in self._candidate_keys_from_path(entry["video_path"]):
                        if key in self.npy_index:
                            return self.npy_index[key]
                    for key in self._candidate_keys_from_path(entry["name"]):
                        if key in self.npy_index:
                            return self.npy_index[key]
            except Exception:
                pass

        if fallback_pos is not None:
            try:
                idx = int(fallback_pos)
                if 0 <= idx < len(self.csv_entries):
                    entry = self.csv_entries[idx]
                    for key in self._candidate_keys_from_path(entry["video_path"]):
                        if key in self.npy_index:
                            return self.npy_index[key]
                    for key in self._candidate_keys_from_path(entry["name"]):
                        if key in self.npy_index:
                            return self.npy_index[key]
            except Exception:
                pass

        if fallback_pos is not None:
            try:
                idx = int(fallback_pos)
                if 0 <= idx < len(self.npy_files):
                    return self.npy_files[idx]
            except Exception:
                pass

        if video_idx is not None:
            try:
                idx = int(video_idx)
                if 0 <= idx < len(self.npy_files):
                    return self.npy_files[idx]
            except Exception:
                pass

        return None

    def load_one(self, video_idx=None, meta_path=None, fallback_pos=None, target_len=None):
        npy_path = self._find_npy(
            video_idx=video_idx,
            meta_path=meta_path,
            fallback_pos=fallback_pos,
        )

        if npy_path is None:
            record = {
                "video_idx": str(video_idx),
                "meta_path": str(meta_path),
                "fallback_pos": str(fallback_pos),
                "reason": "npy_not_found",
            }
            self.missing_records.append(record)

            arr = np.zeros((target_len or 1, self.feature_dim), dtype=np.float32)
            return arr, None

        cache_key = (str(npy_path), int(target_len or -1))

        if cache_key in self.cache:
            return self.cache[cache_key], npy_path

        try:
            arr = np.load(str(npy_path), allow_pickle=True)
            arr = normalize_keypoint_array(
                arr,
                target_len=target_len,
                feature_dim=self.feature_dim,
            )
        except Exception as e:
            record = {
                "video_idx": str(video_idx),
                "meta_path": str(meta_path),
                "fallback_pos": str(fallback_pos),
                "npy_path": str(npy_path),
                "reason": repr(e),
            }
            self.missing_records.append(record)

            arr = np.zeros((target_len or 1, self.feature_dim), dtype=np.float32)

        self.cache[cache_key] = arr
        return arr, npy_path

    def make_batch_keypoints(
        self,
        labels,
        video_idx,
        meta,
        inputs,
        global_sample_start,
        device,
    ):
        batch_size = get_batch_size_from_labels(labels)
        video_idx_list = get_video_idx_list(video_idx, batch_size)
        target_len = infer_temporal_len_from_inputs(inputs)

        if target_len is None:
            target_len = 16

        arrays = []
        paths = []

        for i in range(batch_size):
            meta_path = get_meta_item(
                meta,
                [
                    "path",
                    "paths",
                    "video_path",
                    "video_paths",
                    "filename",
                    "filenames",
                    "video_name",
                    "video_names",
                    "video_id",
                    "video_ids",
                ],
                i,
            )

            fallback_pos = global_sample_start + i

            arr, npy_path = self.load_one(
                video_idx=video_idx_list[i],
                meta_path=meta_path,
                fallback_pos=fallback_pos,
                target_len=target_len,
            )

            arrays.append(arr)
            paths.append(str(npy_path) if npy_path is not None else "")

        keypoints = np.stack(arrays, axis=0).astype(np.float32)
        keypoints = torch.from_numpy(keypoints).to(device)

        return keypoints, paths

    def save_missing_records(self, out_dir):
        out_dir = Path(out_dir)
        path = out_dir / "missing_keypoints.csv"

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video_idx",
                    "meta_path",
                    "fallback_pos",
                    "npy_path",
                    "reason",
                ],
            )
            writer.writeheader()

            for r in self.missing_records:
                writer.writerow(
                    {
                        "video_idx": r.get("video_idx", ""),
                        "meta_path": r.get("meta_path", ""),
                        "fallback_pos": r.get("fallback_pos", ""),
                        "npy_path": r.get("npy_path", ""),
                        "reason": r.get("reason", ""),
                    }
                )

        return path


# =============================================================================
# Model / loader
# =============================================================================

def get_num_classes_from_cfg(cfg):
    try:
        if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "NUM_CLASSES"):
            return int(cfg.MODEL.NUM_CLASSES)
    except Exception:
        pass

    try:
        if hasattr(cfg, "DATA") and hasattr(cfg.DATA, "NUM_CLASSES"):
            return int(cfg.DATA.NUM_CLASSES)
    except Exception:
        pass

    return 3


def get_class_names(cfg, num_classes):
    """
    可以在这里改成真实类别名。

    例如：
    return ["normal", "mild", "severe"]

    注意：
    混淆矩阵 PNG 会自动显示为 HOC_class_0 / HOC_class_1 / HOC_class_2。
    CSV 和 JSON 仍保持原始 class_0 / class_1 / class_2，便于程序处理。
    """
    return [f"class_{i}" for i in range(num_classes)]


def strip_prefix_if_present(state_dict, prefix):
    keys = list(state_dict.keys())

    if len(keys) == 0:
        return state_dict

    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint_flexible(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading network weights from {checkpoint_path}.")

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "model.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("missing keys:", list(missing))
    print("unexpected keys:", list(unexpected))

    return model


def build_test_loader(cfg, split):
    return loader.construct_loader(cfg, split)


def build_network(cfg, device):
    try:
        cfg.defrost()
    except Exception:
        pass

    model = build_model(cfg)
    model = model.to(device)

    return model


def maybe_call_model(model, inputs, keypoints):
    """
    优先使用关键字 keypoints。
    如果 forward 是 forward(self, x, bboxes=None, keypoints=None)，可以正常传入。
    """
    if keypoints is None:
        return model(inputs)

    try:
        return model(inputs, keypoints=keypoints)
    except TypeError:
        pass

    try:
        return model(inputs, None, keypoints)
    except TypeError:
        pass

    try:
        return model(inputs, keypoints)
    except TypeError:
        pass

    raise RuntimeError(
        "Failed to call model with keypoints. "
        "Please check model.forward signature in video_model_builder.py."
    )


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader,
    device,
    out_dir,
    keypoint_provider,
    class_names=None,
    print_freq=20,
    use_amp=False,
):
    out_dir = ensure_dir(out_dir)

    model.eval()

    video_probs = defaultdict(list)
    video_labels = {}
    video_npy_paths = {}

    sample_counter = 0
    total_iters = len(test_loader)

    for cur_iter, batch in enumerate(test_loader):
        if cur_iter % print_freq == 0:
            print(f"Iter {cur_iter:04d}/{total_iters:04d}")

        inputs, labels, video_idx, meta = unpack_batch(batch)

        inputs = to_device(inputs, device)
        labels = to_device(labels, device)

        keypoints, npy_paths = keypoint_provider.make_batch_keypoints(
            labels=labels,
            video_idx=video_idx,
            meta=meta,
            inputs=inputs,
            global_sample_start=sample_counter,
            device=device,
        )

        video_idx_list = get_video_idx_list(
            video_idx,
            get_batch_size_from_labels(labels),
        )

        with torch.cuda.amp.autocast(
            enabled=bool(use_amp and device.type == "cuda")
        ):
            preds = maybe_call_model(model, inputs, keypoints)

        probs = safe_softmax(preds)

        probs_np = tensor_to_numpy(probs)
        labels_np = tensor_to_numpy(labels).reshape(-1)

        batch_size = len(labels_np)

        for i in range(batch_size):
            if video_idx_list[i] is not None:
                vid = str(video_idx_list[i])
            else:
                vid = str(sample_counter + i)

            label_i = int(labels_np[i])

            video_probs[vid].append(probs_np[i])
            video_labels[vid] = label_i

            if i < len(npy_paths):
                video_npy_paths[vid] = npy_paths[i]

        sample_counter += batch_size

    if len(video_probs) == 0:
        raise RuntimeError("No predictions were collected. Please check test loader.")

    rows = []
    y_true = []
    y_pred = []
    y_prob = []

    sorted_vids = sorted(
        video_probs.keys(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    for vid in sorted_vids:
        probs_list = video_probs[vid]
        avg_prob = np.mean(np.stack(probs_list, axis=0), axis=0)

        label = int(video_labels[vid])
        pred = int(np.argmax(avg_prob))

        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(avg_prob)

        rows.append(
            {
                "video_id": vid,
                "label": label,
                "pred": pred,
                "correct": bool(label == pred),
                "probs": avg_prob.tolist(),
                "npy_path": video_npy_paths.get(vid, ""),
            }
        )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    inferred_num_classes = int(max(y_true.max(), y_pred.max(), y_prob.shape[1] - 1) + 1)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(inferred_num_classes)]

    num_classes = max(inferred_num_classes, len(class_names))

    if len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if HAS_SKLEARN:
        conf_mat = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        )
    else:
        conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            conf_mat[int(t), int(p)] += 1

    metrics = build_basic_metrics(y_true, y_pred)

    roc_pr_summary = save_multiclass_roc_pr(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        out_dir=out_dir,
    )

    metrics.update(
        {
            "num_classes": int(num_classes),
            "class_names": list(class_names),
            "output_dir": str(out_dir),
            "test_csv": str(keypoint_provider.test_csv),
            "keypoint_dir": str(keypoint_provider.keypoint_dir),
            "missing_keypoints_count": int(len(keypoint_provider.missing_records)),
            **roc_pr_summary,
        }
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    np.save(out_dir / "confusion_matrix.npy", conf_mat)

    save_confusion_matrix_csv(
        out_dir / "confusion_matrix.csv",
        conf_mat,
        class_names,
    )

    save_confusion_matrix_png(
        out_dir / "confusion_matrix.png",
        conf_mat,
        class_names=class_names,
        normalize=False,
    )

    save_confusion_matrix_png(
        out_dir / "confusion_matrix_normalized.png",
        conf_mat,
        class_names=class_names,
        normalize=True,
    )

    save_per_class_metrics_csv(
        out_dir / "per_class_metrics.csv",
        y_true,
        y_pred,
        conf_mat,
        class_names,
    )

    save_predictions_csv(
        out_dir / "predictions.csv",
        rows,
        class_names,
    )

    save_json(
        out_dir / "metrics.json",
        metrics,
    )

    keypoint_provider.save_missing_records(out_dir)

    if HAS_SKLEARN:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )

        report_text = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )

        save_json(out_dir / "classification_report.json", report_dict)
        save_report_txt(out_dir / "classification_report.txt", report_text)
    else:
        report_text = (
            "sklearn is not installed, so classification_report is unavailable.\n"
            f"accuracy: {metrics['accuracy']:.6f}\n"
        )
        save_report_txt(out_dir / "classification_report.txt", report_text)

    print("=" * 80)
    print("Evaluation finished.")
    print(f"Videos:     {metrics['num_samples']}")
    print(f"Accuracy:   {metrics['accuracy']:.6f}")

    if "macro_precision" in metrics:
        print(f"Precision:  {metrics['macro_precision']:.6f}")
        print(f"Recall:     {metrics['macro_recall']:.6f}")
        print(f"Macro F1:   {metrics['macro_f1']:.6f}")

    if metrics.get("roc_auc_macro") is not None:
        print(f"ROC AUC macro: {metrics['roc_auc_macro']:.6f}")

    if metrics.get("pr_ap_macro") is not None:
        print(f"PR AP macro:   {metrics['pr_ap_macro']:.6f}")

    print(f"Missing keypoints: {len(keypoint_provider.missing_records)}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    return metrics, conf_mat, rows


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    out_dir = ensure_dir(args.output_dir)

    use_gpu = bool((not args.no_gpu) and torch.cuda.is_available())
    device = torch.device("cuda" if use_gpu else "cpu")

    print("=" * 80)
    print("Independent RGB + 10D Keypoint Fusion Video-level Test")
    print("=" * 80)
    print(f"Project:      {PROJECT_ROOT}")
    print(f"Split:        {args.split}")
    print(f"Config:       {args.cfg}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Test CSV:     {args.test_csv}")
    print(f"Keypoint dir: {args.keypoint_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Use GPU:      {use_gpu}")
    print("=" * 80)

    cfg = setup_cfg(args)

    keypoint_provider = KeypointProvider(
        test_csv=args.test_csv,
        keypoint_dir=args.keypoint_dir,
        feature_dim=10,
    )

    model = build_network(cfg, device)
    model = load_checkpoint_flexible(model, args.checkpoint, device)
    model.to(device)

    test_loader = build_test_loader(cfg, args.split)

    cfg_num_classes = get_num_classes_from_cfg(cfg)
    class_names = get_class_names(cfg, cfg_num_classes)

    evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        out_dir=out_dir,
        keypoint_provider=keypoint_provider,
        class_names=class_names,
        print_freq=args.print_freq,
        use_amp=args.amp,
    )


if __name__ == "__main__":
    main()