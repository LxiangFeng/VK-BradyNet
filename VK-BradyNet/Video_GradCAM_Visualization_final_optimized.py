# -*- coding: utf-8 -*-
"""
HOC-Video Grad-CAM + Hand Keypoint Fusion Visualization
Final Complete Version

功能：
1. 从测试集 class_0 / class_1 / class_2 每类随机选择 20 个样本，总计最多 60 个。
2. 自动读取本地 C:\\Users\\Administrator\\Desktop\\npy\\test 中对应 .npy。
3. 支持 RGB + 10D Keypoint Fusion MViT 模型。
4. 支持 MViT token Grad-CAM: [B, N, C]。
5. 优先读取 test.csv 中原始视频路径，使用原始视频帧作为 RAW 背景。
6. 保持原始视频宽高比例，不再强制正方形。
7. 如果无法找到原始视频，则自动退回模型输入 tensor 可视化。
8. 输出 RGB Grad-CAM、21关键点归因、10D特征归因、融合 panel、MP4、CSV、JSON。

重要参数：
- CAM_ORIGINAL_MODE = "center_crop":
    更严谨，只把 CAM 贴到模型中心 crop 区域，并画白框。
- CAM_ORIGINAL_MODE = "stretch_full_frame":
    把 CAM 拉伸到整张原始视频帧，视觉上更满。
"""

import sys
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Administrator\Desktop\SlowFast_PD_10D_speed_gated_optimized\SlowFast-main"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
import slowfast.datasets.loader as loader

from pd_mvitv2_s_repro_pack.test_rgb_keypoint_fusion_metrics import (
    KeypointProvider,
    unpack_batch,
    maybe_call_model,
    to_device,
    tensor_to_numpy,
    get_batch_size_from_labels,
    get_video_idx_list,
    load_checkpoint_flexible,
)


# =============================================================================
# Paths
# =============================================================================

CFG_YAML = (
    PROJECT_ROOT
    / "pd_mvitv2_s_repro_pack"
    / "MVITv2_S_PD_3cls_rgb_keypoint_fusion.yaml"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "PD_MVITv2_S_RGB_Keypoint_Fusion"
    / "checkpoints"
    / "checkpoint_epoch_00030.pyth"
)

TEST_CSV = Path(r"C:\Users\Administrator\Desktop\data\data\test.csv")
KEYPOINT_DIR = Path(r"C:\Users\Administrator\Desktop\data\npy\test")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "FT_gradcam_keypoint_fused_random60_original_video_final"
)


# =============================================================================
# Sampling params
# =============================================================================

TARGET_CLASSES = [0, 1, 2]
SAMPLES_PER_CLASS = 20
RANDOM_SEED = 2026


# =============================================================================
# Visualization params
# =============================================================================

FPS = 8
ALPHA = 0.55
TARGET_BLOCK_FROM_END = 4

# 原始视频可视化长边尺寸。
# 768：保持比例，把长边缩放到 768。
# None：不缩放，保持原视频帧原始尺寸。
VIS_LONG_SIDE = 768

# True：优先读取 test.csv 中的原始视频帧。
# False：只使用模型输入 tensor，可视化会是模型裁剪后的形状。
USE_ORIGINAL_VIDEO_FRAMES = True

# center_crop：把 CAM 贴到原始视频的中心正方形 crop 区域，更严谨。
# stretch_full_frame：把 CAM 拉伸到整张原始视频帧，视觉更满。
CAM_ORIGINAL_MODE = "stretch_full_frame"

# center_crop 模式下是否画出模型中心 crop 区域白框。
DRAW_MODEL_CROP_BOX = False

HEATMAP_LOW_PERCENTILE = 5
HEATMAP_HIGH_PERCENTILE = 99
HEATMAP_GAMMA = 0.55

ENABLE_INPUT_GRADIENT_FALLBACK = True
CAM_VALID_STD_THRESHOLD = 1e-5
CAM_VALID_MAX_THRESHOLD = 1e-6


# =============================================================================
# 21 hand landmark connections
# =============================================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 关键点编号显示参数：居中显示在圆圈中，带白色描边，避免偏移和看不清。
KEYPOINT_LABEL_FONT_SIZE = 8
KEYPOINT_LABEL_STROKE_WIDTH = 1.5
KEYPOINT_LABEL_COLOR = "black"
KEYPOINT_LABEL_STROKE_COLOR = "white"


# =============================================================================
# Basic utils
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hoc_class_name(class_idx):
    return f"FT_score_{int(class_idx)}"


def hoc_short_class_name(class_idx):
    return f"FT{int(class_idx)}"


def normalize_map_minmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(x.min())
    mx = float(x.max())

    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn + 1e-8)


def enhance_heatmap(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

    enhanced = np.zeros_like(cam, dtype=np.float32)

    for t in range(cam.shape[0]):
        m = cam[t]

        lo = np.percentile(m, HEATMAP_LOW_PERCENTILE)
        hi = np.percentile(m, HEATMAP_HIGH_PERCENTILE)

        if hi - lo < 1e-8:
            enhanced[t] = normalize_map_minmax(m)
            continue

        m = np.clip(m, lo, hi)
        m = (m - lo) / (hi - lo + 1e-8)
        m = np.power(m, HEATMAP_GAMMA)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=1.2)

        enhanced[t] = normalize_map_minmax(m)

    return enhanced


def is_cam_valid(cam):
    cam = np.asarray(cam, dtype=np.float32)
    return bool(
        cam.max() > CAM_VALID_MAX_THRESHOLD
        and cam.std() > CAM_VALID_STD_THRESHOLD
    )


def make_even(x):
    """
    MP4 编码更稳定：宽高设置为偶数。
    """
    x = int(round(float(x)))

    if x < 2:
        x = 2

    if x % 2 == 1:
        x -= 1

    return max(2, x)


def compute_aspect_size(h, w, long_side=768):
    """
    保持宽高比例。
    long_side=768 表示长边缩放到 768。
    long_side=None 表示不缩放。
    """
    h = int(h)
    w = int(w)

    if long_side is None:
        return make_even(w), make_even(h)

    long_side = int(long_side)

    if w >= h:
        new_w = long_side
        new_h = h * long_side / max(w, 1)
    else:
        new_h = long_side
        new_w = w * long_side / max(h, 1)

    return make_even(new_w), make_even(new_h)


def resize_for_visualization(frame_bgr, heatmap=None, long_side=768):
    """
    保持原视频宽高比例，不强制正方形。
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    new_w, new_h = compute_aspect_size(orig_h, orig_w, long_side)

    frame_resized = cv2.resize(
        frame_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    if heatmap is None:
        return frame_resized, None

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap_resized = normalize_map_minmax(heatmap_resized)

    return frame_resized, heatmap_resized


def find_first_tensor(obj):
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in obj:
            t = find_first_tensor(item)
            if t is not None:
                return t

    if isinstance(obj, dict):
        for _, item in obj.items():
            t = find_first_tensor(item)
            if t is not None:
                return t

    return None


def extract_thw_from_output(output):
    if not isinstance(output, (list, tuple)):
        return None

    if len(output) < 2:
        return None

    thw = output[1]

    if torch.is_tensor(thw):
        thw = thw.detach().cpu().tolist()

    if isinstance(thw, np.ndarray):
        thw = thw.tolist()

    if isinstance(thw, (list, tuple)) and len(thw) >= 3:
        try:
            return int(thw[0]), int(thw[1]), int(thw[2])
        except Exception:
            return None

    return None


def extract_video_tensor(inputs):
    x = inputs

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("inputs is empty.")
        x = x[0]

    if not torch.is_tensor(x):
        raise TypeError(f"Cannot extract video tensor from type: {type(x)}")

    if x.dim() != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(x.shape)}")

    return x


def replace_video_tensor(inputs, new_tensor):
    if isinstance(inputs, list):
        out = list(inputs)
        out[0] = new_tensor
        return out

    if isinstance(inputs, tuple):
        out = list(inputs)
        out[0] = new_tensor
        return tuple(out)

    return new_tensor


def get_cfg_mean_std(cfg):
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    try:
        if hasattr(cfg, "DATA") and hasattr(cfg.DATA, "MEAN"):
            mean = list(cfg.DATA.MEAN)

        if hasattr(cfg, "DATA") and hasattr(cfg.DATA, "STD"):
            std = list(cfg.DATA.STD)
    except Exception:
        pass

    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def video_tensor_to_uint8_frames(video_tensor_4d, cfg):
    """
    video_tensor_4d: [C,T,H,W]
    return: list of BGR uint8 frames
    """
    x = video_tensor_4d.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 3, 0))  # [T,H,W,C]

    mean, std = get_cfg_mean_std(cfg)

    if x.min() < -0.05 or x.max() > 1.2:
        x = x * std.reshape(1, 1, 1, 3) + mean.reshape(1, 1, 1, 3)

    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)

    frames_bgr = []

    for frame_rgb in x:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frames_bgr.append(frame_bgr)

    return frames_bgr


# =============================================================================
# Original video loading
# =============================================================================

def normalize_possible_path_string(s):
    s = str(s).strip()
    s = s.strip('"').strip("'")
    s = s.replace("/", "\\")
    return s


def resolve_video_path_from_entry(entry):
    """
    从 test.csv 的一行 entry 中解析原始视频路径。

    兼容常见字段名：
    video_path/path/file_path/filepath/filename/video/video_name
    如果字段名不匹配，会扫描所有字段中以视频后缀结尾的字符串。
    """
    if entry is None:
        return None

    candidate = None

    preferred_keys = [
        "video_path",
        "path",
        "file_path",
        "filepath",
        "filename",
        "file",
        "video",
        "video_name",
        "video_file",
        "clip_path",
    ]

    if isinstance(entry, dict):
        for key in preferred_keys:
            if key in entry and entry[key] is not None and str(entry[key]).strip() != "":
                candidate = str(entry[key]).strip()
                break

        if candidate is None:
            for _, value in entry.items():
                s = str(value).strip()
                if s.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg")):
                    candidate = s
                    break
    else:
        try:
            for value in entry:
                s = str(value).strip()
                if s.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg")):
                    candidate = s
                    break
        except Exception:
            pass

    if candidate is None:
        return None

    candidate = normalize_possible_path_string(candidate)
    p = Path(candidate)

    if p.exists():
        return p

    # 如果 CSV 里写的是相对路径，则尝试多个常见根目录。
    bases = [
        TEST_CSV.parent,
        TEST_CSV.parent.parent,
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT.parent.parent,
        KEYPOINT_DIR.parent,
        KEYPOINT_DIR.parent.parent,
        Path(r"C:\Users\Administrator\Desktop"),
        Path(r"C:\Users\Administrator\Desktop\data"),
    ]

    for base in bases:
        q = base / candidate
        if q.exists():
            return q

    # 尝试仅用文件名在常见目录下找
    file_name = Path(candidate).name
    search_roots = [
        TEST_CSV.parent,
        TEST_CSV.parent.parent,
        Path(r"C:\Users\Administrator\Desktop"),
    ]

    for root in search_roots:
        try:
            matches = list(root.rglob(file_name))
            if len(matches) > 0:
                return matches[0]
        except Exception:
            pass

    return p


def load_original_video_frames_uniform(video_path, target_num_frames):
    """
    从原始视频中均匀采样 target_num_frames 帧。
    返回 BGR uint8 frames。
    """
    if video_path is None:
        return None

    video_path = Path(video_path)

    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)

        cap.release()

        if len(all_frames) == 0:
            return None

        if target_num_frames <= 1:
            return [all_frames[len(all_frames) // 2]]

        idxs = np.linspace(0, len(all_frames) - 1, target_num_frames)
        idxs = np.clip(np.round(idxs).astype(int), 0, len(all_frames) - 1)

        return [all_frames[i] for i in idxs]

    if target_num_frames <= 1:
        idxs = [frame_count // 2]
    else:
        idxs = np.linspace(0, frame_count - 1, target_num_frames)
        idxs = np.clip(np.round(idxs).astype(int), 0, frame_count - 1)

    frames = []
    last_good = None

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if ret and frame is not None:
            last_good = frame
            frames.append(frame)
        elif last_good is not None:
            frames.append(last_good.copy())

    cap.release()

    if len(frames) == 0:
        return None

    return frames


# =============================================================================
# Overlay utils
# =============================================================================

def overlay_heatmap_on_frame(frame_bgr, heatmap, alpha=0.55):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(255 * heatmap)

    try:
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    except Exception:
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

    return overlay, heatmap_color


def overlay_cam_on_original_frame(
    frame_bgr,
    heatmap,
    alpha=0.55,
    long_side=768,
    mode="stretch_full_frame",
):
    """
    在原始长方形视频帧上叠加 CAM。

    mode:
    - center_crop:
        更严谨。模型通常看到的是中心方形 crop，
        因此只把 CAM 贴到中心 crop 区域。
    - stretch_full_frame:
        把 CAM 拉伸到整张原始视频帧，视觉更满。
    """
    frame_vis, _ = resize_for_visualization(
        frame_bgr,
        heatmap=None,
        long_side=long_side,
    )

    vis_h, vis_w = frame_vis.shape[:2]

    if mode == "stretch_full_frame":
        heatmap_vis = cv2.resize(
            heatmap.astype(np.float32),
            (vis_w, vis_h),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmap_vis = normalize_map_minmax(heatmap_vis)

        overlay, heatmap_color = overlay_heatmap_on_frame(
            frame_bgr=frame_vis,
            heatmap=heatmap_vis,
            alpha=alpha,
        )

        return frame_vis, heatmap_color, overlay

    # center_crop mode
    crop_size = min(vis_h, vis_w)
    x0 = int((vis_w - crop_size) / 2)
    y0 = int((vis_h - crop_size) / 2)
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    heatmap_crop = cv2.resize(
        heatmap.astype(np.float32),
        (crop_size, crop_size),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap_crop = normalize_map_minmax(heatmap_crop)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_crop, 0.0, 1.0))

    try:
        heatmap_crop_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    except Exception:
        heatmap_crop_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    heatmap_color_full = np.zeros_like(frame_vis)
    heatmap_color_full[y0:y1, x0:x1] = heatmap_crop_color

    overlay = frame_vis.copy()
    roi = overlay[y0:y1, x0:x1]

    roi_overlay = cv2.addWeighted(
        roi,
        1.0 - alpha,
        heatmap_crop_color,
        alpha,
        0,
    )

    overlay[y0:y1, x0:x1] = roi_overlay

    if DRAW_MODEL_CROP_BOX:
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1 - 1, y1 - 1),
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            heatmap_color_full,
            (x0, y0),
            (x1 - 1, y1 - 1),
            (255, 255, 255),
            2,
        )

    return frame_vis, heatmap_color_full, overlay


def draw_label_banner(frame_bgr, text):
    """
    仅用于 overlay_frames 和 MP4。
    fused panel 使用无 banner overlay，避免遮挡。
    """
    frame = frame_bgr.copy()

    h, w = frame.shape[:2]
    banner_h = max(30, int(h * 0.09))

    cv2.rectangle(frame, (0, 0), (w, banner_h), (255, 255, 255), -1)

    font_scale = 0.43 if max(h, w) <= 512 else 0.55

    cv2.putText(
        frame,
        text,
        (10, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    return frame


def save_video(frames_bgr, path, fps=8):
    path = Path(path)

    if len(frames_bgr) == 0:
        return

    h, w = frames_bgr[0].shape[:2]

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame in frames_bgr:
        writer.write(frame)

    writer.release()


# =============================================================================
# Random balanced selection
# =============================================================================

def infer_label_from_entry_or_npy(keypoint_provider, idx, entry):
    label = None

    if isinstance(entry, dict):
        label = entry.get("label", None)

    if label is not None:
        try:
            return int(label)
        except Exception:
            pass

    if isinstance(entry, dict):
        video_path = str(entry.get("video_path", "")).replace("\\", "/")
    else:
        video_path = str(entry).replace("\\", "/")

    parts = video_path.split("/")

    for p in parts:
        if p in ["0", "1", "2"]:
            return int(p)

    try:
        npy_path = keypoint_provider._find_npy(video_idx=idx, fallback_pos=idx)
        if npy_path is not None:
            parent_name = Path(npy_path).parent.name
            if parent_name in ["0", "1", "2"]:
                return int(parent_name)
    except Exception:
        pass

    return None


def select_random_indices_by_class(keypoint_provider):
    rng = random.Random(RANDOM_SEED)

    pools = {int(c): [] for c in TARGET_CLASSES}

    for idx, entry in enumerate(keypoint_provider.csv_entries):
        label = infer_label_from_entry_or_npy(
            keypoint_provider=keypoint_provider,
            idx=idx,
            entry=entry,
        )

        if label in pools:
            pools[int(label)].append(idx)

    selected = {}
    selected_set = set()

    for cls in TARGET_CLASSES:
        candidates = pools[int(cls)]
        rng.shuffle(candidates)

        n = min(SAMPLES_PER_CLASS, len(candidates))
        selected[int(cls)] = candidates[:n]

        for idx in selected[int(cls)]:
            selected_set.add(int(idx))

    selected_info = {
        "random_seed": int(RANDOM_SEED),
        "target_classes": [int(c) for c in TARGET_CLASSES],
        "samples_per_class": int(SAMPLES_PER_CLASS),
        "num_selected_total": int(len(selected_set)),
        "class_counts_available": {
            str(cls): int(len(pools[int(cls)])) for cls in TARGET_CLASSES
        },
        "selected_indices_by_class": {
            str(cls): [int(x) for x in selected[int(cls)]]
            for cls in TARGET_CLASSES
        },
    }

    return selected, selected_set, selected_info


# =============================================================================
# Video Grad-CAM
# =============================================================================

class VideoGradCAM:
    """
    支持：
    - [B,C,T,H,W]
    - [B,N,C] MViT token
    - [B,C,H,W]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.thw = None

        self.forward_handle = None
        self.backward_handle = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, module_input, module_output):
            act = find_first_tensor(module_output)

            if act is None:
                raise RuntimeError("Forward hook did not find tensor activation.")

            self.activations = act
            self.thw = extract_thw_from_output(module_output)

        def backward_hook(module, grad_input, grad_output):
            grad = find_first_tensor(grad_output)

            if grad is None:
                raise RuntimeError("Backward hook did not find tensor gradient.")

            self.gradients = grad

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)

        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()

        if self.backward_handle is not None:
            self.backward_handle.remove()

    def _cam_from_5d(self, activations, gradients):
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        raw = (weights * activations).sum(dim=1)

        cam_relu = torch.relu(raw)

        if float(cam_relu.detach().max().cpu()) < 1e-8:
            return torch.abs(raw), "abs_fallback_5d"

        return cam_relu, "relu_5d"

    def _infer_thw(self, token_count, target_t):
        if self.thw is not None:
            return self.thw

        if target_t <= 0:
            target_t = 1

        if token_count % target_t == 0:
            spatial = token_count // target_t
            side = int(round(spatial ** 0.5))
            if side * side == spatial:
                return target_t, side, side

        n_no_cls = token_count - 1
        if n_no_cls > 0 and n_no_cls % target_t == 0:
            spatial = n_no_cls // target_t
            side = int(round(spatial ** 0.5))
            if side * side == spatial:
                return target_t, side, side

        side = int(round(token_count ** 0.5))
        if side * side == token_count:
            return 1, side, side

        side = int(round((token_count - 1) ** 0.5))
        if token_count > 1 and side * side == token_count - 1:
            return 1, side, side

        raise RuntimeError(
            f"Cannot infer THW from token_count={token_count}, target_t={target_t}. "
            f"Try changing TARGET_BLOCK_FROM_END."
        )

    def _cam_from_3d_tokens(self, activations, gradients, target_t):
        b, n, c = activations.shape

        thw = self._infer_thw(n, target_t)
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        token_count = t * h * w

        if n == token_count + 1:
            act = activations[:, 1:, :]
            grad = gradients[:, 1:, :]
        elif n == token_count:
            act = activations
            grad = gradients
        elif n > token_count:
            act = activations[:, -token_count:, :]
            grad = gradients[:, -token_count:, :]
        else:
            raise RuntimeError(
                f"Token mismatch: N={n}, expected={token_count}, thw={thw}"
            )

        act = act.reshape(b, t, h, w, c)
        grad = grad.reshape(b, t, h, w, c)

        weights = grad.mean(dim=(1, 2, 3), keepdim=True)
        raw = (weights * act).sum(dim=-1)

        cam_relu = torch.relu(raw)

        if float(cam_relu.detach().max().cpu()) < 1e-8:
            return torch.abs(raw), "abs_fallback_3d_token"

        return cam_relu, "relu_3d_token"

    def _cam_from_4d(self, activations, gradients, target_t):
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        raw = (weights * activations).sum(dim=1)

        cam_relu = torch.relu(raw)

        if float(cam_relu.detach().max().cpu()) < 1e-8:
            cam2d = torch.abs(raw)
            mode = "abs_fallback_4d"
        else:
            cam2d = cam_relu
            mode = "relu_4d"

        cam = cam2d.unsqueeze(1).repeat(1, target_t, 1, 1)

        return cam, mode

    def compute_cam(self, target_class, inputs, keypoints):
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None
        self.thw = None

        video_tensor = extract_video_tensor(inputs)
        target_t = int(video_tensor.shape[2])
        target_h = int(video_tensor.shape[3])
        target_w = int(video_tensor.shape[4])

        logits = maybe_call_model(self.model, inputs, keypoints)

        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        score = logits[0, int(target_class)]
        score.backward(retain_graph=True)

        if self.activations is None:
            raise RuntimeError("Grad-CAM activations are None.")

        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradients are None.")

        activations = self.activations
        gradients = self.gradients

        if activations.shape != gradients.shape:
            raise RuntimeError(
                f"Activation/gradient shape mismatch: "
                f"{tuple(activations.shape)} vs {tuple(gradients.shape)}"
            )

        if activations.dim() == 5:
            cam, cam_mode = self._cam_from_5d(activations, gradients)

        elif activations.dim() == 3:
            cam, cam_mode = self._cam_from_3d_tokens(
                activations,
                gradients,
                target_t=target_t,
            )

        elif activations.dim() == 4:
            cam, cam_mode = self._cam_from_4d(
                activations,
                gradients,
                target_t=target_t,
            )

        else:
            raise RuntimeError(
                f"Unsupported activation shape: {tuple(activations.shape)}"
            )

        cam = cam.unsqueeze(1)
        cam = F.interpolate(
            cam,
            size=(target_t, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze(1)

        cam0_raw = cam[0].detach().cpu().float().numpy()
        cam0_norm = normalize_map_minmax(cam0_raw)
        cam0_enhanced = enhance_heatmap(cam0_norm)

        debug_stats = {
            "activation_shape": list(activations.shape),
            "gradient_shape": list(gradients.shape),
            "captured_thw": list(self.thw) if self.thw is not None else None,
            "cam_mode": cam_mode,
            "cam_raw_min": float(cam0_raw.min()),
            "cam_raw_max": float(cam0_raw.max()),
            "cam_raw_std": float(cam0_raw.std()),
            "cam_enhanced_min": float(cam0_enhanced.min()),
            "cam_enhanced_max": float(cam0_enhanced.max()),
            "cam_enhanced_std": float(cam0_enhanced.std()),
            "used_input_gradient_fallback": False,
        }

        return cam0_enhanced, logits.detach(), debug_stats


def compute_input_gradient_heatmap(model, inputs, keypoints, target_class):
    model.zero_grad(set_to_none=True)

    video_tensor = extract_video_tensor(inputs)
    video_grad = video_tensor.detach().clone()
    video_grad.requires_grad_(True)

    inputs_for_grad = replace_video_tensor(inputs, video_grad)

    logits = maybe_call_model(model, inputs_for_grad, keypoints)

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    score = logits[0, int(target_class)]
    score.backward()

    if video_grad.grad is None:
        raise RuntimeError("video input gradient is None.")

    grad = video_grad.grad.detach().abs()
    sal = grad * video_grad.detach().abs()
    sal = sal.mean(dim=1)[0]

    sal = sal.detach().cpu().float().numpy()
    sal = normalize_map_minmax(sal)
    sal = enhance_heatmap(sal)

    return sal


# =============================================================================
# Keypoint attribution
# =============================================================================

def compute_keypoint_attribution(model, inputs, keypoints, target_class):
    keypoints_for_grad = keypoints.detach().clone()
    keypoints_for_grad.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = maybe_call_model(model, inputs, keypoints_for_grad)

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    score = logits[0, int(target_class)]
    score.backward()

    if keypoints_for_grad.grad is None:
        raise RuntimeError("keypoints.grad is None.")

    grad = keypoints_for_grad.grad.detach()
    kp = keypoints_for_grad.detach()

    attr = torch.abs(grad * kp)

    if attr.dim() == 3:
        token_feature_attr = attr[0].detach().cpu().float().numpy()
    elif attr.dim() == 2:
        token_feature_attr = attr.detach().cpu().float().numpy()
    else:
        token_feature_attr = attr.reshape(attr.shape[0], -1).detach().cpu().float().numpy()

    temporal_saliency = token_feature_attr.mean(axis=1)
    feature_importance = token_feature_attr.mean(axis=0)

    temporal_saliency = normalize_map_minmax(temporal_saliency)
    feature_importance = normalize_map_minmax(feature_importance)

    return temporal_saliency, feature_importance, token_feature_attr


def load_raw_keypoints(npy_path):
    if npy_path is None or str(npy_path) == "":
        return None

    path = Path(npy_path)

    if not path.exists():
        return None

    arr = np.load(str(path), allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)

    return arr


def map_temporal_saliency_to_21_landmarks(raw_keypoints, temporal_saliency):
    if raw_keypoints is None:
        return None

    raw = np.asarray(raw_keypoints, dtype=np.float32)

    if raw.ndim != 3 or raw.shape[1] != 21:
        return None

    raw_t, raw_l, _ = raw.shape
    total_tokens = raw_t * raw_l

    temporal_saliency = np.asarray(temporal_saliency, dtype=np.float32)
    model_l = len(temporal_saliency)

    if model_l <= 1:
        return np.ones(21, dtype=np.float32) / 21.0

    model_positions = np.linspace(0, total_tokens - 1, model_l)
    raw_positions = np.arange(total_tokens)

    interp_scores = np.interp(raw_positions, model_positions, temporal_saliency)
    interp_scores = interp_scores.reshape(raw_t, raw_l)

    landmark_scores = interp_scores.mean(axis=0)
    landmark_scores = normalize_map_minmax(landmark_scores)

    return landmark_scores.astype(np.float32)


def get_representative_raw_frame_index(raw_keypoints, temporal_saliency):
    if raw_keypoints is None:
        return 0

    raw = np.asarray(raw_keypoints)

    if raw.ndim != 3 or raw.shape[1] != 21:
        return 0

    raw_t = raw.shape[0]
    total_tokens = raw_t * 21

    temporal_saliency = np.asarray(temporal_saliency, dtype=np.float32)
    model_l = len(temporal_saliency)

    best_model_idx = int(np.argmax(temporal_saliency))

    if model_l <= 1:
        return raw_t // 2

    raw_pos = int(round(best_model_idx / (model_l - 1) * (total_tokens - 1)))
    raw_frame_idx = raw_pos // 21
    raw_frame_idx = int(np.clip(raw_frame_idx, 0, raw_t - 1))

    return raw_frame_idx


def get_representative_video_frame_index(num_video_frames, temporal_saliency):
    temporal_saliency = np.asarray(temporal_saliency, dtype=np.float32)

    if len(temporal_saliency) <= 1:
        return num_video_frames // 2

    best_idx = int(np.argmax(temporal_saliency))
    video_idx = int(round(best_idx / (len(temporal_saliency) - 1) * (num_video_frames - 1)))
    video_idx = int(np.clip(video_idx, 0, num_video_frames - 1))

    return video_idx


def canonical_hand_xy(raw_keypoints, frame_idx):
    if raw_keypoints is None:
        return None

    raw = np.asarray(raw_keypoints, dtype=np.float32)

    if raw.ndim != 3 or raw.shape[1] != 21:
        return None

    frame_idx = int(np.clip(frame_idx, 0, raw.shape[0] - 1))

    xy = raw[frame_idx, :, :2].astype(np.float32)

    x = xy[:, 0]
    y = xy[:, 1]

    if np.max(x) - np.min(x) < 1e-8:
        return None

    if np.max(y) - np.min(y) < 1e-8:
        return None

    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

    y = 1.0 - y

    xy_norm = np.stack([x, y], axis=1)

    return xy_norm


# =============================================================================
# Plotting
# =============================================================================

def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_temporal_saliency_plot(path, saliency, title):
    set_paper_style()

    plt.figure(figsize=(7.0, 3.5))
    plt.plot(np.arange(len(saliency)), saliency, marker="o", linewidth=1.8)
    plt.xlabel("Model keypoint token")
    plt.ylabel("Normalized attribution")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot(path, feature_importance):
    set_paper_style()

    x = np.arange(len(feature_importance))

    plt.figure(figsize=(7.0, 3.5))
    plt.bar(x, feature_importance)
    plt.xlabel("Keypoint feature dimension")
    plt.ylabel("Normalized attribution")
    plt.title("10D Keypoint Feature Attribution")
    plt.xticks(x, [f"D{i}" for i in x])
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def draw_hand_skeleton_on_axis(ax, xy, scores, title):
    """
    绘制 21 个手部关键点归因图。

    优化点：
    1. 关键点编号严格居中显示在圆圈中心。
    2. 去掉原来的 +0.012 偏移，避免编号整体向右上偏。
    3. 编号加白色描边，红色/蓝色/绿色圆圈上都更清晰。
    4. 连线、圆点、文字设置 zorder，避免互相遮挡。
    """
    ax.set_title(title, fontweight="bold", pad=10)

    if xy is None:
        ax.text(
            0.5,
            0.5,
            "No valid 21-landmark coordinates",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return None

    scores = np.asarray(scores, dtype=np.float32)
    scores = normalize_map_minmax(scores)

    # 先画骨架连线
    for a, b in HAND_CONNECTIONS:
        ax.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            linewidth=1.6,
            color="#666666",
            alpha=0.75,
            zorder=1,
        )

    # 再画关键点圆圈。归因越高，圆圈越大。
    sizes = 80 + 520 * scores

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=scores,
        s=sizes,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    # 最后画关键点编号：严格居中，带白色描边。
    for i in range(21):
        ax.text(
            xy[i, 0],
            xy[i, 1],
            str(i),
            ha="center",
            va="center",
            fontsize=KEYPOINT_LABEL_FONT_SIZE,
            fontweight="bold",
            color=KEYPOINT_LABEL_COLOR,
            zorder=5,
            path_effects=[
                pe.withStroke(
                    linewidth=KEYPOINT_LABEL_STROKE_WIDTH,
                    foreground=KEYPOINT_LABEL_STROKE_COLOR,
                )
            ],
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Canonical hand x")
    ax.set_ylabel("Canonical hand y")

    return sc


def save_skeleton_importance_plot(path, raw_keypoints, landmark_scores, frame_idx, title):
    set_paper_style()

    xy = canonical_hand_xy(raw_keypoints, frame_idx)

    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=300)
    sc = draw_hand_skeleton_on_axis(ax, xy, landmark_scores, title)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Landmark attribution")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_fused_panel(
    path,
    raw_frame_bgr,
    overlay_frame_bgr,
    raw_keypoints,
    landmark_scores,
    feature_importance,
    raw_frame_idx,
    title,
):
    """
    保持原始视频帧比例的 fused panel。
    """
    set_paper_style()

    raw_frame_bgr, _ = resize_for_visualization(
        raw_frame_bgr,
        heatmap=None,
        long_side=VIS_LONG_SIDE,
    )

    overlay_frame_bgr, _ = resize_for_visualization(
        overlay_frame_bgr,
        heatmap=None,
        long_side=VIS_LONG_SIDE,
    )

    raw_rgb = cv2.cvtColor(raw_frame_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_frame_bgr, cv2.COLOR_BGR2RGB)

    img_h, img_w = raw_rgb.shape[:2]

    xy = canonical_hand_xy(raw_keypoints, raw_frame_idx)

    fig = plt.figure(figsize=(13.2, 10.6), dpi=300)

    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.96,
        bottom=0.07,
        top=0.90,
        wspace=0.22,
        hspace=0.32,
    )

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_skeleton = fig.add_subplot(gs[1, 0])
    ax_feature = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )

    ax_raw.imshow(raw_rgb)
    ax_raw.set_title("(a) Original frame", fontweight="bold", pad=10)
    ax_raw.set_aspect("equal")
    ax_raw.set_xlim(0, img_w)
    ax_raw.set_ylim(img_h, 0)
    ax_raw.axis("off")

    ax_cam.imshow(overlay_rgb)
    ax_cam.set_title("(b) RGB Grad-CAM overlay", fontweight="bold", pad=10)
    ax_cam.set_aspect("equal")
    ax_cam.set_xlim(0, img_w)
    ax_cam.set_ylim(img_h, 0)
    ax_cam.axis("off")

    sc = draw_hand_skeleton_on_axis(
        ax_skeleton,
        xy,
        landmark_scores,
        "(c) 21-landmark attribution",
    )

    x = np.arange(len(feature_importance))
    ax_feature.bar(x, feature_importance)
    ax_feature.set_title("(d) 10D keypoint feature attribution", fontweight="bold", pad=10)
    ax_feature.set_xlabel("Feature dimension")
    ax_feature.set_ylabel("Normalized attribution")
    ax_feature.set_xticks(x)
    ax_feature.set_xticklabels([f"D{i}" for i in x])
    ax_feature.set_ylim(0, max(1.05, float(np.max(feature_importance)) * 1.15))

    if sc is not None:
        cbar = fig.colorbar(
            sc,
            ax=ax_skeleton,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label("Attribution")

    fig.savefig(path, dpi=300)
    plt.close(fig)


# =============================================================================
# CSV saving
# =============================================================================

def save_landmark_importance_csv(path, landmark_scores):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_index", "importance"])

        for i, score in enumerate(landmark_scores):
            writer.writerow([i, float(score)])


def save_feature_importance_csv(path, feature_importance):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_dimension", "importance"])

        for i, score in enumerate(feature_importance):
            writer.writerow([i, float(score)])


def save_temporal_saliency_csv(path, temporal_saliency):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["token_index", "importance"])

        for i, score in enumerate(temporal_saliency):
            writer.writerow([i, float(score)])


# =============================================================================
# Config / model
# =============================================================================

def setup_cfg():
    if not CFG_YAML.exists():
        raise FileNotFoundError(f"Config yaml not found: {CFG_YAML}")

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(str(CFG_YAML))

    try:
        cfg.TEST.BATCH_SIZE = 1
    except Exception:
        pass

    cfg.defrost()

    return cfg


def select_target_layer(model):
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        num_blocks = len(model.blocks)
        idx = max(0, num_blocks - TARGET_BLOCK_FROM_END)
        print(f"Grad-CAM target layer: model.blocks[{idx}] / total {num_blocks}")
        return model.blocks[idx]

    if hasattr(model, "s5"):
        print("Grad-CAM target layer: model.s5")
        return model.s5

    raise RuntimeError("Cannot find target layer.")


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 80)
    print("HOC-Video Grad-CAM + Hand Keypoint Fusion Visualization")
    print("Random balanced sampling: class 0/1/2 × 20 samples")
    print("Original-video-aspect visualization")
    print("=" * 80)
    print(f"Project root:              {PROJECT_ROOT}")
    print(f"Config:                    {CFG_YAML}")
    print(f"Checkpoint:                {CHECKPOINT}")
    print(f"Test CSV:                  {TEST_CSV}")
    print(f"Keypoint dir:              {KEYPOINT_DIR}")
    print(f"Output dir:                {OUTPUT_DIR}")
    print(f"Target classes:            {TARGET_CLASSES}")
    print(f"Each class:                {SAMPLES_PER_CLASS}")
    print(f"Random seed:               {RANDOM_SEED}")
    print(f"VIS_LONG_SIDE:             {VIS_LONG_SIDE}")
    print(f"USE_ORIGINAL_VIDEO_FRAMES: {USE_ORIGINAL_VIDEO_FRAMES}")
    print(f"CAM_ORIGINAL_MODE:         {CAM_ORIGINAL_MODE}")
    print(f"DRAW_MODEL_CROP_BOX:       {DRAW_MODEL_CROP_BOX}")
    print("=" * 80)

    cfg = setup_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")

    model = build_model(cfg)
    model = model.to(device)
    model = load_checkpoint_flexible(model, CHECKPOINT, device)
    model.eval()

    keypoint_provider = KeypointProvider(
        test_csv=TEST_CSV,
        keypoint_dir=KEYPOINT_DIR,
        feature_dim=10,
    )

    selected_by_class, selected_set, selected_info = select_random_indices_by_class(
        keypoint_provider
    )

    save_json(OUTPUT_DIR / "selected_samples.json", selected_info)

    print("Selected indices:")
    for cls in TARGET_CLASSES:
        print(f"  class_{cls}: {selected_by_class[int(cls)]}")

    expected_total = len(selected_set)

    if expected_total == 0:
        raise RuntimeError(
            "No selected samples. Please check test.csv labels or npy/test class folders."
        )

    test_loader = loader.construct_loader(cfg, "test")

    target_layer = select_target_layer(model)
    gradcam = VideoGradCAM(model, target_layer)

    generated_counts = {int(c): 0 for c in TARGET_CLASSES}
    processed_indices = set()

    global_sample_start = 0

    for batch_idx, batch in enumerate(test_loader):
        if len(processed_indices) >= expected_total:
            break

        inputs, labels, video_idx, meta = unpack_batch(batch)
        batch_size = get_batch_size_from_labels(labels)

        video_idx_list = get_video_idx_list(video_idx, batch_size)

        if video_idx_list[0] is not None:
            try:
                sample_index = int(video_idx_list[0])
            except Exception:
                sample_index = int(global_sample_start)
        else:
            sample_index = int(global_sample_start)

        if sample_index not in selected_set:
            global_sample_start += batch_size
            continue

        if sample_index in processed_indices:
            global_sample_start += batch_size
            continue

        inputs = to_device(inputs, device)
        labels = to_device(labels, device)

        keypoints, npy_paths = keypoint_provider.make_batch_keypoints(
            labels=labels,
            video_idx=video_idx,
            meta=meta,
            inputs=inputs,
            global_sample_start=global_sample_start,
            device=device,
        )

        labels_np = tensor_to_numpy(labels).reshape(-1)
        true_class = int(labels_np[0])

        if true_class not in TARGET_CLASSES:
            global_sample_start += batch_size
            continue

        with torch.no_grad():
            logits_pred = maybe_call_model(model, inputs, keypoints)
            if isinstance(logits_pred, (list, tuple)):
                logits_pred = logits_pred[0]
            probs_pred = torch.softmax(logits_pred, dim=1)

        pred_class = int(torch.argmax(probs_pred[0]).item())
        pred_prob = float(probs_pred[0, pred_class].detach().cpu().item())

        # ---------------------------------------------------------------------
        # RGB Grad-CAM
        # ---------------------------------------------------------------------

        cam, logits_cam, debug_stats = gradcam.compute_cam(
            target_class=pred_class,
            inputs=inputs,
            keypoints=keypoints,
        )

        if ENABLE_INPUT_GRADIENT_FALLBACK and not is_cam_valid(cam):
            print(
                f"Warning: Grad-CAM is weak for sample index {sample_index}. "
                f"Using input-gradient fallback."
            )

            cam = compute_input_gradient_heatmap(
                model=model,
                inputs=inputs,
                keypoints=keypoints,
                target_class=pred_class,
            )

            debug_stats["used_input_gradient_fallback"] = True
            debug_stats["fallback_cam_min"] = float(cam.min())
            debug_stats["fallback_cam_max"] = float(cam.max())
            debug_stats["fallback_cam_std"] = float(cam.std())

        # ---------------------------------------------------------------------
        # Keypoint attribution
        # ---------------------------------------------------------------------

        temporal_saliency, feature_importance, token_feature_attr = compute_keypoint_attribution(
            model=model,
            inputs=inputs,
            keypoints=keypoints,
            target_class=pred_class,
        )

        raw_npy_path = str(npy_paths[0]) if len(npy_paths) > 0 else ""
        raw_keypoints = load_raw_keypoints(raw_npy_path)

        if raw_keypoints is not None:
            debug_stats["raw_keypoint_path"] = raw_npy_path
            debug_stats["raw_keypoint_shape"] = list(raw_keypoints.shape)
            debug_stats["raw_keypoint_min"] = float(np.min(raw_keypoints))
            debug_stats["raw_keypoint_max"] = float(np.max(raw_keypoints))
        else:
            debug_stats["raw_keypoint_path"] = raw_npy_path
            debug_stats["raw_keypoint_shape"] = None

        landmark_scores = map_temporal_saliency_to_21_landmarks(
            raw_keypoints=raw_keypoints,
            temporal_saliency=temporal_saliency,
        )

        if landmark_scores is None:
            landmark_scores = np.zeros(21, dtype=np.float32)
            debug_stats["landmark_scores_available"] = False
        else:
            debug_stats["landmark_scores_available"] = True

        raw_frame_idx = get_representative_raw_frame_index(
            raw_keypoints=raw_keypoints,
            temporal_saliency=temporal_saliency,
        )

        # ---------------------------------------------------------------------
        # Save outputs
        # ---------------------------------------------------------------------

        current_class_count = generated_counts[int(true_class)]
        class_dir = ensure_dir(OUTPUT_DIR / f"class_{true_class}")
        sample_dir = ensure_dir(class_dir / f"sample_{current_class_count:03d}")

        raw_dir = ensure_dir(sample_dir / "raw_frames")
        heatmap_dir = ensure_dir(sample_dir / "heatmap_frames")
        overlay_dir = ensure_dir(sample_dir / "overlay_frames")

        video_tensor = extract_video_tensor(inputs)
        model_input_frames_bgr = video_tensor_to_uint8_frames(video_tensor[0], cfg)

        original_video_path = None
        original_frames_bgr = None

        if USE_ORIGINAL_VIDEO_FRAMES:
            try:
                csv_entry = keypoint_provider.csv_entries[int(sample_index)]
                original_video_path = resolve_video_path_from_entry(csv_entry)
                original_frames_bgr = load_original_video_frames_uniform(
                    video_path=original_video_path,
                    target_num_frames=len(model_input_frames_bgr),
                )
            except Exception as e:
                print(f"Warning: failed to load original video for index {sample_index}: {e}")
                original_video_path = None
                original_frames_bgr = None

        if (
            original_frames_bgr is not None
            and len(original_frames_bgr) == len(model_input_frames_bgr)
        ):
            frames_bgr = original_frames_bgr
            visualization_source = "original_video"
        else:
            frames_bgr = model_input_frames_bgr
            visualization_source = "model_input_tensor_square_crop"

        video_frame_idx = get_representative_video_frame_index(
            num_video_frames=len(frames_bgr),
            temporal_saliency=temporal_saliency,
        )

        overlay_frames = []
        selected_raw_frame = None
        selected_overlay_frame_no_banner = None

        label_text = (
            f"T:{hoc_short_class_name(true_class)} | "
            f"P:{hoc_short_class_name(pred_class)} | "
            f"p={pred_prob:.3f}"
        )

        for t, frame_bgr in enumerate(frames_bgr):
            heatmap_t = cam[t]

            if visualization_source == "original_video":
                frame_vis, heatmap_color, overlay_no_banner = overlay_cam_on_original_frame(
                    frame_bgr=frame_bgr,
                    heatmap=heatmap_t,
                    alpha=ALPHA,
                    long_side=VIS_LONG_SIDE,
                    mode=CAM_ORIGINAL_MODE,
                )
            else:
                frame_vis, heatmap_vis = resize_for_visualization(
                    frame_bgr=frame_bgr,
                    heatmap=heatmap_t,
                    long_side=VIS_LONG_SIDE,
                )

                overlay_no_banner, heatmap_color = overlay_heatmap_on_frame(
                    frame_bgr=frame_vis,
                    heatmap=heatmap_vis,
                    alpha=ALPHA,
                )

            overlay_with_banner = draw_label_banner(overlay_no_banner, label_text)

            cv2.imwrite(str(raw_dir / f"frame_{t:03d}.png"), frame_vis)
            cv2.imwrite(str(heatmap_dir / f"frame_{t:03d}.png"), heatmap_color)
            cv2.imwrite(str(overlay_dir / f"frame_{t:03d}.png"), overlay_with_banner)

            if t == video_frame_idx:
                selected_raw_frame = frame_vis.copy()
                selected_overlay_frame_no_banner = overlay_no_banner.copy()

            overlay_frames.append(overlay_with_banner)

        save_video(
            frames_bgr=overlay_frames,
            path=sample_dir / "gradcam_overlay.mp4",
            fps=FPS,
        )

        if selected_raw_frame is None:
            selected_raw_frame = frames_bgr[0].copy()
            selected_raw_frame, _ = resize_for_visualization(
                selected_raw_frame,
                heatmap=None,
                long_side=VIS_LONG_SIDE,
            )

        if selected_overlay_frame_no_banner is None:
            selected_overlay_frame_no_banner = overlay_frames[0].copy()

        save_temporal_saliency_plot(
            path=sample_dir / "keypoint_temporal_saliency.png",
            saliency=temporal_saliency,
            title=(
                f"Keypoint Temporal Attribution "
                f"({hoc_class_name(true_class)} → {hoc_class_name(pred_class)})"
            ),
        )

        save_feature_importance_plot(
            path=sample_dir / "keypoint_feature_importance.png",
            feature_importance=feature_importance,
        )

        save_skeleton_importance_plot(
            path=sample_dir / "keypoint_skeleton_importance.png",
            raw_keypoints=raw_keypoints,
            landmark_scores=landmark_scores,
            frame_idx=raw_frame_idx,
            title=(
                f"21-Hand-Landmark Attribution "
                f"({hoc_class_name(true_class)} → {hoc_class_name(pred_class)})"
            ),
        )

        save_fused_panel(
            path=sample_dir / "fused_explanation_panel.png",
            raw_frame_bgr=selected_raw_frame,
            overlay_frame_bgr=selected_overlay_frame_no_banner,
            raw_keypoints=raw_keypoints,
            landmark_scores=landmark_scores,
            feature_importance=feature_importance,
            raw_frame_idx=raw_frame_idx,
            title=(
                f"HOC fused explanation: "
                f"True={hoc_short_class_name(true_class)}, "
                f"Pred={hoc_short_class_name(pred_class)}, "
                f"p={pred_prob:.3f}"
            ),
        )

        save_landmark_importance_csv(
            sample_dir / "keypoint_landmark_importance.csv",
            landmark_scores,
        )

        save_feature_importance_csv(
            sample_dir / "keypoint_feature_importance.csv",
            feature_importance,
        )

        save_temporal_saliency_csv(
            sample_dir / "keypoint_temporal_saliency.csv",
            temporal_saliency,
        )

        prediction_info = {
            "sample_index_in_test_csv": int(sample_index),
            "batch_index": int(batch_idx),
            "global_sample_start": int(global_sample_start),
            "true_class": int(true_class),
            "true_class_name": hoc_class_name(true_class),
            "pred_class": int(pred_class),
            "pred_class_name": hoc_class_name(pred_class),
            "pred_probability": float(pred_prob),
            "npy_path": raw_npy_path,
            "representative_video_frame_index": int(video_frame_idx),
            "representative_raw_keypoint_frame_index": int(raw_frame_idx),
            "class_sample_number": int(current_class_count),
            "vis_long_side": None if VIS_LONG_SIDE is None else int(VIS_LONG_SIDE),
            "visualization_source": visualization_source,
            "original_video_path": str(original_video_path) if original_video_path is not None else "",
            "cam_original_mode": CAM_ORIGINAL_MODE,
            "draw_model_crop_box": bool(DRAW_MODEL_CROP_BOX),
            "output_dir": str(sample_dir),
        }

        debug_stats["selected_sample_index"] = int(sample_index)
        debug_stats["selected_true_class"] = int(true_class)
        debug_stats["vis_long_side"] = None if VIS_LONG_SIDE is None else int(VIS_LONG_SIDE)
        debug_stats["visualization_source"] = visualization_source
        debug_stats["original_video_path"] = str(original_video_path) if original_video_path is not None else ""
        debug_stats["cam_original_mode"] = CAM_ORIGINAL_MODE
        debug_stats["draw_model_crop_box"] = bool(DRAW_MODEL_CROP_BOX)
        debug_stats["output_frame_shape"] = list(overlay_frames[0].shape)

        save_json(sample_dir / "prediction_info.json", prediction_info)
        save_json(sample_dir / "debug_stats.json", debug_stats)

        generated_counts[int(true_class)] += 1
        processed_indices.add(int(sample_index))

        print(
            f"[{len(processed_indices)}/{expected_total}] "
            f"class_{true_class} sample_{current_class_count:03d} saved | "
            f"test_index={sample_index} | "
            f"source={visualization_source} | "
            f"pred={hoc_class_name(pred_class)} "
            f"prob={pred_prob:.3f} | "
            f"shape={overlay_frames[0].shape} | "
            f"video={original_video_path} | "
            f"npy={raw_npy_path}"
        )

        global_sample_start += batch_size

    gradcam.remove_hooks()

    summary = {
        "generated_counts": {
            str(k): int(v) for k, v in generated_counts.items()
        },
        "processed_indices": sorted([int(x) for x in processed_indices]),
        "expected_total": int(expected_total),
        "vis_long_side": None if VIS_LONG_SIDE is None else int(VIS_LONG_SIDE),
        "use_original_video_frames": bool(USE_ORIGINAL_VIDEO_FRAMES),
        "cam_original_mode": CAM_ORIGINAL_MODE,
        "draw_model_crop_box": bool(DRAW_MODEL_CROP_BOX),
        "output_dir": str(OUTPUT_DIR),
    }

    save_json(OUTPUT_DIR / "generation_summary.json", summary)

    print("=" * 80)
    print("Visualization finished.")
    print(f"Saved to: {OUTPUT_DIR}")
    print("Generated counts:")
    for cls in TARGET_CLASSES:
        print(f"  class_{cls}: {generated_counts[int(cls)]}")
    print("=" * 80)


if __name__ == "__main__":
    main()