import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# =========================
# 路径设置
# =========================
VIDEO_ROOT = Path(r"C:\Users\Administrator\Desktop\1")
OUT_VIDEO_ROOT = Path(r"C:\Users\Administrator\Desktop\2")
OUT_NPY_ROOT = Path(r"C:\Users\Administrator\Desktop\2")

OUT_VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
OUT_NPY_ROOT.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

# =========================
# MediaPipe Hands 初始化
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def normalize_keypoints(hand_kpts):
    """
    对一帧的 21 个手部关键点做相对归一化。

    输入:
        hand_kpts: np.ndarray, shape [21, 3]
                  原始 MediaPipe 坐标 x, y, z

    输出:
        norm_kpts: np.ndarray, shape [21, 3]
                   相对归一化后的关键点
    """

    hand_kpts = np.asarray(hand_kpts, dtype=np.float32)

    # 如果是全零帧，直接返回全零
    if np.allclose(hand_kpts, 0):
        return np.zeros((21, 3), dtype=np.float32)

    # 以手腕点 0 作为原点
    wrist = hand_kpts[0:1, :]
    relative = hand_kpts - wrist

    # 用 wrist(0) 到 middle_mcp(9) 的距离作为尺度
    # 这样可以消除手离镜头远近、画面位置等影响
    scale = np.linalg.norm(hand_kpts[9, :2] - hand_kpts[0, :2])

    if scale < 1e-6:
        scale = 1.0

    relative = relative / scale

    return relative.astype(np.float32)


def add_10d_features(kpts_seq, valid_mask):
    """
    生成 10 维关键点特征。

    输入:
        kpts_seq: np.ndarray, shape [T, 21, 3]
                  相对归一化后的关键点序列

        valid_mask: np.ndarray, shape [T]
                    每一帧是否检测到手
                    检测到手 = 1
                    未检测到手 = 0

    输出:
        enhanced: np.ndarray, shape [T, 21, 10]

        10维分别是:
        0: x
        1: y
        2: z
        3: dx
        4: dy
        5: dz
        6: speed
        7: accel
        8: dist_to_wrist
        9: confidence
    """

    kpts_seq = np.asarray(kpts_seq, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=np.float32)

    T, K, C = kpts_seq.shape

    if K != 21 or C != 3:
        raise ValueError(f"Expected kpts_seq shape [T, 21, 3], but got {kpts_seq.shape}")

    # =========================
    # 1. delta: dx, dy, dz
    # =========================
    delta = np.zeros_like(kpts_seq, dtype=np.float32)

    if T > 1:
        delta[1:] = kpts_seq[1:] - kpts_seq[:-1]

    # 如果当前帧无效，或者上一帧无效，则该帧 delta 置零
    for t in range(T):
        if valid_mask[t] < 0.5:
            delta[t] = 0.0
        if t > 0 and valid_mask[t - 1] < 0.5:
            delta[t] = 0.0

    # =========================
    # 2. speed: 每个关键点的速度大小
    # speed = sqrt(dx^2 + dy^2 + dz^2)
    # shape [T, 21, 1]
    # =========================
    speed = np.linalg.norm(delta, axis=-1, keepdims=True).astype(np.float32)

    # =========================
    # 3. accel: 加速度
    # accel[t] = speed[t] - speed[t-1]
    # shape [T, 21, 1]
    # =========================
    accel = np.zeros_like(speed, dtype=np.float32)

    if T > 1:
        accel[1:] = speed[1:] - speed[:-1]

    # 当前帧或上一帧无效时，加速度置零
    for t in range(T):
        if valid_mask[t] < 0.5:
            accel[t] = 0.0
        if t > 0 and valid_mask[t - 1] < 0.5:
            accel[t] = 0.0

    # =========================
    # 4. dist_to_wrist: 每个关键点到手腕点的距离
    # 因为前面已经以 wrist 为原点，
    # 所以这里就是 sqrt(x^2 + y^2 + z^2)
    # shape [T, 21, 1]
    # =========================
    dist_to_wrist = np.linalg.norm(kpts_seq, axis=-1, keepdims=True).astype(np.float32)

    # 无效帧距离置零
    for t in range(T):
        if valid_mask[t] < 0.5:
            dist_to_wrist[t] = 0.0

    # =========================
    # 5. confidence: 检测置信标记
    # 检测到手 = 1
    # 未检测到手 = 0
    # shape [T, 21, 1]
    # =========================
    confidence = valid_mask.reshape(T, 1, 1).repeat(K, axis=1).astype(np.float32)

    # =========================
    # 拼接成 10 维
    # [x,y,z, dx,dy,dz, speed, accel, dist_to_wrist, confidence]
    # =========================
    enhanced = np.concatenate(
        [
            kpts_seq,        # [T, 21, 3]
            delta,           # [T, 21, 3]
            speed,           # [T, 21, 1]
            accel,           # [T, 21, 1]
            dist_to_wrist,   # [T, 21, 1]
            confidence,      # [T, 21, 1]
        ],
        axis=-1,
    )

    return enhanced.astype(np.float32)


# =========================
# 遍历视频
# =========================
for video_path in VIDEO_ROOT.rglob("*"):
    if video_path.suffix.lower() not in VIDEO_EXTS:
        continue

    print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    out_video_path = OUT_VIDEO_ROOT / video_path.name
    out_npy_path = OUT_NPY_ROOT / (video_path.stem + ".npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(out_video_path),
        fourcc,
        fps,
        (width, height),
    )

    # 保存每一帧归一化后的关键点
    # 最终 shape: [T, 21, 3]
    keypoints_all_frames = []

    # 保存每一帧是否检测到手
    # shape: [T]
    valid_mask = []

    total_frames = 0
    valid_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 默认这一帧没有检测到手
        frame_keypoints = np.zeros((21, 3), dtype=np.float32)
        frame_valid = 0.0

        if results.multi_hand_landmarks:
            # max_num_hands=1，所以只取第一只手
            hand_landmarks = results.multi_hand_landmarks[0]

            hand_kpts = []
            for lm in hand_landmarks.landmark:
                hand_kpts.append([lm.x, lm.y, lm.z])

            hand_kpts = np.asarray(hand_kpts, dtype=np.float32)

            # 相对归一化
            frame_keypoints = normalize_keypoints(hand_kpts)

            frame_valid = 1.0
            valid_frames += 1

            # 绘制关键点到原视频帧
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

        keypoints_all_frames.append(frame_keypoints)
        valid_mask.append(frame_valid)

        # 写入带关键点的视频
        out_video.write(frame)

    cap.release()
    out_video.release()

    if len(keypoints_all_frames) == 0:
        print(f"No frames found in video: {video_path}")
        continue

    # [T, 21, 3]
    keypoints_all_frames = np.asarray(keypoints_all_frames, dtype=np.float32)

    # [T]
    valid_mask = np.asarray(valid_mask, dtype=np.float32)

    # 生成 [T, 21, 10]
    save_arr = add_10d_features(keypoints_all_frames, valid_mask)

    np.save(str(out_npy_path), save_arr)

    valid_ratio = valid_frames / max(total_frames, 1)

    print(f"Saved video: {out_video_path}")
    print(f"Saved npy:   {out_npy_path}")
    print(f"npy shape:   {save_arr.shape}")
    print(f"valid hand frames: {valid_frames}/{total_frames}, ratio={valid_ratio:.2%}")

    # 简单检查速度信息
    mean_speed = save_arr[:, :, 6].mean()
    max_speed = save_arr[:, :, 6].max()
    mean_accel = save_arr[:, :, 7].mean()

    print(f"mean speed:  {mean_speed:.6f}")
    print(f"max speed:   {max_speed:.6f}")
    print(f"mean accel:  {mean_accel:.6f}")
    print("-" * 80)

hands.close()

print("All videos processed.")