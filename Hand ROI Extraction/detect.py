# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def get_best_box(result, image_shape):
    """
    从YOLO检测结果中取置信度最高的检测框
    返回: (x1, y1, x2, y2) 或 None
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    best_index = int(np.argmax(confs))

    box = boxes[best_index]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    h, w = image_shape[:2]

    # 防止坐标越界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def make_even(value):
    """
    部分视频编码器要求宽高为偶数
    """
    return value if value % 2 == 0 else value + 1


def find_video_files(input_root):
    """
    遍历输入文件夹，查找所有视频文件
    """
    input_root = Path(input_root)

    video_suffix = {
        ".mp4", ".avi", ".mov", ".mkv",
        ".mts", ".MTS", ".m2ts", ".M2TS",
        ".wmv", ".flv"
    }

    video_files = []

    for file_path in input_root.rglob("*"):
        if file_path.is_file() and file_path.suffix in video_suffix:
            video_files.append(file_path)

    video_files = sorted(video_files)
    return video_files


def crop_one_video(
        model,
        video_path,
        save_video_path,
        conf_thres=0.25,
        write_black_when_miss=True,
        output_format="avi"
):
    """
    对单个视频进行YOLO检测并裁切检测区域，保存成新视频
    """

    video_path = Path(video_path)
    save_video_path = Path(save_video_path)

    print("=" * 100)
    print(f"开始处理视频: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"视频打开失败，跳过: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    print(f"原视频尺寸: {original_width} x {original_height}")
    print(f"FPS: {fps}")
    print(f"总帧数: {total_frames}")

    # =========================
    # 第一遍：检测所有帧，记录检测框
    # =========================

    frame_boxes = []
    max_crop_w = 0
    max_crop_h = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf_thres,
            save=False,
            show=False,
            save_txt=False,
            verbose=False
        )

        result = results[0]
        best_box = get_best_box(result, frame.shape)
        frame_boxes.append(best_box)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            crop_w = x2 - x1
            crop_h = y2 - y1

            max_crop_w = max(max_crop_w, crop_w)
            max_crop_h = max(max_crop_h, crop_h)

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"第一遍检测进度: {frame_index}/{total_frames}")

    cap.release()

    if max_crop_w == 0 or max_crop_h == 0:
        print(f"整段视频未检测到目标，跳过: {video_path}")
        return False

    canvas_w = make_even(max_crop_w)
    canvas_h = make_even(max_crop_h)

    print(f"裁切视频画布尺寸: {canvas_w} x {canvas_h}")

    # =========================
    # 第二遍：根据检测框裁切并写入视频
    # =========================

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"视频重新打开失败，跳过: {video_path}")
        return False

    save_video_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format.lower() == "avi":
        # FFV1 无损编码，清晰度最好，推荐保存 avi
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    elif output_format.lower() == "mp4":
        # mp4v 可以保存 mp4，但会有压缩损失
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        # 兜底方案
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(save_video_path),
        fourcc,
        fps,
        (canvas_w, canvas_h)
    )

    if not writer.isOpened():
        cap.release()
        print(f"视频写入器打开失败，跳过: {save_video_path}")
        return False

    frame_index = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index >= len(frame_boxes):
            break

        best_box = frame_boxes[frame_index]

        if best_box is None:
            if write_black_when_miss:
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                writer.write(canvas)
                saved_frames += 1

            frame_index += 1
            continue

        x1, y1, x2, y2 = best_box

        # 原尺寸裁切，不resize，尽量不改变目标区域清晰度
        cropped_frame = frame[y1:y2, x1:x2]

        if cropped_frame.size == 0:
            if write_black_when_miss:
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                writer.write(canvas)
                saved_frames += 1

            frame_index += 1
            continue

        crop_h, crop_w = cropped_frame.shape[:2]

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 居中放置裁切区域
        left = (canvas_w - crop_w) // 2
        top = (canvas_h - crop_h) // 2

        canvas[top:top + crop_h, left:left + crop_w] = cropped_frame

        writer.write(canvas)
        saved_frames += 1

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"第二遍裁切进度: {frame_index}/{total_frames}")

    cap.release()
    writer.release()

    print(f"视频裁切完成: {save_video_path}")
    print(f"输出帧数: {saved_frames}")

    return True


def batch_crop_videos(
        model_path,
        input_root,
        output_root,
        conf_thres=0.25,
        write_black_when_miss=True,
        output_format="avi"
):
    """
    批量遍历文件夹内所有视频，逐个预测裁切并保存到新文件夹
    """

    model_path = Path(model_path)
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if not input_root.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    print("正在加载YOLO模型...")
    model = YOLO(str(model_path))
    print("模型加载完成。")

    video_files = find_video_files(input_root)

    if len(video_files) == 0:
        raise RuntimeError(f"输入文件夹内没有找到视频文件: {input_root}")

    print(f"共找到 {len(video_files)} 个视频文件。")

    success_count = 0
    fail_count = 0

    for index, video_path in enumerate(video_files, start=1):
        print(f"\n正在处理第 {index}/{len(video_files)} 个视频")

        # 保持原始相对目录结构
        relative_path = video_path.relative_to(input_root)

        # 输出文件后缀
        if output_format.lower() == "avi":
            save_relative_path = relative_path.with_suffix(".avi")
        elif output_format.lower() == "mp4":
            save_relative_path = relative_path.with_suffix(".mp4")
        else:
            save_relative_path = relative_path.with_suffix(".avi")

        save_video_path = output_root / save_relative_path

        ok = crop_one_video(
            model=model,
            video_path=video_path,
            save_video_path=save_video_path,
            conf_thres=conf_thres,
            write_black_when_miss=write_black_when_miss,
            output_format=output_format
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

    print("=" * 100)
    print("全部视频处理完成！")
    print(f"成功处理: {success_count}")
    print(f"处理失败或未检测到目标: {fail_count}")
    print(f"输出文件夹: {output_root}")


if __name__ == "__main__":

    # YOLO模型路径
    model_path = r"C:\Users\Administrator\Desktop\ultralytics-main\best.pt"

    # 输入视频文件夹
    input_root = r"C:\Users\Administrator\Desktop\2"

    # 裁切后视频保存文件夹
    output_root = r"C:\Users\Administrator\Desktop\RGB K Finger Tapping data\data\2"

    # 置信度阈值
    conf_thres = 0.25

    # True：没有检测到目标的帧写黑帧，保持原视频长度
    # False：没有检测到目标的帧跳过，输出视频可能变短
    write_black_when_miss = True

    # 推荐 avi：使用 FFV1，尽量保持清晰度
    # 如果你必须输出 mp4，可以改成 "mp4"
    output_format = "mp4"

    batch_crop_videos(
        model_path=model_path,
        input_root=input_root,
        output_root=output_root,
        conf_thres=conf_thres,
        write_black_when_miss=write_black_when_miss,
        output_format=output_format
    )