# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path


def get_video_duration(input_video):
    """
    使用 ffprobe 获取视频总时长，单位：秒
    """

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    return float(result.stdout.strip())


def find_mts_files(input_dir):
    """
    从文件夹中查找所有 .MTS / .MP4 视频
    """

    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    if not input_dir.is_dir():
        raise RuntimeError(f"输入路径不是文件夹: {input_dir}")

    mts_files = []

    mts_files.extend(input_dir.glob("*.MP4"))
    mts_files.extend(input_dir.glob("*.MTS"))

    mts_files = sorted(mts_files)

    if len(mts_files) == 0:
        raise RuntimeError(f"该文件夹下没有找到 .MTS 或 .mts 视频: {input_dir}")

    return mts_files


def split_one_mts_to_mp4_by_2s(input_video, output_dir, segment_time=2, fps=25):
    """
    将单个 MTS / AVCHD 视频每 segment_time 秒切成一个 mp4
    稳定可打开版本：重新编码为标准 H.264 MP4
    """

    input_video = Path(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_duration = get_video_duration(input_video)

    print("=" * 80)
    print(f"正在处理视频: {input_video}")
    print(f"视频总时长: {total_duration:.3f} 秒")
    print(f"每段时长: {segment_time} 秒")
    print(f"输出目录: {output_dir}")

    output_pattern = output_dir / f"{input_video.stem}_part_%03d.mp4"

    cmd = [
        "ffmpeg",
        "-y",

        # 重新生成时间戳，减少 MTS 时间戳异常导致的切片问题
        "-fflags", "+genpts",

        # 输入视频
        "-i", str(input_video),

        # 只处理第一个视频流，不保留音频
        "-map", "0:v:0",
        "-an",

        # 重新编码为 H.264，保证 mp4 片段稳定可打开
        "-c:v", "libx264",

        # 高质量参数
        # 数值越小画质越高，文件越大
        "-crf", "16",

        # 编码速度与压缩效率
        "-preset", "slow",

        # mp4 通用兼容格式
        "-pix_fmt", "yuv420p",

        # 保持你的视频帧率 25fps
        "-r", str(fps),

        # 每 2 秒强制一个关键帧
        # 这样切出来的每段 mp4 更稳定
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",

        # 按固定时长切分
        "-f", "segment",
        "-segment_time", str(segment_time),

        # 每个片段时间戳从 0 开始
        "-reset_timestamps", "1",

        # 指定分段格式为 mp4
        "-segment_format", "mp4",

        str(output_pattern)
    ]

    print("开始切分...")
    subprocess.run(cmd, check=True)

    print(f"完成: {input_video.name}")


def split_folder_mts_to_mp4_by_2s(input_dir, output_root, segment_time=2, fps=25):
    """
    读取文件夹下所有 MTS 视频，每个视频切成 2 秒一个 mp4
    """

    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mts_files = find_mts_files(input_dir)

    print(f"输入文件夹: {input_dir}")
    print(f"共找到 {len(mts_files)} 个 MTS 视频")

    for mts_file in mts_files:
        # 每个原视频单独建一个输出文件夹，避免文件名冲突
        video_output_dir = output_root / mts_file.stem

        split_one_mts_to_mp4_by_2s(
            input_video=mts_file,
            output_dir=video_output_dir,
            segment_time=segment_time,
            fps=fps
        )

    print("=" * 80)
    print("全部视频切分完成！")


if __name__ == "__main__":

    # 这里写你的 MTS 文件夹路径
    input_dir = r"C:\Users\Administrator\Desktop\MTS"

    # 这里写输出 mp4 片段的保存路径
    output_root = r"C:\Users\Administrator\Desktop\MTS10"

    # 每 2 秒切成一个视频
    split_folder_mts_to_mp4_by_2s(
        input_dir=input_dir,
        output_root=output_root,
        segment_time=2,
        fps=25
    )