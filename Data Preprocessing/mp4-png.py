import subprocess
from pathlib import Path


def extract_frames_from_mp4(input_dir, output_dir, image_format="png"):
    """
    遍历文件夹下所有 MP4 视频，逐帧导出图片。

    特点：
    1. 不改变视频原始分辨率；
    2. 不抽帧；
    3. 不改变帧顺序；
    4. 默认导出 PNG，无损保存；
    5. 每个视频单独生成一个同名文件夹保存帧图片。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = list(input_dir.rglob("*.mp4")) + list(input_dir.rglob("*.mp4"))

    if not mp4_files:
        print("未找到 MP4 文件。")
        return

    for mp4_file in mp4_files:
        video_name = mp4_file.stem

        # 为每个视频创建单独的输出文件夹
        video_output_dir = output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        output_pattern = video_output_dir / f"{video_name}_frame_%06d.{image_format}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(mp4_file),

            # 不改变原始帧率，不抽帧
            "-vsync", "0",

            # 输出图片路径
            str(output_pattern)
        ]

        print("=" * 80)
        print(f"正在处理视频：{mp4_file}")
        print(f"帧图片输出目录：{video_output_dir}")

        try:
            subprocess.run(cmd, check=True)
            print(f"完成：{mp4_file.name}")

        except subprocess.CalledProcessError as e:
            print(f"处理失败：{mp4_file.name}")
            print(f"错误代码：{e.returncode}")


if __name__ == "__main__":
    input_folder = r"C:\Users\Administrator\Desktop\1"
    output_folder = r"C:\Users\Administrator\Desktop\2"

    extract_frames_from_mp4(
        input_dir=input_folder,
        output_dir=output_folder,
        image_format="png"
    )