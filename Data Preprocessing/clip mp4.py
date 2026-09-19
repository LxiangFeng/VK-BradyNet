import os
import re
import subprocess
import tempfile
from pathlib import Path

# 支持的图片格式后缀
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def natural_key(path: Path):
    """按文件名中的数字自然排序（防止出现 10.jpg 排在 2.jpg 前面的情况）"""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def find_frame_folders(root: Path):
    """递归查找 root 目录下所有直接包含图片的文件夹"""
    folders = []

    # 如果 root 本身直接包含图片，把它加进去
    root_images = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if root_images:
        folders.append(root)

    # 遍历所有子文件夹
    for p in root.rglob("*"):
        if p.is_dir():
            images = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            if images:
                folders.append(p)
    return folders


def escape_concat_path(path: Path):
    """
    针对 Windows 环境下 FFmpeg concat 协议的路径转义函数
    处理斜杠、单引号以及各种特殊字符
    """
    s = str(path.resolve())
    s = s.replace("\\", "/")  # 统一转换为正斜杠
    s = s.replace("'", "'\\''")  # 转义单引号
    return s


def make_concat_file(images, concat_path: Path, frame_duration: float):
    """生成符合 FFmpeg 规范的 concat 文本文件"""
    with concat_path.open("w", encoding="utf-8") as f:
        for img in images:
            f.write(f"file '{escape_concat_path(img)}'\n")
            f.write(f"duration {frame_duration:.10f}\n")
        # FFmpeg 规范：最后一行需要重复写一次最后的图片，否则最后一帧会缺失时间
        if images:
            f.write(f"file '{escape_concat_path(images[-1])}'\n")


def convert_folder_to_mp4(folder: Path, output_dir: Path, fps: int = 25, crf: int = 18, overwrite: bool = True):
    # 获取当前文件夹下的所有图片并排序
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=natural_key)

    if not images:
        print(f"[-] [跳过] {folder} : 该文件夹内没有发现有效图片。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保持文件夹名字作为视频文件名
    output_file = output_dir / f"{folder.name}.mp4"
    frame_duration = 1.0 / fps

    # 使用临时文件夹存放文本清单，用完自动销毁
    with tempfile.TemporaryDirectory() as tmpdir:
        concat_file = Path(tmpdir) / "frames.txt"
        make_concat_file(images, concat_file, frame_duration)

        # 构建 FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-fps_mode", "vfr",  # 修复老版本 -vsync 警告，适配 FFmpeg 7.x+
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 💡 核心修复：强行将奇数宽高自动转换为偶数，防止 libx264 报错崩溃
            "-pix_fmt", "yuv420p",  # 最通用的像素格式，确保所有播放器和 OpenCV 都能识别
            "-c:v", "libx264",  # H.264 编码
            "-crf", str(crf),  # 18 代表高质量无损体感
            str(output_file),
        ]

        print(f"\n[+] [开始转换] 文件夹: {folder.name} (共 {len(images)} 帧 -> 目标: {output_file.name})")

        try:
            # 执行命令，同时捕获标准输出和错误日志
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            # 获取生成文件的大小
            file_size_kb = output_file.stat().st_size / 1024
            print(f"✅ [成功] 视频已完美生成: {output_file} (大小: {file_size_kb:.1f} KB)")

        except subprocess.CalledProcessError as e:
            print(f"❌ [失败] 压制过程中发生致命错误！")
            print("==================== FFmpeg 错误诊断日志 ====================")
            print(e.stderr if e.stderr else "未能捕获到标准错误输出，请确认 FFmpeg 是否正常运行。")
            print("=============================================================")

            # 如果失败了且产生了一个 0 KB 的坏文件，将其清除，防止污染数据集
            if output_file.exists() and output_file.stat().st_size == 0:
                output_file.unlink()


# ---------------------------
# 主程序入口
# ---------------------------
def main():
    # ==========================================
    # 请在此处核对并修改你的输入和输出路径
    # ==========================================
    root = Path(r"C:\Users\Administrator\Desktop\cropdata\val\2").resolve()
    output_dir = Path(r"C:\Users\Administrator\Desktop\data\val\2").resolve()  # 直接对齐你 SlowFast 读取的 data/0 目录

    fps = 25  # 帧率
    crf = 18  # 压缩率（越小质量越高，18-23为标准范围）
    overwrite = True

    if not root.exists():
        print(f"❌ 错误：输入目录不存在，请检查: {root}")
        return

    print("🔍 正在扫描图片文件夹，请稍候...")
    folders = find_frame_folders(root)

    if not folders:
        print("❌ 未能找到任何包含图片帧的文件夹，请检查路径层级！")
        return

    print(f"🎉 成功找到 {len(folders)} 个帧文件夹，开始压制...\n")

    for folder in folders:
        convert_folder_to_mp4(folder, output_dir, fps, crf, overwrite)

    print("\n🚀 【全部转换任务完成！】你可以直接运行 SlowFast 模型开始训练了。")


if __name__ == "__main__":
    main()