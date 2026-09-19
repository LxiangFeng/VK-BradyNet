from pathlib import Path
from collections import Counter

# =========================
# 总路径设置
# =========================
VIDEO_BASE = Path(r"C:\Users\Administrator\Desktop\data\data")
NPY_BASE = Path(r"C:\Users\Administrator\Desktop\data\npy")

# CSV保存位置
CSV_OUT_DIR = Path(r"C:\Users\Administrator\Desktop\data\data")
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]

# 三个数据集
SPLITS = ["train", "val", "test"]


def generate_csv(split):
    """
    生成：
        train.csv
        val.csv
        test.csv

    每一行格式：
        视频路径 npy路径 标签
    """

    video_root = VIDEO_BASE / split
    npy_root = NPY_BASE / split

    out_csv = CSV_OUT_DIR / f"{split}.csv"

    rows = []
    labels = []

    # =========================
    # 遍历当前 split 下所有 npy
    # =========================
    for npy_path in sorted(npy_root.rglob("*.npy")):

        # 例如:
        # npy/train/0/abc.npy
        #
        # relative_path:
        # 0/abc.npy
        relative_path = npy_path.relative_to(npy_root)

        # =========================
        # 获取标签
        # =========================
        # 第一层文件夹名作为标签
        # 0/abc.npy -> label = 0
        # 1/abc.npy -> label = 1
        # 2/abc.npy -> label = 2
        label = relative_path.parts[0]

        # =========================
        # 找对应的视频
        # =========================
        relative_no_suffix = relative_path.with_suffix("")

        video_path = None

        for ext in VIDEO_EXTS:
            candidate = (
                video_root / relative_no_suffix
            ).with_suffix(ext)

            if candidate.exists():
                video_path = candidate
                break

        # =========================
        # 找不到视频
        # =========================
        if video_path is None:
            print(f"[{split}] 找不到对应视频:")
            print(f"    {npy_path}")
            continue

        # Windows路径转换成 /
        # 保持和你之前CSV一样
        video_str = video_path.as_posix()
        npy_str = npy_path.as_posix()

        row = f"{video_str} {npy_str} {label}"

        rows.append(row)
        labels.append(label)

    # =========================
    # 写入CSV
    # =========================
    with open(out_csv, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")

    # =========================
    # 打印统计信息
    # =========================
    label_count = Counter(labels)

    print()
    print("=" * 70)
    print(f"{split}.csv 生成完成")
    print(f"保存位置: {out_csv}")
    print(f"样本总数: {len(rows)}")

    for label in sorted(label_count.keys()):
        print(f"类别 {label}: {label_count[label]}")

    print("=" * 70)


# =========================
# 一次生成三个 CSV
# =========================
for split in SPLITS:
    generate_csv(split)

print()
print("全部完成！")
print("已生成:")
print(CSV_OUT_DIR / "train.csv")
print(CSV_OUT_DIR / "val.csv")
print(CSV_OUT_DIR / "test.csv")