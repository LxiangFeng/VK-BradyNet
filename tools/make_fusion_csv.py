# make_fusion_csv.py
# Generate train.csv / val.csv for RGB + Keypoint fusion training.
# CSV format:
#   absolute_video_path absolute_keypoint_npy_path label

from pathlib import Path

DATA_ROOT = Path(r"C:\Users\Administrator\Desktop\data")
KEYPOINT_ROOT = Path(r"C:\Users\Administrator\Desktop\npy")
OUT_ROOT = Path(r"C:\Users\Administrator\Desktop\data")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SPLITS = ["train", "val", "test"]


def make_one_split(split: str):
    video_split = DATA_ROOT / split
    keypoint_split = KEYPOINT_ROOT / split
    out_csv = OUT_ROOT / f"{split}.csv"
    rows = []
    missing = []

    for cls_dir in sorted(video_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for video in sorted(cls_dir.rglob("*")):
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
                continue
            rel_to_split = video.relative_to(video_split)
            # Example: 0/10_1.mp4 -> 0/10_1.npy
            kpt = keypoint_split / rel_to_split.with_suffix(".npy")
            if not kpt.exists():
                missing.append(str(kpt))
            rows.append(f"{video.as_posix()} {kpt.as_posix()} {label}\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("".join(rows), encoding="utf-8")
    print(f"{split}: wrote {len(rows)} rows -> {out_csv}")
    if missing:
        miss_file = OUT_ROOT / f"missing_keypoints_{split}.txt"
        miss_file.write_text("\n".join(missing), encoding="utf-8")
        print(f"{split}: missing keypoints {len(missing)} -> {miss_file}")


def main():
    for split in SPLITS:
        make_one_split(split)


if __name__ == "__main__":
    main()
