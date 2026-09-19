#!/usr/bin/env python3
"""
Prepare SlowFast Kinetics-style CSV files for PD 0/1/2 video grading.

Input metadata CSV must contain at least:
  video_path,label,patient_id
Optional:
  split  # values: train/val/test. If provided, this script respects it.

Output:
  <out_dir>/train.csv
  <out_dir>/val.csv
  <out_dir>/test.csv

Each output line follows SlowFast kinetics loader format:
  path/to/video.mp4 label

Example:
  python prepare_pd_csv.py \
    --metadata labels.csv \
    --out_dir data/pd_slowfast \
    --video_root /absolute/path/to/roi_clips \
    --path_mode relative
"""
import argparse
from pathlib import Path
import random
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def normalize_split(s: str) -> str:
    s = str(s).strip().lower()
    aliases = {"valid": "val", "validation": "val", "dev": "val"}
    return aliases.get(s, s)


def patient_level_split(df, seed=42, train_ratio=0.70, val_ratio=0.15):
    patients = sorted(df["patient_id"].astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    def assign(pid):
        pid = str(pid)
        if pid in train_p:
            return "train"
        if pid in val_p:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(assign)
    return df


def format_path(p, video_root=None, path_mode="as_is"):
    p = Path(str(p))
    if path_mode == "as_is":
        return str(p)
    if video_root is None:
        raise ValueError("--video_root is required when --path_mode is relative or absolute")
    root = Path(video_root).resolve()
    if p.is_absolute():
        abs_p = p.resolve()
    else:
        abs_p = (root / p).resolve()
    if path_mode == "absolute":
        return str(abs_p)
    if path_mode == "relative":
        return str(abs_p.relative_to(root))
    raise ValueError(f"Unknown path_mode: {path_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="CSV with video_path,label,patient_id; optional split")
    ap.add_argument("--out_dir", required=True, help="Directory to write train.csv/val.csv/test.csv")
    ap.add_argument("--video_root", default=None, help="Root directory for video paths")
    ap.add_argument("--path_mode", choices=["as_is", "relative", "absolute"], default="as_is")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--check_exists", action="store_true", help="Check whether files exist")
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    required = {"video_path", "label", "patient_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    bad_labels = sorted(set(df["label"]) - {0, 1, 2})
    if bad_labels:
        raise ValueError(f"labels must be 0/1/2, got: {bad_labels}")

    if "split" in df.columns:
        df["split"] = df["split"].map(normalize_split)
        bad = sorted(set(df["split"]) - {"train", "val", "test"})
        if bad:
            raise ValueError(f"split must be train/val/test, got: {bad}")
    else:
        df = patient_level_split(df, args.seed, args.train_ratio, args.val_ratio)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].copy()
        lines = []
        for _, row in sub.iterrows():
            p = format_path(row["video_path"], args.video_root, args.path_mode)
            if args.check_exists:
                check_path = Path(p) if Path(p).is_absolute() else Path(args.video_root or ".") / p
                if not check_path.exists():
                    raise FileNotFoundError(f"Missing video: {check_path}")
                if check_path.suffix.lower() not in VIDEO_EXTS:
                    raise ValueError(f"Unexpected video suffix: {check_path}")
            lines.append(f"{p} {int(row['label'])}")
        (out_dir / f"{split}.csv").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{split}: {len(sub)} clips, {sub['patient_id'].nunique()} patients -> {out_dir / f'{split}.csv'}")

    print("\nLabel distribution by split:")
    print(pd.crosstab(df["split"], df["label"], dropna=False))


if __name__ == "__main__":
    main()
