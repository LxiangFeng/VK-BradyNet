# convert_frame_json_to_video_npy.py
# Convert per-frame MediaPipe JSON folders to one .npy file per video.
# Output shape for each video: [T, 21, 3] = x_norm, y_norm, z_norm.

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Example input layout:
# C:/Users/Administrator/Desktop/keypoint_json/train/0/10_1/000000.json
# C:/Users/Administrator/Desktop/keypoint_json/train/0/10_1/000001.json
#
# Output layout:
# C:/Users/Administrator/Desktop/data_keypoints_npy/train/0/10_1.npy

JSON_ROOT = Path(r"C:\Users\Administrator\Desktop\keypoint_json")
OUT_ROOT = Path(r"C:\Users\Administrator\Desktop\data_keypoints_npy")
NUM_KEYPOINTS = 21
KEYPOINT_DIM = 3


def read_one_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        hands = obj.get("hands", [])
        if not hands:
            return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        # Use first detected hand. If you need right/left selection, modify here.
        lms = hands[0].get("landmarks", [])
        arr = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, lm in enumerate(lms[:NUM_KEYPOINTS]):
            arr[i, 0] = float(lm.get("x_norm", 0.0))
            arr[i, 1] = float(lm.get("y_norm", 0.0))
            arr[i, 2] = float(lm.get("z_norm", 0.0))
        return arr
    except Exception:
        return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)


def main():
    if not JSON_ROOT.exists():
        raise FileNotFoundError(f"JSON_ROOT does not exist: {JSON_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # A video folder is any folder containing at least one .json file.
    video_dirs = []
    for d in JSON_ROOT.rglob("*"):
        if d.is_dir() and any(d.glob("*.json")):
            video_dirs.append(d)
    video_dirs = sorted(video_dirs)

    print(f"Found video json folders: {len(video_dirs)}")
    for d in tqdm(video_dirs, desc="Converting"):
        json_files = sorted(d.glob("*.json"))
        seq = [read_one_json(p) for p in json_files]
        if not seq:
            continue
        arr = np.stack(seq, axis=0).astype(np.float32)  # [T,21,3]
        rel = d.relative_to(JSON_ROOT)
        out_path = OUT_ROOT / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)

    print("Done. Output:", OUT_ROOT)


if __name__ == "__main__":
    main()
