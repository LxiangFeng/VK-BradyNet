from pathlib import Path
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "data" / "test_manifest.csv"
video_root = ROOT / "data" / "test_videos"

df = pd.read_csv(manifest_path)
required = {"filename", "label", "patient_id", "side"}
missing_cols = required - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing manifest columns: {sorted(missing_cols)}")

bad_labels = sorted(set(df["label"].dropna().astype(int)) - {0,1,2})
if bad_labels:
    raise ValueError(f"Labels must be 0/1/2. Found: {bad_labels}")

bad_sides = sorted(set(df["side"].dropna().astype(str)) - {"Left", "Right"})
if bad_sides:
    raise ValueError(f"side must be Left/Right. Found: {bad_sides}")

records = []
missing_files = []

for _, row in df.iterrows():
    p = video_root / str(row["filename"])
    if not p.exists():
        missing_files.append(str(p))
        continue

    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = nframes / fps if fps > 0 else float("nan")
    records.append({
        "filename": row["filename"],
        "fps": fps,
        "frames": int(nframes),
        "duration_s": duration,
        "width": width,
        "height": height,
    })

print("=" * 72)
print("FastEval × Our FT Test Set: dataset audit")
print("=" * 72)
print(f"N manifest rows       : {len(df)}")
print(f"N unique participants : {df['patient_id'].nunique()}")
print("\nLabel distribution:")
print(df["label"].value_counts().sort_index().to_string())
print("\nSide distribution:")
print(df["side"].value_counts().to_string())

if missing_files:
    print(f"\n[ERROR] Missing files: {len(missing_files)}")
    for x in missing_files[:20]:
        print("  ", x)
else:
    print("\n[OK] All video files exist.")

if records:
    q = pd.DataFrame(records)
    print("\nVideo statistics:")
    print(q[["fps","frames","duration_s","width","height"]].describe().round(2).to_string())
    short = q[q["duration_s"] < 5.0]
    print(f"\nVideos < 5 s: {len(short)}/{len(q)}")
    if len(short):
        print("[WARNING] FastEval's official guidance recommends videos longer than 5 s.")
        print("          This script does NOT loop/duplicate/pad video content.")

if len(df) != 150:
    print(f"\n[WARNING] Expected 150 FT test clips for the current Table 5 plan, found {len(df)}.")

audit_path = ROOT / "results" / "dataset_audit.csv"
audit_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(audit_path, index=False)
print(f"\nSaved: {audit_path}")
