from pathlib import Path
import argparse
import importlib.util
import traceback
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from _common import ROOT, read_manifest

def load_official_extractor():
    fp = ROOT / "third_party" / "finger-tapping-severity" / "feature_extraction.py"
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} missing. Run scripts/01_setup_official_repo.py first."
        )

    # The official extractor opens OpenCV GUI windows. For batch extraction
    # we disable only display/event-loop calls; numerical feature logic is unchanged.
    cv2.startWindowThread = lambda *a, **k: 0
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None

    spec = importlib.util.spec_from_file_location("islam_official_feature_extraction", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Avoid an external ffprobe dependency. This replaces only duration retrieval.
    def cv_duration(filename):
        cap = cv2.VideoCapture(str(filename))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps is None or fps <= 0 or frames is None or frames <= 0:
            raise RuntimeError(f"Cannot read duration from {filename}")
        return float(frames / fps)

    mod.get_length = cv_duration
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    args = ap.parse_args()

    df = read_manifest(args.split)
    if "side" not in df:
        raise ValueError("Manifest has no side column.")

    invalid = ~df["side"].fillna("").str.lower().isin(["left","right"])
    if invalid.any():
        raise ValueError(
            f"{invalid.sum()} rows have no valid left/right side. "
            f"Fill the manifest or run 03_autodetect_side.py --split {args.split}"
        )

    fe = load_official_extractor()
    raw_dir = ROOT / "features" / "official_outputs" / args.split
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"features:{args.split}"):
        video = ROOT / row["relative_path"]
        side = str(row["side"]).lower()
        try:
            feats = fe.extract_features(
                str(video),
                str(raw_dir),
                side,
                labels=(int(row["label"]), "")
            )
            clean = {}
            for k, v in feats.items():
                if isinstance(v, (np.generic,)):
                    v = v.item()
                clean[k] = v
            clean.update({
                "filename": row["filename"],
                "relative_path": row["relative_path"],
                "label": int(row["label"]),
                "patient_id": str(row["patient_id"]),
                "side": side,
            })
            rows.append(clean)
        except Exception as e:
            failures.append({
                "filename": row["filename"],
                "relative_path": row["relative_path"],
                "label": row["label"],
                "patient_id": row["patient_id"],
                "side": side,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(limit=5).replace("\n"," | "),
            })

    feat_df = pd.DataFrame(rows)
    feat_out = ROOT / "features" / f"{args.split}_features.csv"
    feat_df.to_csv(feat_out, index=False, encoding="utf-8-sig")

    fail_df = pd.DataFrame(failures)
    fail_out = ROOT / "results" / f"extraction_failures_{args.split}.csv"
    fail_df.to_csv(fail_out, index=False, encoding="utf-8-sig")

    print(f"[DONE] {args.split}: success={len(rows)}, failed={len(failures)}")
    print("Features:", feat_out)
    print("Failures:", fail_out)
    if len(df):
        print(f"Extraction success rate: {len(rows)/len(df)*100:.2f}%")
    if failures:
        print("[WARNING] Do not silently drop failed test clips in a paper comparison.")
        print("Inspect the failure file and assess whether input duration is incompatible.")

if __name__ == "__main__":
    main()
