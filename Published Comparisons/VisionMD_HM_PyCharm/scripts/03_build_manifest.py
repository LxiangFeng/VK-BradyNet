import argparse
from pathlib import Path
import pandas as pd
from _common import ROOT, CONFIG, split_dir, manifest_path, patient_id_from_filename

VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".webm"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    args = ap.parse_args()

    rows=[]
    base=split_dir(args.split)
    for label in [0,1,2]:
        for p in sorted((base/str(label)).rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append({
                    "filename": p.name,
                    "relative_path": str(p.relative_to(ROOT)).replace("\\","/"),
                    "label": label,
                    "patient_id": patient_id_from_filename(p.name),
                    "side": "auto",
                    "roi_x": "",
                    "roi_y": "",
                    "roi_w": "",
                    "roi_h": "",
                })
    if not rows:
        raise RuntimeError(f"No videos found under {base}/0|1|2")

    out = manifest_path(args.split)
    df = pd.DataFrame(rows)

    # Preserve manually edited side/ROI values.
    if out.exists():
        old = pd.read_csv(out, dtype={"patient_id":str})
        key = ["filename","relative_path"]
        keep_cols = key + ["side","roi_x","roi_y","roi_w","roi_h"]
        old = old[[c for c in keep_cols if c in old.columns]]
        df = df.merge(old, on=key, how="left", suffixes=("","_old"))
        for c in ["side","roi_x","roi_y","roi_w","roi_h"]:
            oc=c+"_old"
            if oc in df.columns:
                df[c] = df[oc].where(df[oc].notna(), df[c])
                df.drop(columns=[oc], inplace=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"[OK] {args.split}: {len(df)} clips")
    print(f"Participants from filename prefix: {df.patient_id.nunique()}")
    print(df.label.value_counts().sort_index().to_string())

    exp = CONFIG["expected_split_sizes"].get(args.split)
    if exp is not None and len(df) != exp:
        print(f"[WARNING] Expected {exp} clips for {args.split}, found {len(df)}.")

if __name__ == "__main__":
    main()
