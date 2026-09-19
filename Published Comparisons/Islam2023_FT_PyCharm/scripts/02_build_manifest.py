from pathlib import Path
import argparse
import pandas as pd
from _common import ROOT, split_dir, manifest_path, patient_id_from_filename

VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".webm"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    args = ap.parse_args()

    base = split_dir(args.split)
    rows = []
    for label in [0,1,2]:
        cdir = base / str(label)
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append({
                    "filename": p.name,
                    "relative_path": str(p.relative_to(ROOT)).replace("\\","/"),
                    "label": label,
                    "patient_id": patient_id_from_filename(p.name),
                    "side": "",
                    "side_source": "",
                    "side_confidence": "",
                })

    if not rows:
        raise RuntimeError(f"No videos found under {base}/0|1|2")

    df = pd.DataFrame(rows)
    out = manifest_path(args.split)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing manually filled side information when rebuilding.
    if out.exists():
        old = pd.read_csv(out, dtype={"patient_id":str})
        keep = old[["filename","relative_path","side","side_source","side_confidence"]].copy()
        df = df.merge(keep, on=["filename","relative_path"], how="left", suffixes=("","_old"))
        for col in ["side","side_source","side_confidence"]:
            oldcol = f"{col}_old"
            if oldcol in df:
                df[col] = df[oldcol].where(df[oldcol].notna(), df[col])
                df.drop(columns=[oldcol], inplace=True)

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {args.split}: {len(df)} clips, {df.patient_id.nunique()} participants")
    print(df["label"].value_counts().sort_index().to_string())
    print("Saved:", out)

if __name__ == "__main__":
    main()
