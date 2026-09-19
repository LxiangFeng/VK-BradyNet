from pathlib import Path
import argparse
import cv2
import pandas as pd
from _common import ROOT, read_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    args = ap.parse_args()
    df = read_manifest(args.split)
    rows=[]
    for _,r in df.iterrows():
        p=ROOT/r["relative_path"]
        cap=cv2.VideoCapture(str(p))
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        rows.append({
            "filename":r["filename"],"label":r["label"],"patient_id":r["patient_id"],
            "fps":fps,"frames":frames,
            "duration_s":(frames/fps if fps>0 else None),
            "width":width,"height":height,
            "side":r.get("side","")
        })
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/"results"/f"video_audit_{args.split}.csv",index=False,encoding="utf-8-sig")
    print(f"N clips: {len(out)}")
    print(f"N participants: {out.patient_id.nunique()}")
    print("Class counts:")
    print(out.label.value_counts().sort_index().to_string())
    print("\nDuration:")
    print(out.duration_s.describe().round(3).to_string())
    print("\nFPS:")
    print(out.fps.value_counts().sort_index().to_string())
    print("\nMissing side:", (~out.side.fillna("").str.lower().isin(["left","right"])).sum())

if __name__ == "__main__":
    main()
