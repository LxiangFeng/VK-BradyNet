from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from _common import ROOT, read_manifest, manifest_path

def infer_side(video_path: Path, sample_every=3):
    cap = cv2.VideoCapture(str(video_path))
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    votes = {"Left":0.0, "Right":0.0}
    n_seen = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every:
            idx += 1
            continue
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_handedness:
            # For ROI videos, choose the highest-confidence detected hand in each sampled frame.
            best = max(
                (h.classification[0] for h in res.multi_handedness),
                key=lambda c: c.score
            )
            if best.score >= 0.5:
                votes[best.label] += float(best.score)
                n_seen += 1
    cap.release()
    hands.close()

    total = votes["Left"] + votes["Right"]
    if total <= 0:
        return "", 0.0, "no_mediapipe_hand"

    mp_label = max(votes, key=votes.get)
    confidence = votes[mp_label] / total

    # Match the official repository's assumption:
    # t_label = {"left":"Right", "right":"Left"}
    requested_side = "left" if mp_label == "Right" else "right"
    return requested_side, confidence, f"auto_mediapipe_{mp_label}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--overwrite-manual", action="store_true",
                    help="Normally existing left/right entries are preserved.")
    args = ap.parse_args()

    df = read_manifest(args.split)
    out_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"side:{args.split}"):
        current = str(row.get("side","")).strip().lower()
        if current in {"left","right"} and not args.overwrite_manual:
            out_rows.append(row)
            continue
        p = ROOT / row["relative_path"]
        side, conf, source = infer_side(p)
        row["side"] = side
        row["side_confidence"] = round(conf, 4)
        row["side_source"] = source
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(manifest_path(args.split), index=False, encoding="utf-8-sig")
    print(out["side"].value_counts(dropna=False).to_string())
    low = out[pd.to_numeric(out["side_confidence"], errors="coerce").fillna(1.0) < 0.70]
    print(f"Low-confidence (<0.70) auto side assignments: {len(low)}")
    print("Saved:", manifest_path(args.split))

if __name__ == "__main__":
    main()
