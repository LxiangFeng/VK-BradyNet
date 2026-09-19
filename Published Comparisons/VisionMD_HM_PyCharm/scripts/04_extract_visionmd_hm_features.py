import argparse
import sys
import traceback
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from _common import ROOT, CONFIG, read_manifest, official_backend

FEATURE_KEYS = [
    "MeanAmplitude","StdAmplitude","MeanSpeed","StdSpeed",
    "MeanRMSVelocity","StdRMSVelocity",
    "MeanOpeningSpeed","StdOpeningSpeed",
    "MeanClosingSpeed","StdClosingSpeed",
    "MeanMaxOpeningSpeed","StdMaxOpeningSpeed",
    "MeanMaxClosingSpeed","StdMaxClosingSpeed",
    "MeanCycleDuration","StdCycleDuration",
    "CVAmplitude","CVSpeed","CVRMSVelocity",
    "CVOpeningSpeed","CVClosingSpeed",
    "CVMaxOpeningSpeed","CVMaxClosingSpeed",
    "CVCycleDuration","Frequency","AmplitudeDecay",
    "VelocityDecay","RangeCycleDuration","NumberofPauses",
    "numberofHesitations",
]

def load_official_analyzer():
    backend = official_backend()
    if not backend.exists():
        raise FileNotFoundError(
            "Official VisionMD backend not found. Run 01_setup_official_visionmd.py."
        )
    sys.path.insert(0, str(backend))
    from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer
    return PeakfinderSignalAnalyzer

def make_detector():
    model = ROOT/"models"/"hand_landmarker.task"
    if not model.exists():
        raise FileNotFoundError("Run 02_download_hand_model.py first.")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def parse_roi(row, width, height):
    vals=[]
    for c in ["roi_x","roi_y","roi_w","roi_h"]:
        v=row.get(c, np.nan)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)
    if any(np.isnan(vals)):
        return 0,0,width,height
    x,y,w,h = [int(round(v)) for v in vals]
    x=max(0,min(x,width-1)); y=max(0,min(y,height-1))
    w=max(1,min(w,width-x)); h=max(1,min(h,height-y))
    return x,y,w,h

def interpolate_landmarks(arr, valid):
    # arr: T x 21 x 2
    out=arr.copy()
    T=arr.shape[0]
    idx=np.arange(T)
    good=np.where(valid)[0]
    if len(good)==0:
        raise RuntimeError("No valid hand landmark frames.")
    for j in range(21):
        for d in range(2):
            vals=out[:,j,d]
            vals[~valid]=np.nan
            mask=np.isfinite(vals)
            if mask.sum()==0:
                raise RuntimeError("Landmark interpolation failed.")
            out[:,j,d]=np.interp(idx, idx[mask], vals[mask])
    return out

def dominant_label(labels):
    labels=[x for x in labels if x in {"Left","Right"}]
    if not labels:
        return ""
    return max(set(labels), key=labels.count)

def extract_landmarks(video_path, requested_side):
    cap=cv2.VideoCapture(str(video_path))
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0 or nframes <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata.")
    cap.release()

    return fps,width,height,nframes

def process_video(video_path, row, analyzer_cls):
    cap=cv2.VideoCapture(str(video_path))
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0 or nframes <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata.")

    x,y,w,h=parse_roi(row,width,height)
    requested=str(row.get("side","auto")).strip().lower()
    if requested not in {"left","right","auto",""}:
        raise ValueError(f"Invalid side={requested}")
    if requested=="":
        requested="auto"

    detector=make_detector()
    frames=[]
    valid=[]
    chosen_labels=[]
    missing=0
    frame_idx=0

    while True:
        ok,frame=cap.read()
        if not ok:
            break
        crop=frame[y:y+h, x:x+w]
        rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts=int(round(frame_idx/fps*1000))
        result=detector.detect_for_video(image, ts)

        candidates=[]
        for k,lms in enumerate(result.hand_landmarks):
            label=""
            score=0.0
            if k < len(result.handedness) and result.handedness[k]:
                label=result.handedness[k][0].category_name
                score=float(result.handedness[k][0].score)
            candidates.append((label,score,lms))

        chosen=None
        if requested in {"left","right"}:
            target=requested.capitalize()
            matches=[c for c in candidates if c[0]==target]
            if matches:
                chosen=max(matches, key=lambda z:z[1])
        else:
            if candidates:
                chosen=max(candidates, key=lambda z:z[1])

        if chosen is None:
            frames.append(np.full((21,2),np.nan,dtype=float))
            valid.append(False)
            chosen_labels.append("")
            missing += 1
        else:
            label,score,lms=chosen
            xy=np.array([[lm.x*w + x, lm.y*h + y] for lm in lms],dtype=float)
            frames.append(xy)
            valid.append(True)
            chosen_labels.append(label)
        frame_idx += 1

    cap.release()
    detector.close()

    if frame_idx == 0:
        raise RuntimeError("No frames decoded.")
    missing_fraction=missing/frame_idx
    if missing_fraction > float(CONFIG["max_missing_fraction"]):
        raise RuntimeError(
            f"Hand missing in {missing_fraction:.1%} of frames, "
            f"exceeding VisionMD-style 10% threshold."
        )

    arr=np.stack(frames,axis=0)
    valid=np.asarray(valid,dtype=bool)
    arr=interpolate_landmarks(arr,valid)

    # Current public VisionMD hand_movement_{left,right}.py:
    # average(index-tip-to-wrist, middle-tip-to-wrist, ring-tip-to-wrist)
    wrist=arr[:,0,:]
    index_tip=arr[:,8,:]
    middle_tip=arr[:,12,:]
    ring_tip=arr[:,16,:]
    raw_signal=(
        np.linalg.norm(index_tip-wrist,axis=1)
        + np.linalg.norm(middle_tip-wrist,axis=1)
        + np.linalg.norm(ring_tip-wrist,axis=1)
    )/3.0

    strategy=str(CONFIG.get("norm_strategy","INDEXSIZE")).upper()
    if strategy=="INDEXSIZE":
        norm_per_frame=(
            np.linalg.norm(arr[:,5,:]-arr[:,6,:],axis=1)
            + np.linalg.norm(arr[:,6,:]-arr[:,7,:],axis=1)
            + np.linalg.norm(arr[:,7,:]-arr[:,8,:],axis=1)
        )
    elif strategy=="PALMSIZE":
        norm_per_frame=np.mean(np.stack([
            np.linalg.norm(arr[:,0,:]-arr[:,5,:],axis=1),
            np.linalg.norm(arr[:,0,:]-arr[:,9,:],axis=1),
            np.linalg.norm(arr[:,0,:]-arr[:,13,:],axis=1),
            np.linalg.norm(arr[:,0,:]-arr[:,17,:],axis=1),
        ],axis=1),axis=1)
    elif strategy=="THUMBSIZE":
        norm_per_frame=(
            np.linalg.norm(arr[:,2,:]-arr[:,3,:],axis=1)
            + np.linalg.norm(arr[:,3,:]-arr[:,4,:],axis=1)
        )
    else:
        raise ValueError(f"Unsupported norm_strategy={strategy}")

    normalization_factor=float(np.nanmax(norm_per_frame))
    if not np.isfinite(normalization_factor) or normalization_factor <= 0:
        raise RuntimeError("Invalid normalization factor.")

    duration=frame_idx/fps
    analyzer=analyzer_cls()
    output=analyzer.analyze(
        raw_signal.tolist(),
        normalization_factor,
        0.0,
        duration
    )
    if not output or "radarTable" not in output or output["radarTable"] is None:
        raise RuntimeError(
            "VisionMD PeakfinderSignalAnalyzer returned no kinematic features "
            "(likely insufficient complete movement cycles)."
        )

    feats=output["radarTable"]
    flat={k:float(feats[k]) for k in feats if np.isscalar(feats[k])}
    meta={
        "fps":fps,
        "frames":frame_idx,
        "duration_s":duration,
        "missing_fraction":missing_fraction,
        "detected_side":dominant_label(chosen_labels).lower(),
        "normalization_factor":normalization_factor,
        "n_detected_peaks":len(output.get("peaks",{}).get("data",[])),
    }
    return flat,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",required=True,choices=["train","val","test"])
    args=ap.parse_args()

    analyzer_cls=load_official_analyzer()
    df=read_manifest(args.split)
    rows=[]
    failures=[]

    for _,row in tqdm(df.iterrows(),total=len(df),desc=f"VisionMD HM:{args.split}"):
        p=ROOT/row["relative_path"]
        try:
            feats,meta=process_video(p,row,analyzer_cls)
            out={
                "filename":row["filename"],
                "relative_path":row["relative_path"],
                "label":int(row["label"]),
                "patient_id":str(row["patient_id"]),
                "requested_side":str(row.get("side","auto")),
                **meta,
                **feats,
            }
            rows.append(out)
        except Exception as e:
            failures.append({
                "filename":row["filename"],
                "relative_path":row["relative_path"],
                "label":row["label"],
                "patient_id":row["patient_id"],
                "side":row.get("side","auto"),
                "error_type":type(e).__name__,
                "error":str(e),
                "traceback":traceback.format_exc(limit=5).replace("\n"," | "),
            })

    feat_df=pd.DataFrame(rows)
    fail_df=pd.DataFrame(failures)
    feat_out=ROOT/"features"/f"{args.split}_features.csv"
    fail_out=ROOT/"results"/f"extraction_failures_{args.split}.csv"
    feat_df.to_csv(feat_out,index=False,encoding="utf-8-sig")
    fail_df.to_csv(fail_out,index=False,encoding="utf-8-sig")

    print(f"[DONE] {args.split}: success={len(rows)}, failed={len(failures)}")
    if len(df):
        print(f"Success rate: {len(rows)/len(df)*100:.2f}%")
    print("Features:",feat_out)
    print("Failures:",fail_out)
    if args.split=="test" and failures:
        print("[WARNING] Test failures exist. Do not describe metrics as full-test-set accuracy.")

if __name__=="__main__":
    main()
