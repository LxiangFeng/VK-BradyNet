from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def split_dir(split):
    if split not in {"train","val","test"}:
        raise ValueError("split must be train, val, or test")
    return ROOT / "data" / split

def manifest_path(split):
    return ROOT / "data" / "manifests" / f"{split}_manifest.csv"

def patient_id_from_filename(filename):
    m = re.search(CONFIG["patient_id_regex"], filename)
    return m.group(1) if m else Path(filename).stem

def read_manifest(split):
    p = manifest_path(split)
    if not p.exists():
        raise FileNotFoundError(f"Missing manifest: {p}")
    return pd.read_csv(p, dtype={"patient_id":str})

def official_backend():
    return ROOT / "third_party" / "VisionMD-DesktopApp" / "VisionMD-DesktopApp-BackEnd"
