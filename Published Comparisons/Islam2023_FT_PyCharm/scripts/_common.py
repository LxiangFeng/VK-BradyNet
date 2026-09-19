from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def split_dir(split: str) -> Path:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    return ROOT / "data" / split

def manifest_path(split: str) -> Path:
    return ROOT / "data" / "manifests" / f"{split}_manifest.csv"

def patient_id_from_filename(filename: str) -> str:
    m = re.search(CONFIG["patient_id_regex"], filename)
    return m.group(1) if m else Path(filename).stem

def read_manifest(split: str) -> pd.DataFrame:
    p = manifest_path(split)
    if not p.exists():
        raise FileNotFoundError(f"Manifest missing: {p}. Run 02_build_manifest.py first.")
    return pd.read_csv(p, dtype={"patient_id": str})
