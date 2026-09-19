from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
dst = ROOT / "third_party" / "finger-tapping-severity"
url = "https://github.com/ROC-HCI/finger-tapping-severity.git"

required = [
    "feature_extraction.py",
    "model_training.py",
    "severity_dataset_dropped_correlated_columns.csv",
]

if dst.exists() and all((dst/x).exists() for x in required):
    print("[OK] Official Islam et al. repository is already present.")
    sys.exit(0)

if shutil.which("git") is None:
    print("[ERROR] git was not found.")
    print("Install Git, or manually download the official repository ZIP:")
    print("https://github.com/ROC-HCI/finger-tapping-severity")
    print("and extract it to:")
    print(dst)
    sys.exit(1)

dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists():
    shutil.rmtree(dst)

print("[INFO] Cloning official repository...")
subprocess.check_call(["git", "clone", "--depth", "1", url, str(dst)])

missing = [x for x in required if not (dst/x).exists()]
if missing:
    raise RuntimeError(f"Official repo cloned, but required files are missing: {missing}")

print("[OK] Official repository ready:", dst)
