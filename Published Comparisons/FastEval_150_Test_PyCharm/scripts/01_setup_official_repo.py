from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
dst = ROOT / "third_party" / "fast_eval_Parkinsonism"
url = "https://github.com/yuyuan871111/fast_eval_Parkinsonism.git"

if dst.exists() and any(dst.iterdir()):
    print(f"[OK] Official repo already exists: {dst}")
    sys.exit(0)

dst.parent.mkdir(parents=True, exist_ok=True)
print("[INFO] Cloning official FastEval Parkinsonism repository...")
cmd = ["git", "clone", url, str(dst)]
print(" ".join(cmd))
subprocess.check_call(cmd)
print("[OK] Clone complete.")
print(f"[NEXT] Create the official environment from: {dst / 'environment.yml'}")
