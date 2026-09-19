from pathlib import Path
import subprocess, shutil, json
from _common import ROOT, CONFIG

dst = ROOT/"third_party"/"VisionMD-DesktopApp"
url = CONFIG["visionmd_repo_url"]

if dst.exists() and (dst/".git").exists():
    print("[INFO] VisionMD repository already exists. Updating commit record only.")
else:
    if shutil.which("git") is None:
        raise RuntimeError(
            "Git not found. Install Git or manually clone:\n"
            f"{url}\ninto:\n{dst}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git","clone","--depth","1",url,str(dst)])

commit = subprocess.check_output(
    ["git","-C",str(dst),"rev-parse","HEAD"], text=True
).strip()
(ROOT/"third_party"/"VISIONMD_COMMIT.txt").write_text(commit+"\n", encoding="utf-8")

backend = dst/"VisionMD-DesktopApp-BackEnd"
required = [
    backend/"app/analysis/tasks/hand_movement_left.py",
    backend/"app/analysis/tasks/hand_movement_right.py",
    backend/"app/analysis/signal_analyzers/peakfinder_signal_analyzer.py",
]
missing = [str(x) for x in required if not x.exists()]
if missing:
    raise RuntimeError("Required VisionMD files not found:\n" + "\n".join(missing))

print("[OK] VisionMD source ready.")
print("Commit:", commit)
