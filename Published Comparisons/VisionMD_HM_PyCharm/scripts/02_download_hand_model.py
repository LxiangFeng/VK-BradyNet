from pathlib import Path
import requests
from _common import ROOT, CONFIG

url = CONFIG["hand_model_url"]
out = ROOT/"models"/"hand_landmarker.task"
out.parent.mkdir(parents=True, exist_ok=True)

if out.exists() and out.stat().st_size > 1_000_000:
    print("[OK] Model already exists:", out)
else:
    print("[INFO] Downloading MediaPipe Hand Landmarker...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
    print("[OK] Saved:", out, out.stat().st_size, "bytes")
