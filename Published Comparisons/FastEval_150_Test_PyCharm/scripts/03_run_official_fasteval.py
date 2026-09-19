from pathlib import Path
import pandas as pd
import subprocess
import re
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
repo = ROOT / "third_party" / "fast_eval_Parkinsonism"
hand_dir = repo / "src" / "lib" / "hand_predictor"
script = hand_dir / "hand_predictor.py"
manifest_path = ROOT / "data" / "test_manifest.csv"
video_root = ROOT / "data" / "test_videos"
results_root = ROOT / "results"
raw_root = results_root / "raw"
raw_root.mkdir(parents=True, exist_ok=True)

if not script.exists():
    raise FileNotFoundError(
        f"Official hand_predictor.py not found: {script}\n"
        "Run scripts/01_setup_official_repo.py first."
    )

df = pd.read_csv(manifest_path)

def parse_score(text: str):
    # Conservative parser. It only accepts an explicit score-like phrase.
    patterns = [
        r"(?:MDS[- ]?UPDRS(?:\s+item)?\s+score|estimated[_ ]?score|pred(?:icted)?[_ ]?score|score)\D{0,20}(3\+|[0-4])",
        r"(?:prediction|pred)\D{0,10}(3\+|[0-4])",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            v = m.group(1)
            return 3 if v == "3+" else int(v)
    return None

rows = []
for i, row in df.iterrows():
    filename = str(row["filename"])
    src = video_root / filename
    if not src.exists():
        rows.append({**row.to_dict(), "status":"missing_video", "pred_score":None, "stdout_file":""})
        continue

    stem = src.stem
    ext = src.suffix.lstrip(".")
    sample_input_dir = raw_root / f"{i:03d}_{stem}" / "input"
    sample_output_dir = raw_root / f"{i:03d}_{stem}" / "output"
    sample_input_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy instead of modifying original test clip.
    staged = sample_input_dir / src.name
    if not staged.exists():
        shutil.copy2(src, staged)

    side = str(row["side"])
    hand_pos = "1"  # Matches the official README quick-start example.
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(script),
        "--wkdir_path", str(hand_dir),
        "--seed", "42",
        "--filename", stem,
        "--ext", ext,
        "--hand_LR", side,
        "--hand_pos", hand_pos,
        "--input_root_path", str(sample_input_dir),
        "--output_root_path", str(sample_output_dir),
        "--mode", "single",
    ]

    print(f"[{i+1}/{len(df)}] {filename} | GT={row['label']} | {side}")
    proc = subprocess.run(
        cmd,
        cwd=str(hand_dir),
        capture_output=True,
        text=True,
        errors="replace"
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    stdout_path = raw_root / f"{i:03d}_{stem}" / "console.txt"
    stdout_path.write_text(text, encoding="utf-8", errors="replace")

    pred = parse_score(text)

    # Search generated text/json/csv files if stdout does not expose the score.
    if pred is None:
        for fp in sample_output_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in {".txt", ".json", ".csv", ".log"}:
                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                pred = parse_score(content)
                if pred is not None:
                    break

    status = "ok" if proc.returncode == 0 else f"returncode_{proc.returncode}"
    if pred is None and proc.returncode == 0:
        status = "ok_score_not_parsed"

    rows.append({
        "filename": filename,
        "label": int(row["label"]),
        "patient_id": row["patient_id"],
        "side": side,
        "pred_score": pred,
        "status": status,
        "stdout_file": str(stdout_path.relative_to(ROOT)),
    })

out = pd.DataFrame(rows)
results_root.mkdir(parents=True, exist_ok=True)
out_path = results_root / "predictions_raw.csv"
out.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(out["status"].value_counts(dropna=False).to_string())
nparsed = out["pred_score"].notna().sum()
print(f"Parsed predictions: {nparsed}/{len(out)}")
if nparsed < len(out):
    print("\nSome scores were not parsed automatically.")
    print("Send me one successful sample's results/raw/.../console.txt and output folder listing;")
    print("I can adapt the parser to the exact official-repo output format.")
