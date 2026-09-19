from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
pred_path = ROOT / "results" / "predictions_raw.csv"
if not pred_path.exists():
    raise FileNotFoundError("Run scripts/03_run_official_fasteval.py first.")

df = pd.read_csv(pred_path)
usable = df.dropna(subset=["pred_score"]).copy()
usable["label"] = usable["label"].astype(int)
usable["pred_score"] = usable["pred_score"].astype(int)

# Primary paper-comparison metric: exact accuracy.
acc = accuracy_score(usable["label"], usable["pred_score"]) if len(usable) else float("nan")

# If FastEval outputs 3/3+ on our 0-2 dataset, exact accuracy naturally counts it as wrong.
macro_f1 = f1_score(
    usable["label"],
    usable["pred_score"],
    labels=[0,1,2],
    average="macro",
    zero_division=0
) if len(usable) else float("nan")

qwk = cohen_kappa_score(
    usable["label"],
    usable["pred_score"],
    weights="quadratic"
) if len(usable) > 1 else float("nan")

cm_labels = [0,1,2,3,4]
cm = confusion_matrix(usable["label"], usable["pred_score"], labels=cm_labels).tolist() if len(usable) else []

metrics = {
    "n_manifest": int(len(df)),
    "n_predictions_parsed": int(len(usable)),
    "n_unique_participants": int(df["patient_id"].nunique()),
    "exact_accuracy": float(acc),
    "exact_accuracy_percent": float(acc * 100) if acc == acc else None,
    "macro_f1_0_1_2": float(macro_f1),
    "qwk": float(qwk) if qwk == qwk else None,
    "confusion_matrix_labels": cm_labels,
    "confusion_matrix": cm,
    "note": (
        "This is zero-shot inference using the official FastEval model/pipeline. "
        "Exact accuracy is pred_score == ground-truth label; it is NOT FastEval AAC."
    )
}

out_json = ROOT / "results" / "metrics.json"
out_csv = ROOT / "results" / "predictions_scored.csv"
out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
usable.to_csv(out_csv, index=False)

print(json.dumps(metrics, indent=2, ensure_ascii=False))
print("\nSaved:", out_json)
print("Saved:", out_csv)
