from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "results" / "metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError("Run scripts/04_evaluate_exact_accuracy.py first.")

m = json.loads(metrics_path.read_text(encoding="utf-8"))
acc = m.get("exact_accuracy_percent")
acc_text = "TBD" if acc is None else f"{acc:.2f}"
n = m.get("n_manifest", "TBD")
p = m.get("n_unique_participants", "TBD")

print("\nTable 5 row:")
print(
    f"Zero-shot published-model transfer | FastEval Parkinsonism | "
    f"FastEval original cohort | Our held-out FT test set | FT | "
    f"{n} clips / {p} participants | {acc_text}"
)
