from pathlib import Path
import json
import pandas as pd
from _common import ROOT

mfile = ROOT/"results"/"metrics.json"
pfile = ROOT/"results"/"test_predictions.csv"
if not mfile.exists() or not pfile.exists():
    raise FileNotFoundError("Run 05_train_evaluate_lightgbm.py first.")

m = json.loads(mfile.read_text(encoding="utf-8"))
df = pd.read_csv(pfile, dtype={"patient_id":str})
n = len(df)
p = df["patient_id"].nunique()
acc = m["exact_accuracy_percent"]

print("Table 5 row:")
print(
    f"Published-method reproduction | Islam et al. (2023) | "
    f"Our FT cohort | Our held-out FT test set | FT | "
    f"{n} clips / {p} participants | {acc:.2f}"
)
