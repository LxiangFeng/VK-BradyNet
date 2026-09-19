import json
import pandas as pd
from _common import ROOT

mfile=ROOT/"results"/"metrics.json"
pfile=ROOT/"results"/"test_predictions.csv"
if not mfile.exists() or not pfile.exists():
    raise FileNotFoundError("Run 05_train_evaluate.py first.")

m=json.loads(mfile.read_text(encoding="utf-8"))
df=pd.read_csv(pfile,dtype={"patient_id":str})
acc=m["exact_accuracy_percent"]
n=len(df)
p=df.patient_id.nunique()

print("Recommended Table 5 row:")
print(
    f"Published-tool reproduction | VisionMD (2025) | "
    f"VisionMD kinematics + LightGBM | Our HM cohort | "
    f"Our held-out HM test set | HM | {n} clips / {p} participants | {acc:.2f}"
)

if not m.get("full_test_feature_extraction",False):
    print("\n[WARNING] Feature extraction did not succeed for the full expected test set.")
    print("Do not present this row as a full held-out-test-set result without qualification.")
