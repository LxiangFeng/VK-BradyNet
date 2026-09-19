import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from _common import ROOT

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    p=ROOT/"results"/"test_predictions.csv"
    if not p.exists():
        raise FileNotFoundError("Run 05_train_evaluate.py first.")
    df=pd.read_csv(p,dtype={"patient_id":str})
    rng=np.random.default_rng(args.seed)
    patients=df.patient_id.unique()
    values=[]

    for _ in range(args.n):
        sampled=rng.choice(patients,size=len(patients),replace=True)
        chunks=[]
        # duplicated patient clusters must remain duplicated
        for new_i,pid in enumerate(sampled):
            c=df[df.patient_id==pid].copy()
            c["_boot_cluster"]=new_i
            chunks.append(c)
        b=pd.concat(chunks,ignore_index=True)
        values.append(accuracy_score(b.label.astype(int),b.pred_class.astype(int)))

    lo,hi=np.quantile(values,[0.025,0.975])
    out={
        "cluster_unit":"patient_id",
        "replicates":args.n,
        "accuracy_95ci_percent":[float(lo*100),float(hi*100)]
    }
    (ROOT/"results"/"bootstrap_accuracy_ci.json").write_text(
        json.dumps(out,indent=2),encoding="utf-8"
    )
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
