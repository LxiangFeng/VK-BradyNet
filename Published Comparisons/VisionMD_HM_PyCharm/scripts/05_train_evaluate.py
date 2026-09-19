import argparse, itertools, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support
)

from _common import ROOT, CONFIG

META = {
    "filename","relative_path","label","patient_id",
    "requested_side","detected_side","fps","frames","duration_s",
    "missing_fraction","normalization_factor","n_detected_peaks"
}

def load(split):
    p=ROOT/"features"/f"{split}_features.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing.")
    return pd.read_csv(p,dtype={"patient_id":str})

def metrics(y,p):
    pr,rc,f1c,_=precision_recall_fscore_support(
        y,p,labels=[0,1,2],zero_division=0
    )
    return {
        "accuracy":float(accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)),
        "qwk":float(cohen_kappa_score(y,p,weights="quadratic")),
        "class_precision":[float(x) for x in pr],
        "class_recall":[float(x) for x in rc],
        "class_f1":[float(x) for x in f1c],
    }

def model(params):
    base=dict(
        objective="multiclass",
        num_class=3,
        random_state=int(CONFIG["random_state"]),
        n_jobs=-1,
        verbosity=-1,
        class_weight=None,
    )
    base.update(params)
    return LGBMClassifier(**base)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["fixed","val_tune"],default="val_tune")
    args=ap.parse_args()

    tr,va,te=load("train"),load("val"),load("test")

    feature_cols=[
        c for c in tr.columns
        if c not in META and pd.api.types.is_numeric_dtype(tr[c])
        and c in va.columns and c in te.columns
    ]
    # Drop accidental metadata-like numeric fields if present.
    feature_cols=[c for c in feature_cols if c not in {"label"}]
    if not feature_cols:
        raise RuntimeError("No common VisionMD kinematic features found.")

    Xtr=tr[feature_cols].replace([np.inf,-np.inf],np.nan)
    Xva=va[feature_cols].replace([np.inf,-np.inf],np.nan)
    Xte=te[feature_cols].replace([np.inf,-np.inf],np.nan)
    ytr=tr.label.astype(int).to_numpy()
    yva=va.label.astype(int).to_numpy()
    yte=te.label.astype(int).to_numpy()

    imp=SimpleImputer(strategy="median")
    Xtr_i=imp.fit_transform(Xtr)
    Xva_i=imp.transform(Xva)
    Xte_i=imp.transform(Xte)

    if args.mode=="fixed":
        candidates=[dict(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=-1,
            subsample=1.0,
            colsample_bytree=1.0,
        )]
    else:
        candidates=[]
        for n_estimators,lr,leaves,depth in itertools.product(
            [100,200,400],
            [0.01,0.03,0.05],
            [7,15,31],
            [-1,5]
        ):
            candidates.append(dict(
                n_estimators=n_estimators,
                learning_rate=lr,
                num_leaves=leaves,
                max_depth=depth,
            ))

    trials=[]
    best=None
    for params in candidates:
        m=model(params)
        m.fit(Xtr_i,ytr)
        pred=m.predict(Xva_i)
        met=metrics(yva,pred)
        trials.append({**params,**{k:v for k,v in met.items() if not isinstance(v,list)}})
        # maximize macro-F1, then accuracy
        key=(met["macro_f1"],met["accuracy"])
        if best is None or key>best[0]:
            best=(key,params,met)

    best_params=best[1]

    # Refit selected model on train+validation after parameter selection.
    Xtv=np.vstack([Xtr_i,Xva_i])
    ytv=np.concatenate([ytr,yva])
    final=model(best_params)
    final.fit(Xtv,ytv)
    pred=final.predict(Xte_i).astype(int)
    met=metrics(yte,pred)

    outdir=ROOT/"results"
    outdir.mkdir(parents=True,exist_ok=True)

    pred_df=te[["filename","patient_id","label"]].copy()
    pred_df["pred_class"]=pred
    pred_df["correct"]=pred_df.label.astype(int)==pred
    pred_df.to_csv(outdir/"test_predictions.csv",index=False,encoding="utf-8-sig")

    cm=confusion_matrix(yte,pred,labels=[0,1,2])
    pd.DataFrame(cm,index=["true_0","true_1","true_2"],
                 columns=["pred_0","pred_1","pred_2"]).to_csv(
        outdir/"confusion_matrix.csv",encoding="utf-8-sig"
    )

    pd.DataFrame(trials).sort_values(
        ["macro_f1","accuracy"],ascending=[False,False]
    ).to_csv(outdir/"validation_trials.csv",index=False,encoding="utf-8-sig")

    joblib.dump(
        {"imputer":imp,"model":final,"feature_columns":feature_cols,
         "best_params":best_params},
        outdir/"model.joblib"
    )

    expected_test=int(CONFIG["expected_split_sizes"]["test"])
    full_test=(len(te)==expected_test)
    result={
        "method":"VisionMD HM kinematic features + LightGBM",
        "evaluation_setting":"Published-tool reproduction",
        "best_validation_params":best_params,
        "n_features":len(feature_cols),
        "feature_columns":feature_cols,
        "n_train_feature_success":len(tr),
        "n_val_feature_success":len(va),
        "n_test_feature_success":len(te),
        "expected_test_clips":expected_test,
        "full_test_feature_extraction":bool(full_test),
        "n_test_participants":int(te.patient_id.nunique()),
        "exact_accuracy":met["accuracy"],
        "exact_accuracy_percent":met["accuracy"]*100,
        "macro_f1":met["macro_f1"],
        "qwk":met["qwk"],
        "class_precision":met["class_precision"],
        "class_recall":met["class_recall"],
        "class_f1":met["class_f1"],
        "warning":(
            "VisionMD itself is a kinematic analysis tool; LightGBM is an added "
            "downstream classifier for common severity evaluation. "
            "If full_test_feature_extraction is false, do not report this as "
            f"accuracy on all {expected_test} held-out clips."
        ),
    }
    (outdir/"metrics.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("\nConfusion matrix:\n",cm)

if __name__=="__main__":
    main()
