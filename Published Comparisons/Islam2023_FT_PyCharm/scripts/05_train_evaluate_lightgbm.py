from pathlib import Path
import argparse
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    mean_absolute_error, confusion_matrix
)

from _common import ROOT

META = {"filename","relative_path","label","patient_id","side"}

def read_features(split):
    p = ROOT / "features" / f"{split}_features.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run feature extraction first.")
    return pd.read_csv(p, dtype={"patient_id":str})

def official_feature_names():
    p = ROOT/"third_party"/"finger-tapping-severity"/"severity_dataset_dropped_correlated_columns.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, nrows=2)
    cols = list(df.columns)
    start = "wrist_mvmnt_x_median"
    end = "acceleration_min_trimmed"
    if start in cols and end in cols:
        i, j = cols.index(start), cols.index(end)
        if i <= j:
            return cols[i:j+1]
    return [c for c in cols if c not in META and c not in {
        "Rating","Rating1","Rating2","Rating3","Rating4","Rating5",
        "Diagnosis","diagnosis","id"
    }]

def choose_feature_columns(train_df):
    published = official_feature_names()
    numeric_candidates = [
        c for c in train_df.columns
        if c not in META and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    if published:
        selected = [c for c in published if c in numeric_candidates]
        if selected:
            return selected, "official_reduced_feature_columns_intersection"
    return numeric_candidates, "all_extracted_numeric_features_fallback"

def round_to_class(pred):
    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0.0, 2.0)
    # Half-up for non-negative values, matching "nearest integer" intent.
    return np.floor(pred + 0.5).astype(int)

def score_continuous(y, pred):
    cls = round_to_class(pred)
    out = {
        "mae": float(mean_absolute_error(y, pred)),
        "accuracy": float(accuracy_score(y, cls)),
        "macro_f1": float(f1_score(y, cls, labels=[0,1,2], average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, cls, weights="quadratic")),
    }
    try:
        out["pearson_r"] = float(pearsonr(y, pred).statistic)
    except Exception:
        out["pearson_r"] = None
    return out

def build_model(params):
    base = dict(
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    base.update(params)
    return LGBMRegressor(**base)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed","val_tune"], default="val_tune")
    args = ap.parse_args()

    tr = read_features("train")
    va = read_features("val")
    te = read_features("test")

    feat_cols, feature_policy = choose_feature_columns(tr)
    common = [c for c in feat_cols if c in va.columns and c in te.columns]
    if not common:
        raise RuntimeError("No common Islam feature columns across train/val/test.")

    Xtr = tr[common].replace([np.inf,-np.inf], np.nan)
    Xva = va[common].replace([np.inf,-np.inf], np.nan)
    Xte = te[common].replace([np.inf,-np.inf], np.nan)
    ytr = tr["label"].astype(float).to_numpy()
    yva = va["label"].astype(float).to_numpy()
    yte = te["label"].astype(int).to_numpy()

    # Fit imputation ONLY on training data.
    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr)
    Xva_i = imp.transform(Xva)
    Xte_i = imp.transform(Xte)

    if args.mode == "fixed":
        candidates = [dict(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
        )]
    else:
        candidates = []
        for n_estimators, lr, leaves, max_depth in itertools.product(
            [100, 200, 400],
            [0.01, 0.03, 0.05],
            [15, 31],
            [-1, 5],
        ):
            candidates.append(dict(
                n_estimators=n_estimators,
                learning_rate=lr,
                num_leaves=leaves,
                max_depth=max_depth,
            ))

    trials = []
    best = None
    for params in candidates:
        model = build_model(params)
        model.fit(Xtr_i, ytr)
        pred = model.predict(Xva_i)
        met = score_continuous(yva, pred)
        row = {**params, **met}
        trials.append(row)
        # Primary selector = validation MAE; exact accuracy only breaks ties.
        key = (met["mae"], -met["accuracy"])
        if best is None or key < best[0]:
            best = (key, params, met)

    best_params = best[1]
    # Final model uses train + val after parameter selection.
    Xtv = np.vstack([Xtr_i, Xva_i])
    ytv = np.concatenate([ytr, yva])
    final = build_model(best_params)
    final.fit(Xtv, ytv)

    pred_cont = final.predict(Xte_i)
    pred_cls = round_to_class(pred_cont)
    met = score_continuous(yte, pred_cont)

    cm = confusion_matrix(yte, pred_cls, labels=[0,1,2])

    results_dir = ROOT/"results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_df = te[["filename","patient_id","side","label"]].copy()
    pred_df["pred_continuous"] = pred_cont
    pred_df["pred_class"] = pred_cls
    pred_df["correct"] = pred_df["pred_class"] == pred_df["label"]
    pred_df.to_csv(results_dir/"test_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        cm,
        index=["true_0","true_1","true_2"],
        columns=["pred_0","pred_1","pred_2"]
    ).to_csv(results_dir/"confusion_matrix.csv", encoding="utf-8-sig")

    pd.DataFrame(trials).sort_values(["mae","accuracy"], ascending=[True,False]).to_csv(
        results_dir/"validation_trials.csv", index=False, encoding="utf-8-sig"
    )

    (results_dir/"selected_features.txt").write_text(
        "\n".join(common), encoding="utf-8"
    )

    bundle = {
        "imputer": imp,
        "model": final,
        "feature_columns": common,
        "best_params": best_params,
        "feature_policy": feature_policy,
    }
    joblib.dump(bundle, results_dir/"model.joblib")

    metrics = {
        "method": "Islam et al. (2023) adapted reproduction",
        "model": "LightGBM regressor",
        "mode": args.mode,
        "feature_policy": feature_policy,
        "n_features": len(common),
        "best_validation_params": best_params,
        "n_train_features_success": int(len(tr)),
        "n_val_features_success": int(len(va)),
        "n_test_features_success": int(len(te)),
        "n_test_participants": int(te["patient_id"].nunique()),
        "exact_accuracy": met["accuracy"],
        "exact_accuracy_percent": met["accuracy"] * 100.0,
        "macro_f1": met["macro_f1"],
        "qwk": met["qwk"],
        "mae_continuous": met["mae"],
        "pearson_r": met["pearson_r"],
        "rounding": "clip continuous prediction to [0,2], then nearest integer (half-up)",
        "warning": (
            "If feature extraction failed for any held-out test clip, the reported metrics "
            "are not a complete 150-clip evaluation and must not be presented as such."
        ),
    }
    (results_dir/"metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nConfusion matrix:")
    print(cm)

if __name__ == "__main__":
    main()
