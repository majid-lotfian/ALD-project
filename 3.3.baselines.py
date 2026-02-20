#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb


def set_seed(seed: int):
    np.random.seed(seed)


def make_sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency per-sample weights (normalized)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w_class = 1.0 / counts
    w_class = w_class * (n_classes / w_class.sum())
    return w_class[y].astype(np.float64)


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(
            np.diag(cm),
            cm.sum(axis=1),
            out=np.zeros(cm.shape[0], dtype=np.float64),
            where=(cm.sum(axis=1) != 0),
        )
    return float(recalls.mean())


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--real_csv", type=str, required=True)

    ap.add_argument("--sex_col", type=str, default="sex")
    ap.add_argument("--severity_col", type=str, default="severity")
    ap.add_argument("--id_col", type=str, default="Sample_ID")

    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--model", type=str, default="xgb",
                    choices=["logreg", "rf", "xgb", "lgbm"])

    ap.add_argument("--use_class_weights", action="store_true",
                    help="Apply class/sample weighting for imbalance where supported.")

    ap.add_argument("--out_dir", type=str, default="./baselines/tabular_cv")

    # Logistic Regression
    ap.add_argument("--logreg_C", type=float, default=1.0)
    ap.add_argument("--logreg_max_iter", type=int, default=5000)

    # Random Forest
    ap.add_argument("--rf_n_estimators", type=int, default=1000)
    ap.add_argument("--rf_max_depth", type=int, default=None)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=1)

    # XGBoost
    ap.add_argument("--xgb_n_estimators", type=int, default=800)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_learning_rate", type=float, default=0.03)
    ap.add_argument("--xgb_subsample", type=float, default=0.9)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    # LightGBM
    ap.add_argument("--lgbm_n_estimators", type=int, default=1200)
    ap.add_argument("--lgbm_num_leaves", type=int, default=31)
    ap.add_argument("--lgbm_learning_rate", type=float, default=0.03)
    ap.add_argument("--lgbm_subsample", type=float, default=0.9)
    ap.add_argument("--lgbm_colsample_bytree", type=float, default=0.9)
    ap.add_argument("--lgbm_reg_lambda", type=float, default=0.0)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.real_csv)

    for c in [args.severity_col, args.sex_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {args.real_csv}")

    # Features: all numeric except severity/id
    drop_cols = {args.severity_col, args.id_col}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    if args.sex_col not in feature_cols:
        if pd.api.types.is_numeric_dtype(df[args.sex_col]):
            feature_cols.append(args.sex_col)
        else:
            raise ValueError(f"sex_col='{args.sex_col}' is not numeric; encode it first.")

    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df[args.severity_col].astype(int).to_numpy()

    uniq = np.unique(y)
    if uniq.min() != 0:
        raise ValueError(f"Expected severity labels starting at 0, got {uniq}")
    n_classes = int(y.max() + 1)
    if n_classes < 2:
        raise ValueError("severity has <2 classes.")

    print(f"Using {len(feature_cols)} numeric features. Example: {feature_cols[:10]}", flush=True)
    print(f"Class counts: {np.bincount(y, minlength=n_classes).tolist()}", flush=True)

    skf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    fold_metrics: List[Dict] = []
    all_cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # weights
        sample_w = None
        class_w_dict = None
        if args.use_class_weights:
            sample_w = make_sample_weights(ytr, n_classes=n_classes)
            counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
            counts[counts == 0] = 1.0
            w_class = 1.0 / counts
            w_class = w_class * (n_classes / w_class.sum())
            class_w_dict = {i: float(w_class[i]) for i in range(n_classes)}

        if args.model == "logreg":
            # multinomial, handles class_weight directly
            clf = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                C=args.logreg_C,
                max_iter=args.logreg_max_iter,
                n_jobs=-1,
                class_weight="balanced" if args.use_class_weights else None,
                random_state=args.seed,
            )
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)

        elif args.model == "rf":
            clf = RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                random_state=args.seed,
                n_jobs=-1,
                class_weight=("balanced_subsample" if args.use_class_weights else None),
            )
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)

        elif args.model == "xgb":
            clf = xgb.XGBClassifier(
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                reg_lambda=args.xgb_reg_lambda,
                objective="multi:softprob",
                num_class=n_classes,
                tree_method="hist",
                random_state=args.seed,
                n_jobs=-1,
            )
            if sample_w is not None:
                clf.fit(Xtr, ytr, sample_weight=sample_w)
            else:
                clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)

        else:  # lgbm
            clf = lgb.LGBMClassifier(
                n_estimators=args.lgbm_n_estimators,
                num_leaves=args.lgbm_num_leaves,
                learning_rate=args.lgbm_learning_rate,
                subsample=args.lgbm_subsample,
                colsample_bytree=args.lgbm_colsample_bytree,
                reg_lambda=args.lgbm_reg_lambda,
                objective="multiclass",
                num_class=n_classes,
                random_state=args.seed,
                n_jobs=-1,
                class_weight=class_w_dict if args.use_class_weights else None,
            )
            if sample_w is not None:
                clf.fit(Xtr, ytr, sample_weight=sample_w)
            else:
                clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)

        cm = confusion_matrix(yva, pred, labels=list(range(n_classes)))
        all_cm += cm

        bal = balanced_accuracy_from_cm(cm)
        f1m = f1_score(yva, pred, average="macro", labels=list(range(n_classes)), zero_division=0)

        fold_metrics.append({
            "fold": fold,
            "balanced_accuracy": float(bal),
            "macro_f1": float(f1m),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
        })

        print(f"[Fold {fold}] bal_acc={bal:.4f} macro_f1={f1m:.4f}", flush=True)

    bal_list = [m["balanced_accuracy"] for m in fold_metrics]
    f1_list = [m["macro_f1"] for m in fold_metrics]

    summary = {
        "model": args.model,
        "use_class_weights": bool(args.use_class_weights),
        "splits_requested": args.splits,
        "splits_used": args.splits,
        "balanced_accuracy_mean": float(np.mean(bal_list)),
        "balanced_accuracy_std": float(np.std(bal_list, ddof=1)) if len(bal_list) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1_list)),
        "macro_f1_std": float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0,
        "folds": fold_metrics,
        "confusion_matrix_sum": all_cm.tolist(),
        "feature_cols_used": feature_cols,
        "args": vars(args),
    }

    out_json = os.path.join(args.out_dir, "cv_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CV Summary ===")
    print(json.dumps({k: summary[k] for k in summary if k.endswith(("_mean", "_std"))}, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()