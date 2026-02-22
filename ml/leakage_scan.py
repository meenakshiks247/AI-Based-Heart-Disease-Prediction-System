from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("ml/data/heart_cleaned_dedup.csv")
OUT_PATH = Path("ml/models/leakage_scan.csv")
TARGET_COL = "target"
SEED = 42


def compute_correlation(feature: pd.Series, target: pd.Series) -> float:
    """Compute feature-target correlation using numeric conversion fallback."""
    try:
        x = pd.to_numeric(feature, errors="coerce")
        y = pd.to_numeric(target, errors="coerce")
        corr = x.corr(y)
        return float(corr) if pd.notna(corr) else np.nan
    except Exception:
        return np.nan


def compute_mutual_information(feature: pd.Series, target: pd.Series) -> float:
    """Compute mutual information between one feature and target."""
    x = feature.copy()
    if pd.api.types.is_numeric_dtype(x):
        x = pd.to_numeric(x, errors="coerce").fillna(x.median() if x.notna().any() else 0)
        X = x.to_numpy().reshape(-1, 1)
        discrete = False
    else:
        x = x.fillna("__NA__").astype(str)
        codes, _ = pd.factorize(x)
        X = codes.reshape(-1, 1)
        discrete = True

    mi = mutual_info_classif(X, target.to_numpy(), discrete_features=discrete, random_state=SEED)
    return float(mi[0])


def single_feature_cv_auc(feature: pd.Series, target: pd.Series) -> float:
    """Train Logistic Regression on a single feature and return 5-fold CV ROC-AUC mean."""
    X = feature.to_frame(name=feature.name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if pd.api.types.is_numeric_dtype(feature):
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        )
    else:
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        )

    scores = cross_val_score(pipe, X, target, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def equal_or_nearly_equal(feature: pd.Series, target: pd.Series) -> tuple[bool, bool, float]:
    """Check exact equality and near-equality ratio after numeric coercion."""
    fx = pd.to_numeric(feature, errors="coerce")
    ty = pd.to_numeric(target, errors="coerce")
    both = pd.DataFrame({"f": fx, "t": ty}).dropna()
    if both.empty:
        return False, False, 0.0

    exact = bool((both["f"] == both["t"]).all())
    match_ratio = float((both["f"] == both["t"]).mean())
    near = match_ratio >= 0.98
    return exact, near, match_ratio


def main() -> int:
    if not DATA_PATH.exists():
        print(f"[ERROR] Missing dataset: {DATA_PATH}")
        return 2

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        print(f"[ERROR] Missing target column '{TARGET_COL}' in {DATA_PATH}")
        return 2

    y = df[TARGET_COL]
    features = [c for c in df.columns if c != TARGET_COL]
    if not features:
        print("[ERROR] No feature columns found.")
        return 2

    print(f"[INFO] Loaded dataset: {DATA_PATH} ({len(df)} rows)")
    print(f"[INFO] Scanning {len(features)} feature(s) for leakage signals...")

    rows: list[dict[str, object]] = []
    for col in features:
        x = df[col]
        corr = compute_correlation(x, y)
        mi = compute_mutual_information(x, y)
        auc = single_feature_cv_auc(x, y)
        exact_eq, near_eq, match_ratio = equal_or_nearly_equal(x, y)
        suspicious_auc = auc > 0.90

        rows.append(
            {
                "feature": col,
                "correlation_with_target": corr,
                "mutual_information": mi,
                "single_feature_cv_roc_auc": auc,
                "suspicious_auc_gt_0_90": suspicious_auc,
                "exactly_equal_target": exact_eq,
                "nearly_equal_target": near_eq,
                "target_match_ratio": match_ratio,
            }
        )

    results = pd.DataFrame(rows).sort_values("single_feature_cv_roc_auc", ascending=False).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    suspicious = results[
        (results["suspicious_auc_gt_0_90"])
        | (results["exactly_equal_target"])
        | (results["nearly_equal_target"])
    ]

    print(f"[INFO] Saved scan results: {OUT_PATH}")
    if suspicious.empty:
        print("[INFO] No suspicious features found by configured rules.")
    else:
        print("\n[WARN] Suspicious features detected:")
        print(
            suspicious[
                [
                    "feature",
                    "single_feature_cv_roc_auc",
                    "correlation_with_target",
                    "mutual_information",
                    "exactly_equal_target",
                    "nearly_equal_target",
                    "target_match_ratio",
                ]
            ].to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
