"""Train and evaluate models on the cleaned Cardiovascular Disease dataset.

Source      : ml/data/cardio_cleaned.csv
Models out  : ml/models/cardio_*.joblib
Leaderboard : ml/models/cardio_results.csv
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET_COL = "target"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cardio_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

# Mapping: model_name -> (estimator, output_filename)
MODELS: dict = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        "cardio_logistic.joblib",
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "cardio_rf.joblib",
    ),
    "lightgbm": (
        lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1),
        "cardio_lgbm.joblib",
    ),
    "xgboost": (
        XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED),
        "cardio_xgb.joblib",
    ),
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split features / target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"[INFO] Loaded {len(df)} rows, {X.shape[1]} features")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """StandardScaler for numeric columns, passthrough for the rest."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", "passthrough", categorical_cols),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Wrap preprocessor + estimator in one Pipeline."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    preprocessor = build_preprocessor(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results: list[dict] = []

    for model_name, (estimator, filename) in MODELS.items():
        print(f"\n{'='*50}")
        print(f"[INFO] Training: {model_name}")
        print(f"{'='*50}")

        pipeline = make_pipeline(preprocessor, estimator)

        # 5-fold cross-validation
        start = perf_counter()
        cv_scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_time = perf_counter() - start

        # Fit on full training data and save
        pipeline.fit(X, y)
        model_path = MODEL_DIR / filename
        joblib.dump(pipeline, model_path)

        row = {
            "model_name": model_name,
            "accuracy": float(np.mean(cv_scores["test_accuracy"])),
            "precision": float(np.mean(cv_scores["test_precision"])),
            "recall": float(np.mean(cv_scores["test_recall"])),
            "f1": float(np.mean(cv_scores["test_f1"])),
            "roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "cv_time_seconds": round(cv_time, 2),
        }
        results.append(row)

        print(f"  Accuracy  : {row['accuracy']:.4f}")
        print(f"  Precision : {row['precision']:.4f}")
        print(f"  Recall    : {row['recall']:.4f}")
        print(f"  F1        : {row['f1']:.4f}")
        print(f"  ROC AUC   : {row['roc_auc']:.4f} (+/- {row['roc_auc_std']:.4f})")
        print(f"  CV time   : {row['cv_time_seconds']}s")
        print(f"  Saved     : {model_path}")

    # ── Leaderboard ──────────────────────────────────────────────────
    results_df = (
        pd.DataFrame(results)
        .sort_values(by="roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = MODEL_DIR / "cardio_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*50}")
    print("[INFO] Leaderboard (by ROC AUC):")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))

    # ── Best model ───────────────────────────────────────────────────
    best = results_df.iloc[0]
    print(f"\n[BEST] {best['model_name']}  --  ROC AUC: {best['roc_auc']:.4f}")
    print(f"[INFO] Saved leaderboard: {csv_path}")


if __name__ == "__main__":
    main()
