from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
TARGET_COL = "target"


def load_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load training dataset."""
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_df = pd.read_csv(train_path)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Training dataset must include target column '{TARGET_COL}'.")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    return X_train, y_train


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric features and passthrough non-numeric features."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", "passthrough", categorical_cols),
        ],
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, Any]:
    """Define model registry."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "svm": SVC(probability=True, random_state=SEED),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "lightgbm": lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1),
        "xgboost": XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            random_state=SEED,
        ),
    }


def make_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    """Create model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def holdout_roc_auc(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Compute ROC AUC on holdout, handling models without predict_proba."""
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)
        return roc_auc_score(y_test, y_scores[:, 1] if y_scores.ndim == 2 else y_scores)
    if hasattr(estimator, "decision_function"):
        return roc_auc_score(y_test, pipeline.decision_function(X_test))
    return float("nan")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "data" / "train.csv"
    test_path = base_dir / "data" / "test.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_data(train_path)

    # Load holdout test set
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    print(f"[INFO] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    preprocessor = build_preprocessor(X_train)
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results: list[dict[str, float | str]] = []
    holdout_results: list[dict[str, float | str]] = []
    pipelines: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"\n[INFO] Training + CV for: {model_name}")
        pipeline = make_pipeline(preprocessor, model)

        start = perf_counter()
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_time = perf_counter() - start

        # Fit on full training set and persist
        pipeline.fit(X_train, y_train)
        model_path = model_dir / f"{model_name}_safe.joblib"
        joblib.dump(pipeline, model_path)
        pipelines[model_name] = pipeline

        mean_auc = float(np.mean(cv_scores["test_roc_auc"]))
        std_auc = float(np.std(cv_scores["test_roc_auc"]))
        print(f"[INFO] {model_name} CV ROC AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

        cv_results.append(
            {
                "model_name": model_name,
                "accuracy": float(np.mean(cv_scores["test_accuracy"])),
                "precision": float(np.mean(cv_scores["test_precision"])),
                "recall": float(np.mean(cv_scores["test_recall"])),
                "f1": float(np.mean(cv_scores["test_f1"])),
                "roc_auc": mean_auc,
                "roc_auc_std": std_auc,
                "cv_time_seconds": cv_time,
            }
        )

        # ── Holdout evaluation ───────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        h_roc = holdout_roc_auc(pipeline, X_test, y_test)
        print(f"[INFO] {model_name} HOLDOUT ROC AUC: {h_roc:.4f}")

        holdout_results.append(
            {
                "model_name": model_name,
                "holdout_accuracy": accuracy_score(y_test, y_pred),
                "holdout_precision": precision_score(y_test, y_pred, zero_division=0),
                "holdout_recall": recall_score(y_test, y_pred, zero_division=0),
                "holdout_f1": f1_score(y_test, y_pred, zero_division=0),
                "holdout_roc_auc": h_roc,
            }
        )

    # ── Save CV results ──────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    cv_csv = model_dir / "model_results_safe.csv"
    cv_df.to_csv(cv_csv, index=False)
    print(f"\n[INFO] Saved CV results: {cv_csv}")
    print("\n[INFO] === CV Leaderboard (ROC AUC) ===")
    print(cv_df.to_string(index=False))

    # ── Save holdout results ─────────────────────────────────────────
    ho_df = pd.DataFrame(holdout_results).sort_values(by="holdout_roc_auc", ascending=False).reset_index(drop=True)
    ho_csv = model_dir / "model_results_holdout.csv"
    ho_df.to_csv(ho_csv, index=False)
    print(f"\n[INFO] Saved holdout results: {ho_csv}")
    print("\n[INFO] === Holdout Leaderboard (ROC AUC) ===")
    print(ho_df.to_string(index=False))


if __name__ == "__main__":
    main()
