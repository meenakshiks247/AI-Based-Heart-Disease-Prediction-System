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


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "data" / "train.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_data(train_path)
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

    results: list[dict[str, float | str]] = []
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

        # Fit full training split and persist model for inference.
        pipeline.fit(X_train, y_train)
        model_path = model_dir / f"{model_name}_safe.joblib"
        joblib.dump(pipeline, model_path)

        mean_auc = float(np.mean(cv_scores["test_roc_auc"]))
        std_auc = float(np.std(cv_scores["test_roc_auc"]))
        print(f"[INFO] {model_name} mean ROC AUC: {mean_auc:.6f}")
        print(f"[INFO] {model_name} std ROC AUC:  {std_auc:.6f}")
        print(f"[INFO] Saved model: {model_path}")

        results.append(
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

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    out_csv = model_dir / "model_results_safe.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved CV results: {out_csv}")
    print("\n[INFO] CV leaderboard by ROC AUC:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
