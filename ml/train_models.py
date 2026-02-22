from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
LEAK_CHECK = True


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and split features/target."""
    print(f"[INFO] Loading dataset from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Target column 'target' was not found in the dataset.")

    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return X, y


def build_preprocessor(X_ref: pd.DataFrame) -> ColumnTransformer:
    """Construct an *unfitted* preprocessor based on column types in *X_ref*.

    The caller must pass **training data only** so that column-type
    detection never peeks at the test set.  The returned object is
    unfitted — fitting must happen downstream (e.g. inside a Pipeline
    that is fit on X_train).
    """
    numeric_features = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X_ref.columns if col not in numeric_features]

    print(f"[INFO] Numeric features to scale: {len(numeric_features)}")
    print(f"[INFO] Non-numeric features passthrough: {len(categorical_features)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", "passthrough", categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    # Keep DataFrame output so downstream estimators consistently receive feature names.
    preprocessor.set_output(transform="pandas")
    return preprocessor


def run_leak_check(
    X_full: pd.DataFrame,
    X_train: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> None:
    """Smoke-check: compare preprocessor means fitted on full X vs X_train only."""
    try:
        full_pp = clone(preprocessor)
        full_pp.fit(X_full)
        train_pp = clone(preprocessor)
        train_pp.fit(X_train)

        # Extract scaler means from the numeric sub-pipeline
        full_means = full_pp.named_transformers_["num"]["scaler"].mean_
        train_means = train_pp.named_transformers_["num"]["scaler"].mean_

        if np.array_equal(full_means, train_means):
            print(
                "[WARNING] Preprocessor means from full X and train-only X are "
                "identical — potential data leakage!"
            )
        else:
            print("[INFO] Leak check passed: train-only preprocessor differs from full-X preprocessor.")
    except Exception as exc:
        print(f"[WARNING] Leak check could not run: {exc}")


def build_models() -> dict[str, Any]:
    """Return model registry."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "svm": SVC(probability=True, random_state=SEED),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "lightgbm": lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbosity=-1),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=SEED,
        ),
    }


def get_roc_auc(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Compute ROC AUC using predict_proba when available (fallback to decision_function)."""
    estimator = model_pipeline.named_steps["model"]

    if hasattr(estimator, "predict_proba"):
        y_scores = model_pipeline.predict_proba(X_test)
        if y_scores.ndim == 2 and y_scores.shape[1] > 1:
            return roc_auc_score(y_test, y_scores[:, 1])
        return roc_auc_score(y_test, y_scores)

    if hasattr(estimator, "decision_function"):
        y_scores = model_pipeline.decision_function(X_test)
        return roc_auc_score(y_test, y_scores)

    raise AttributeError("Model does not support predict_proba or decision_function for ROC AUC.")


def evaluate_model(
    model_name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_dir: Path,
) -> dict[str, float | str]:
    """Train, evaluate, and persist a single model."""
    print(f"\n[INFO] Training model: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - start

    y_pred = pipeline.predict(X_test)

    try:
        roc_auc = get_roc_auc(pipeline, X_test, y_test)
    except Exception as exc:
        print(f"[WARN] Could not compute ROC AUC for {model_name}: {exc}")
        roc_auc = float("nan")

    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "training_time_seconds": training_time_seconds,
    }

    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[INFO] Saved model: {model_path}")

    return results


def main() -> None:
    np.random.seed(SEED)

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "heart_cleaned.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory ready: {model_dir}")

    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    print(
        "[INFO] Train/test split complete: "
        f"train={X_train.shape[0]} rows, test={X_test.shape[0]} rows"
    )

    # Build preprocessor from X_train only — never peek at test data.
    preprocessor = build_preprocessor(X_train)
    print("[INFO] Fitting preprocessor on training data only")

    # Optional smoke-check: verify train-only means differ from full-X means.
    if LEAK_CHECK:
        run_leak_check(X, X_train, preprocessor)

    models = build_models()

    all_results: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        result = evaluate_model(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_dir=model_dir,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    csv_path = model_dir / "model_results.csv"
    json_path = model_dir / "model_results.json"

    results_df.to_csv(csv_path, index=False)
    results_df.to_json(json_path, orient="records", indent=2)

    # Save a standalone fitted preprocessor (fitted on X_train only).
    fitted_preprocessor = clone(preprocessor)
    fitted_preprocessor.fit(X_train)
    preprocessor_path = model_dir / "preprocessor.joblib"
    joblib.dump(fitted_preprocessor, preprocessor_path)
    print(f"[INFO] Saved preprocessor to {preprocessor_path}")

    print(f"\n[INFO] Saved model comparison CSV: {csv_path}")
    print(f"[INFO] Saved model comparison JSON: {json_path}")
    print("\n[INFO] Model ranking by ROC AUC (desc):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
