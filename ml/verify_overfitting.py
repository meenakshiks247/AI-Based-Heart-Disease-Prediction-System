from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load CSV dataset from disk."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def detect_target_column(df: pd.DataFrame) -> str:
    """Pick target column from common names."""
    for candidate in ("HeartDisease", "target"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Target column not found. Expected 'HeartDisease' or 'target'.")


def run_leakage_checks(df: pd.DataFrame, target_col: str) -> None:
    """Basic leakage checks before modeling."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from dataframe.")

    duplicate_rows = int(df.duplicated().sum())
    print(f"[CHECK] Duplicate rows in dataset: {duplicate_rows}")


def build_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target and verify no leakage by column inclusion."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if target_col in X.columns:
        raise ValueError(f"Data leakage detected: target column '{target_col}' is in features.")

    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing + RandomForest pipeline."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def plot_learning_curve(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    """Generate and save learning curve plot."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=SEED,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training Score")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_title("Learning Curve - Random Forest")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Learning curve saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify overfitting/data leakage for Random Forest model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml/data/heart_cleaned.csv"),
        help="Path to input CSV dataset.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("ml/models/random_forest_learning_curve.png"),
        help="Path to save learning curve image.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    target_col = detect_target_column(df)
    print(f"[INFO] Using target column: {target_col}")

    run_leakage_checks(df, target_col)
    X, y = build_features_target(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"[RESULT] Training accuracy: {train_acc:.6f}")
    print(f"[RESULT] Test accuracy:     {test_acc:.6f}")
    print(f"[RESULT] Accuracy gap:      {train_acc - test_acc:.6f}")

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"[RESULT] 5-fold CV accuracy scores: {np.round(cv_scores, 6)}")
    print(f"[RESULT] CV mean accuracy: {cv_scores.mean():.6f}")
    print(f"[RESULT] CV std accuracy:  {cv_scores.std():.6f}")

    print("\n[REPORT] Classification report (test set):")
    print(classification_report(y_test, test_pred))

    cm_output = args.plot.with_name("random_forest_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Confusion Matrix - Random Forest")
    fig.tight_layout()
    fig.savefig(cm_output, dpi=200)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to: {cm_output}")

    plot_learning_curve(pipeline, X, y, args.plot)


if __name__ == "__main__":
    main()
