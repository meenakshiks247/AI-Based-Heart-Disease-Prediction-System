from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_COL = "target"


def build_preprocessor(X_ref: pd.DataFrame) -> ColumnTransformer:
    """Construct an *unfitted* ColumnTransformer from training-set column types.

    Parameters
    ----------
    X_ref : pd.DataFrame
        Training features only — must **never** include test rows.
    """
    numeric_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_ref.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", "passthrough", categorical_cols),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / "data" / "train.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = model_dir / "preprocessor.joblib"

    # --- Load training data ---
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train_df = pd.read_csv(train_path)
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {train_path}")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    print(f"[INFO] Loaded training data: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    # --- Build & fit preprocessor on training data only ---
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    print("[INFO] Fitting preprocessor on training data only")

    # --- Persist ---
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[INFO] Saved preprocessor to {preprocessor_path}")

    # --- Validation: reload and smoke-test ---
    reloaded = joblib.load(preprocessor_path)
    sample = X_train.head(5)
    transformed = reloaded.transform(sample)
    assert transformed.shape[0] == sample.shape[0], (
        f"Row count mismatch after transform: expected {sample.shape[0]}, got {transformed.shape[0]}"
    )
    print(f"[INFO] Validation passed: transform produced {transformed.shape} from {sample.shape}")


if __name__ == "__main__":
    main()
