from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


def test_model_artifacts_exist() -> None:
    """Ensure expected model artifacts are present without running training."""
    results_csv = MODELS_DIR / "model_results.csv"
    best_model = MODELS_DIR / "best_model.joblib"
    joblib_files = list(MODELS_DIR.glob("*.joblib"))

    assert results_csv.exists(), f"Missing required artifact: {results_csv}"
    assert best_model.exists() or len(joblib_files) > 0, (
        f"Expected {best_model} or at least one .joblib file in {MODELS_DIR}"
    )


def test_model_results_schema() -> None:
    """Validate model_results.csv contains required metric columns."""
    results_csv = MODELS_DIR / "model_results.csv"
    assert results_csv.exists(), f"Missing required artifact: {results_csv}"

    df = pd.read_csv(results_csv)
    required_columns = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    assert not df.empty, f"{results_csv} is empty."
    assert not missing_columns, (
        f"Missing columns in {results_csv}: {missing_columns}. "
        f"Found columns: {list(df.columns)}"
    )
