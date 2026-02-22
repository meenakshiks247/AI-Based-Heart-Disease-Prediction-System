from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

# Prefer safe results; fall back to legacy.
_SAFE_CSV = MODELS_DIR / "model_results_safe.csv"
_LEGACY_CSV = MODELS_DIR / "model_results.csv"


def _resolve_results_csv() -> Path:
    """Return whichever results CSV exists, preferring safe."""
    if _SAFE_CSV.exists():
        print("Using SAFE model results")
        return _SAFE_CSV
    if _LEGACY_CSV.exists():
        print("Using legacy model results")
        return _LEGACY_CSV
    raise FileNotFoundError(
        f"Neither {_SAFE_CSV.name} nor {_LEGACY_CSV.name} found in {MODELS_DIR}"
    )


def test_model_artifacts_exist() -> None:
    """Ensure expected model artifacts are present without running training."""
    results_csv = _resolve_results_csv()
    best_model = MODELS_DIR / "best_model.joblib"
    joblib_files = list(MODELS_DIR.glob("*.joblib"))

    assert results_csv.exists(), f"Missing required artifact: {results_csv}"
    assert best_model.exists() or len(joblib_files) > 0, (
        f"Expected {best_model} or at least one .joblib file in {MODELS_DIR}"
    )


def test_model_results_schema() -> None:
    """Validate model results CSV contains required metric columns."""
    results_csv = _resolve_results_csv()
    assert results_csv.exists(), f"Missing required artifact: {results_csv}"

    df = pd.read_csv(results_csv)
    required_columns = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    assert not df.empty, f"{results_csv} is empty."
    assert not missing_columns, (
        f"Missing columns in {results_csv}: {missing_columns}. "
        f"Found columns: {list(df.columns)}"
    )
