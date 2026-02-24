"""Select the best cardio model by ROC AUC and save as best_cardio_model.joblib."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_CSV = MODEL_DIR / "cardio_results.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_cardio_model.joblib"
BEST_INFO_PATH = MODEL_DIR / "best_cardio_model_info.json"

# model_name in CSV  ->  joblib filename
FILENAME_MAP: dict[str, str] = {
    "logistic_regression": "cardio_logistic.joblib",
    "random_forest": "cardio_rf.joblib",
    "lightgbm": "cardio_lgbm.joblib",
    "xgboost": "cardio_xgb.joblib",
}


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        return 1

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("[ERROR] Results CSV is empty.")
        return 1

    # Find model with highest roc_auc
    best = df.sort_values("roc_auc", ascending=False).iloc[0]
    name = best["model_name"]
    print(f"[INFO] Best model by ROC AUC: {name} (roc_auc={best['roc_auc']:.4f})")

    # Resolve joblib filename
    filename = FILENAME_MAP.get(name, f"cardio_{name}.joblib")
    src_path = MODEL_DIR / filename

    if not src_path.exists():
        print(f"[ERROR] Model file not found: {src_path}")
        return 1

    # Copy to best_cardio_model.joblib (idempotent)
    shutil.copy2(src_path, BEST_MODEL_PATH)
    print(f"[INFO] Copied {filename} -> {BEST_MODEL_PATH.name}")

    # Build info JSON
    def _safe(col: str) -> float | None:
        """Return rounded float if column exists and is not NaN, else None."""
        if col in best.index and pd.notna(best[col]):
            return round(float(best[col]), 6)
        return None

    info = {
        "model_name": name,
        "source": "cardio",
        "roc_auc": _safe("roc_auc"),
        "accuracy": _safe("accuracy"),
        "precision": _safe("precision"),
        "recall": _safe("recall"),
        "f1": _safe("f1"),
        "cv_roc_auc": _safe("cv_roc_auc"),   # null when column absent
        "saved_from": filename,
    }

    BEST_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"[INFO] Saved {BEST_INFO_PATH.name}")
    print(f"\n[DONE] Best cardio model: {name}")
    print(json.dumps(info, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
