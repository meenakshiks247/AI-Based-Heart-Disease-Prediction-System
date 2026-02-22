from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def load_results(results_path: Path) -> pd.DataFrame:
    """Load and validate model results CSV."""
    if not results_path.exists():
        raise FileNotFoundError(f"Model results file not found: {results_path}")

    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(f"Model results file is empty: {results_path}")

    required_cols = {"model_name", "accuracy", "precision", "recall", "f1", "roc_auc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {sorted(missing)}")

    return df


def select_best_model(df: pd.DataFrame) -> pd.Series:
    """Select best model by highest ROC AUC."""
    sorted_df = df.sort_values(by=["roc_auc"], ascending=[False], kind="mergesort").reset_index(drop=True)
    return sorted_df.iloc[0]


def copy_best_model(model_name: str, model_dir: Path) -> Path:
    """Copy selected safe model file to best_model.joblib."""
    src = model_dir / f"{model_name}_safe.joblib"
    dst = model_dir / "best_model.joblib"

    if not src.exists():
        raise FileNotFoundError(f"Selected model file not found: {src}")

    shutil.copy2(src, dst)
    return dst


def save_best_info(best_row: pd.Series, output_path: Path) -> None:
    """Persist selected model metadata to JSON."""
    best_info = {
        "model_name": str(best_row["model_name"]),
        "roc_auc": float(best_row["roc_auc"]),
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    results_path = model_dir / "model_results_safe.csv"
    best_info_path = model_dir / "best_model_info.json"

    try:
        df = load_results(results_path)
        best_row = select_best_model(df)
        copied_path = copy_best_model(str(best_row["model_name"]), model_dir)
        save_best_info(best_row, best_info_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return

    print("SAFE BEST MODEL SELECTED")
    print(f"[INFO] Copied best model to: {copied_path}")
    print(f"[INFO] Saved best model info to: {best_info_path}")
    print("[INFO] Best model metrics:")
    print(f"  model_name: {best_row['model_name']}")
    print(f"  accuracy: {float(best_row['accuracy']):.6f}")
    print(f"  f1: {float(best_row['f1']):.6f}")
    print(f"  roc_auc: {float(best_row['roc_auc']):.6f}")
    print(f"  precision: {float(best_row['precision']):.6f}")
    print(f"  recall: {float(best_row['recall']):.6f}")


if __name__ == "__main__":
    main()
