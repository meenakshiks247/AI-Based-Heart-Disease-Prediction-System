from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────


def resolve_results_source(model_dir: Path) -> tuple[Path, str]:
    """Return (results_csv_path, source) preferring safe over legacy.

    Raises FileNotFoundError with remediation hint when neither exists.
    """
    safe_path = model_dir / "model_results_safe.csv"
    legacy_path = model_dir / "model_results.csv"

    if safe_path.exists():
        print("[INFO] Using safe model results")
        return safe_path, "safe"

    if legacy_path.exists():
        print("[WARN] Falling back to legacy results")
        return legacy_path, "legacy"

    raise FileNotFoundError(
        f"Neither '{safe_path.name}' nor '{legacy_path.name}' found in {model_dir}.\n"
        "  Suggested fix: run  python ml/run_training.py  to generate safe models."
    )


def load_and_validate(results_path: Path) -> pd.DataFrame:
    """Load results CSV and validate required columns."""
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(f"Model results file is empty: {results_path}")

    required_cols = {"model_name", "roc_auc", "f1", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {results_path.name}: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    """Sort models by roc_auc (desc), tie-broken by f1, then accuracy."""
    return (
        df.sort_values(
            by=["roc_auc", "f1", "accuracy"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def resolve_model_file(model_name: str, source: str, model_dir: Path) -> Path:
    """Locate the correct .joblib file based on source type.

    For *safe* results, try <name>_safe.joblib first, then <name>.joblib.
    For *legacy* results, try <name>.joblib first, then <name>_safe.joblib.
    """
    if source == "safe":
        candidates = [
            model_dir / f"{model_name}_safe.joblib",
            model_dir / f"{model_name}.joblib",
        ]
    else:
        candidates = [
            model_dir / f"{model_name}.joblib",
            model_dir / f"{model_name}_safe.joblib",
        ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Model file not found for '{model_name}'. "
        f"Tried: {[str(c) for c in candidates]}.\n"
        "  Suggested fix: run  python ml/run_training.py  to generate safe models."
    )


def copy_model(src: Path, dst: Path) -> Path:
    """Copy a model file to a destination path."""
    shutil.copy2(src, dst)
    return dst


def save_model_info(row: pd.Series, source: str, output_path: Path) -> None:
    """Persist model metadata to JSON including source provenance."""
    info: dict[str, object] = {
        "model_name": str(row["model_name"]),
        "source": source,
        "roc_auc": float(row["roc_auc"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
    }
    # Include optional metrics when available
    for col in ("precision", "recall", "roc_auc_std", "cv_time_seconds"):
        if col in row.index and pd.notna(row[col]):
            info[col] = float(row[col])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def print_model_summary(label: str, row: pd.Series, source: str) -> None:
    """Print a compact metric summary for a selected model."""
    print(f"[INFO] {label} selected: {row['model_name']} (source: {source})")
    print(f"  roc_auc:  {float(row['roc_auc']):.6f}")
    print(f"  f1:       {float(row['f1']):.6f}")
    print(f"  accuracy: {float(row['accuracy']):.6f}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"

    # 1. Resolve and load results
    results_path, source = resolve_results_source(model_dir)
    df = load_and_validate(results_path)
    ranked = rank_models(df)

    # 2. Best model
    best_row = ranked.iloc[0]
    best_src = resolve_model_file(str(best_row["model_name"]), source, model_dir)
    best_dst = copy_model(best_src, model_dir / "best_model.joblib")
    save_model_info(best_row, source, model_dir / "best_model_info.json")

    print(f"\n[INFO] Copied best model to: {best_dst}")
    print(f"[INFO] Saved best model info to: {model_dir / 'best_model_info.json'}")
    print_model_summary("Best model", best_row, source)

    # 3. Backup model (second-best, if available)
    if len(ranked) >= 2:
        backup_row = ranked.iloc[1]
        backup_src = resolve_model_file(str(backup_row["model_name"]), source, model_dir)
        backup_dst = copy_model(backup_src, model_dir / "backup_model.joblib")
        save_model_info(backup_row, source, model_dir / "backup_model_info.json")

        print(f"\n[INFO] Copied backup model to: {backup_dst}")
        print(f"[INFO] Saved backup model info to: {model_dir / 'backup_model_info.json'}")
        print_model_summary("Backup model", backup_row, source)
    else:
        print("\n[WARN] Only one model available — no backup model selected.")

    print("\nSAFE BEST MODEL SELECTED")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
