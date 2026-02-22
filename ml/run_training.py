from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path) -> None:
    """Run one pipeline script and stop on failure."""
    print(f"\n{'='*60}")
    print(f"[STEP] Running: {script_path.name}")
    print(f"{'='*60}", flush=True)
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_path} (exit code {result.returncode})")


def assert_artifact(path: Path, description: str) -> None:
    """Verify that an expected artifact file exists after a step."""
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Expected artifact missing after step: {path}\n"
            f"  Description: {description}\n"
            f"  Hint: re-run with --force or check the failing step above."
        )
    print(f"[CHECK] Verified artifact exists: {path}")


def read_best_model_name(best_info_path: Path) -> str:
    """Read best model name from best_model_info.json."""
    if not best_info_path.exists():
        return "UNKNOWN"

    try:
        with best_info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("model_name", "UNKNOWN"))
    except (OSError, json.JSONDecodeError):
        return "UNKNOWN"


# ── Pipeline definitions ────────────────────────────────────────────


def run_safe_pipeline(base_dir: Path) -> None:
    """Run the safe (leak-free) training pipeline."""
    model_dir = base_dir / "models"
    best_info_path = model_dir / "best_model_info.json"

    # Step (a): Fit preprocessor on training data only
    run_step(base_dir / "fit_pipeline_and_validate.py")
    assert_artifact(
        model_dir / "preprocessor.joblib",
        "Fitted preprocessor (output of fit_pipeline_and_validate.py)",
    )

    # Step (b): Train all models using only train.csv and the pipeline
    run_step(base_dir / "train_models_safe.py")
    assert_artifact(
        model_dir / "model_results_safe.csv",
        "Safe model results CSV (output of train_models_safe.py)",
    )

    # Step (c): Evaluate and plot
    run_step(base_dir / "evaluate_and_plot.py")

    # Step (d): Select best model
    run_step(base_dir / "compare_models.py")

    best_model_name = read_best_model_name(best_info_path)
    print(f"\n{'='*60}")
    print("[INFO] Safe training pipeline completed")
    print(f"[INFO] Best model: {best_model_name}")
    print(f"{'='*60}", flush=True)


def run_legacy_pipeline(base_dir: Path) -> None:
    """Run the original (legacy) training pipeline for backward compatibility."""
    best_info_path = base_dir / "models" / "best_model_info.json"

    print("[WARNING] Running legacy pipeline (may contain data leakage).")
    print("[WARNING] Prefer the default safe pipeline.\n", flush=True)

    run_step(base_dir / "train_models.py")
    run_step(base_dir / "evaluate_and_plot.py")
    run_step(base_dir / "compare_models.py")

    best_model_name = read_best_model_name(best_info_path)
    print("\nTRAINING FINISHED", flush=True)
    print(f"BEST MODEL: {best_model_name}", flush=True)


# ── Entry point ─────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the heart-disease model training pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if artifacts already exist (useful for CI).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the original (legacy) training pipeline instead of the safe one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    if args.force:
        print("[INFO] --force: all steps will run regardless of existing artifacts.\n")

    print("TRAINING STARTED", flush=True)

    if args.legacy:
        run_legacy_pipeline(base_dir)
    else:
        run_safe_pipeline(base_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
