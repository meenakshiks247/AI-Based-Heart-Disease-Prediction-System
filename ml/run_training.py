from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path) -> None:
    """Run one pipeline script and stop on failure."""
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_path} (exit code {result.returncode})")


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


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    train_script = base_dir / "train_models.py"
    eval_script = base_dir / "evaluate_and_plot.py"
    compare_script = base_dir / "compare_models.py"
    best_info_path = base_dir / "models" / "best_model_info.json"

    print("TRAINING STARTED", flush=True)
    run_step(train_script)
    run_step(eval_script)
    run_step(compare_script)
    print("TRAINING FINISHED", flush=True)

    best_model_name = read_best_model_name(best_info_path)
    print(f"BEST MODEL: {best_model_name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
