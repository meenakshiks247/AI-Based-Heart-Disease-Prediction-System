"""Compute global and local SHAP explanations for both best models.

Outputs
-------
JSON summaries:
    ml/models/explain_clinical_global.json
    ml/models/explain_clinical_local.json
    ml/models/explain_cardio_global.json
    ml/models/explain_cardio_local.json

PNG plots:
    ml/models/shap_clinical_summary.png
    ml/models/shap_cardio_summary.png

Usage
-----
    python ml/explain_shap.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402

# ── project root on sys.path ──────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ── Model configs ─────────────────────────────────────────────────────

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "tag": "clinical",
        "model_path": MODEL_DIR / "best_model.joblib",
        "info_path": MODEL_DIR / "best_model_info.json",
        "data_path": DATA_DIR / "train.csv",
        "max_sample": 500,
        "feature_columns": [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal",
        ],
    },
    {
        "tag": "cardio",
        "model_path": MODEL_DIR / "best_cardio_model.joblib",
        "info_path": MODEL_DIR / "best_cardio_model_info.json",
        "data_path": DATA_DIR / "cardio_cleaned.csv",
        "max_sample": 1000,
        "feature_columns": [
            "age", "sex", "height", "weight",
            "systolic_bp", "diastolic_bp", "cholesterol",
            "gluc", "smoke", "alco", "active",
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────

def _load_info(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _get_estimator(model: Any) -> Any:
    """Return the final estimator from a Pipeline, or the model itself."""
    if hasattr(model, "named_steps"):
        # Convention: last step is the estimator
        return model[len(model) - 1]
    return model


def _get_preprocessor(model: Any) -> Any | None:
    """Return the preprocessing step from a Pipeline, if present."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        return model.named_steps["preprocessor"]
    return None


def _is_tree_based(estimator: Any) -> bool:
    return hasattr(estimator, "feature_importances_")


def _is_linear(estimator: Any) -> bool:
    return hasattr(estimator, "coef_")


def _preprocess(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Run the pipeline preprocessor (if any) and return transformed array."""
    preprocessor = _get_preprocessor(model)
    if preprocessor is not None:
        return preprocessor.transform(X)
    return X.values


def _build_explainer(
    model: Any,
    estimator: Any,
    X_bg: pd.DataFrame,
):
    """Pick the fastest SHAP explainer for *estimator* and return it."""
    import shap

    X_bg_transformed = _preprocess(model, X_bg)

    if _is_tree_based(estimator):
        logger.info("  → Using TreeExplainer (fast)")
        return shap.TreeExplainer(estimator, data=X_bg_transformed)

    if _is_linear(estimator):
        logger.info("  → Using LinearExplainer")
        return shap.LinearExplainer(estimator, X_bg_transformed)

    # Fallback — slow but universal
    logger.info("  → Using KernelExplainer (slow — background ≤ 50 rows)")
    bg = shap.sample(X_bg_transformed, min(50, len(X_bg_transformed)))

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return estimator.predict_proba(x)

    return shap.KernelExplainer(predict_fn, bg)


# ── Core routines ─────────────────────────────────────────────────────

def compute_shap(cfg: dict) -> None:
    """Run SHAP for one model config entry and persist results."""
    import shap

    tag = cfg["tag"]
    logger.info("=" * 55)
    logger.info("SHAP explanation for: %s", tag)
    logger.info("=" * 55)

    # ── load model ────────────────────────────────────────────────────
    model_path: Path = cfg["model_path"]
    if not model_path.exists():
        _write_error(tag, f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    info = _load_info(cfg["info_path"])
    model_name = info.get("model_name", type(model).__name__)
    estimator = _get_estimator(model)
    logger.info("  Model name : %s", model_name)
    logger.info("  Estimator  : %s", type(estimator).__name__)

    # ── load data ─────────────────────────────────────────────────────
    data_path: Path = cfg["data_path"]
    if not data_path.exists():
        _write_error(tag, f"Data not found: {data_path}")
        return

    feature_cols = cfg["feature_columns"]
    df = pd.read_csv(data_path)
    df = df[feature_cols]

    max_sample = cfg["max_sample"]
    if len(df) > max_sample:
        df = df.sample(n=max_sample, random_state=42)

    logger.info("  Data rows  : %d", len(df))

    # ── build explainer ───────────────────────────────────────────────
    explainer = _build_explainer(model, estimator, df)

    # ── SHAP values ───────────────────────────────────────────────────
    X_transformed = _preprocess(model, df)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers shap_values may be a list of two arrays
    # (one per class). We want the positive-class (index 1) values.
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values[1])
    else:
        shap_vals = np.array(shap_values)

    # ── Global importance ─────────────────────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    global_importance = sorted(
        [
            {"feature": feature_cols[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in range(len(feature_cols))
        ],
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )

    global_payload = {
        "model_name": model_name,
        "feature_importance": global_importance,
    }
    global_path = MODEL_DIR / f"explain_{tag}_global.json"
    global_path.write_text(json.dumps(global_payload, indent=2), encoding="utf-8")
    logger.info("  Saved → %s", global_path.name)

    # ── Local explanations (10 rows) ──────────────────────────────────
    n_local = min(10, len(df))
    rows_out = []
    for idx in range(n_local):
        row_shap = shap_vals[idx]
        # Top 5 absolute contributors
        top_indices = np.argsort(np.abs(row_shap))[::-1][:5]
        top_features = []
        for j in top_indices:
            top_features.append({
                "feature": feature_cols[j],
                "shap": round(float(row_shap[j]), 6),
                "effect": "increase" if row_shap[j] > 0 else "decrease",
            })
        rows_out.append({
            "index": int(df.index[idx]),
            "top_features": top_features,
        })

    local_payload = {"model_name": model_name, "rows": rows_out}
    local_path = MODEL_DIR / f"explain_{tag}_local.json"
    local_path.write_text(json.dumps(local_payload, indent=2), encoding="utf-8")
    logger.info("  Saved → %s", local_path.name)

    # ── Summary plot ──────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        features=X_transformed,
        feature_names=feature_cols,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {tag.title()} ({model_name})", fontsize=13, pad=12)
    plt.tight_layout()
    plot_path = MODEL_DIR / f"shap_{tag}_summary.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved → %s", plot_path.name)


def _write_error(tag: str, msg: str) -> None:
    """Write a small error JSON when SHAP cannot be computed."""
    logger.error("  %s", msg)
    for suffix in ("global", "local"):
        err_path = MODEL_DIR / f"explain_{tag}_{suffix}.json"
        err_path.write_text(
            json.dumps({"error": msg}, indent=2), encoding="utf-8",
        )


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.error("shap is not installed. Run:  pip install shap")
        sys.exit(1)

    for cfg in MODEL_CONFIGS:
        try:
            compute_shap(cfg)
        except Exception as exc:
            logger.exception("SHAP failed for %s", cfg["tag"])
            _write_error(cfg["tag"], str(exc))

    logger.info("")
    logger.info("[DONE] SHAP explanations complete.")


if __name__ == "__main__":
    main()
