"""Load the best cardiovascular disease model with singleton caching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "models" / "best_cardio_model.joblib"
BEST_INFO_PATH = BASE_DIR / "models" / "best_cardio_model_info.json"

# ── Cached singleton ─────────────────────────────────────────────────
_cached_bundle: ModelBundleCardio | None = None


@dataclass
class ModelBundleCardio:
    model: Any
    is_pipeline: bool
    model_name: str
    info: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        kind = "Pipeline" if self.is_pipeline else "Raw"
        return f"ModelBundleCardio(name={self.model_name!r}, type={kind}, info_keys={list(self.info.keys())})"


def _load_model() -> Any:
    """Load model with joblib, falling back to cloudpickle."""
    try:
        return joblib.load(BEST_MODEL_PATH)
    except Exception as primary_err:
        try:
            import cloudpickle
            with open(BEST_MODEL_PATH, "rb") as f:
                return cloudpickle.load(f)
        except ImportError:
            raise primary_err
        except Exception:
            raise primary_err


def _load_info() -> dict:
    """Load companion JSON metadata if available."""
    if BEST_INFO_PATH.exists():
        return json.loads(BEST_INFO_PATH.read_text())
    return {}


def get_cardio_model() -> ModelBundleCardio:
    """Return a cached ModelBundleCardio singleton (loads once)."""
    global _cached_bundle
    if _cached_bundle is not None:
        return _cached_bundle

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")

    model = _load_model()
    info = _load_info()

    is_pipeline = hasattr(model, "named_steps") and hasattr(model, "predict_proba")
    model_name = info.get("model_name", type(model).__name__)

    kind = "Pipeline" if is_pipeline else type(model).__name__
    print(f"[INFO] Loaded cardio model: {model_name} ({kind})")

    _cached_bundle = ModelBundleCardio(
        model=model,
        is_pipeline=is_pipeline,
        model_name=model_name,
        info=info,
    )
    return _cached_bundle


if __name__ == "__main__":
    bundle = get_cardio_model()
    print(bundle)
    print(f"\nModel type  : {type(bundle.model).__name__}")
    print(f"Is pipeline : {bundle.is_pipeline}")
    print(f"Model name  : {bundle.model_name}")
    if bundle.info:
        print(f"ROC AUC     : {bundle.info.get('roc_auc', 'N/A')}")
        print(f"Accuracy    : {bundle.info.get('accuracy', 'N/A')}")
