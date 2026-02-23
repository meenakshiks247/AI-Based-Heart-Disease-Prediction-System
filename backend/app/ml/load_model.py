from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _PROJECT_ROOT / "ml" / "models"
_BEST_MODEL_PATH = _MODEL_DIR / "best_model.joblib"
_PREPROCESSOR_PATH = _MODEL_DIR / "preprocessor.joblib"
_BEST_INFO_PATH = _MODEL_DIR / "best_model_info.json"


# ── Return type ──────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """Everything needed for inference, loaded once and cached."""

    model: Any = None
    preprocessor: Any | None = None
    is_pipeline: bool = False
    model_name: str = "unknown"


# ── Cached singleton ─────────────────────────────────────────────────

_bundle: ModelBundle | None = None


def _try_load(path: Path) -> Any:
    """Load a joblib file, falling back to cloudpickle if available."""
    try:
        return joblib.load(path)
    except Exception as primary_err:
        logger.warning("joblib.load failed for %s: %s", path, primary_err)
        try:
            import cloudpickle  # optional dependency
            import pickle

            with path.open("rb") as fh:
                obj = pickle.load(fh)
            logger.info("Loaded %s via cloudpickle fallback", path)
            return obj
        except ImportError:
            logger.error("cloudpickle not installed — cannot retry load")
            raise primary_err
        except Exception:
            raise primary_err


def load_best_model() -> ModelBundle:
    """
    Load and cache the best trained model, optional preprocessor, and
    model metadata.

    - If the model is a sklearn Pipeline the preprocessor step is
      already embedded → ``preprocessor`` stays ``None`` and
      ``is_pipeline`` is ``True``.
    - If the model is a plain estimator, ``preprocessor.joblib`` is
      loaded separately so the caller can call
      ``preprocessor.transform()`` before prediction.

    Returns:
        A ``ModelBundle`` with all fields populated.

    Raises:
        FileNotFoundError: if ``best_model.joblib`` does not exist.
    """
    global _bundle

    if _bundle is not None:
        return _bundle

    # ── model ────────────────────────────────────────────────────────
    if not _BEST_MODEL_PATH.exists():
        msg = f"Best model file not found: {_BEST_MODEL_PATH}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    bundle = ModelBundle()
    bundle.model = _try_load(_BEST_MODEL_PATH)
    bundle.is_pipeline = isinstance(bundle.model, Pipeline)
    logger.info(
        "Loaded model from %s (type=%s, is_pipeline=%s)",
        _BEST_MODEL_PATH,
        type(bundle.model).__name__,
        bundle.is_pipeline,
    )

    # ── preprocessor (only needed for plain estimators) ───────────────
    if bundle.is_pipeline:
        print("Pipeline model detected — preprocessor is embedded, skipping preprocessor.joblib")
    else:
        if _PREPROCESSOR_PATH.exists():
            bundle.preprocessor = _try_load(_PREPROCESSOR_PATH)
            print("Separate preprocessing used — loaded preprocessor.joblib")
            logger.info("Loaded standalone preprocessor from %s", _PREPROCESSOR_PATH)
        else:
            logger.warning("No preprocessor.joblib found and model is not a Pipeline")
            print("WARNING: No preprocessor found and model is not a Pipeline")

    # ── metadata ─────────────────────────────────────────────────────
    if _BEST_INFO_PATH.exists():
        with _BEST_INFO_PATH.open("r", encoding="utf-8") as f:
            info = json.load(f)
        bundle.model_name = info.get("model_name", "unknown")
        logger.info("Model name: %s", bundle.model_name)

    _bundle = bundle
    return _bundle
