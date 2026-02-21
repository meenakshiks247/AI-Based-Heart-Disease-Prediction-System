from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Cache loaded objects so repeated calls do not reload from disk.
_cached_model: Any | None = None
_cached_preprocessor: Any | None = None
_is_loaded = False

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_DIR = _PROJECT_ROOT / "ml" / "models"
_BEST_MODEL_PATH = _MODEL_DIR / "best_model.joblib"
_PREPROCESSOR_PATH = _MODEL_DIR / "preprocessor.joblib"


def load_best_model() -> tuple[Any, Any | None]:
    """
    Load and cache the best trained model and optional preprocessor.

    Returns:
        (model, preprocessor)
        preprocessor is None when preprocessor.joblib is not available.
    """
    global _cached_model, _cached_preprocessor, _is_loaded

    if _is_loaded and _cached_model is not None:
        return _cached_model, _cached_preprocessor

    if not _BEST_MODEL_PATH.exists():
        msg = f"Best model file not found: {_BEST_MODEL_PATH}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    _cached_model = joblib.load(_BEST_MODEL_PATH)
    logger.info("Loaded best model from %s", _BEST_MODEL_PATH)

    if _PREPROCESSOR_PATH.exists():
        _cached_preprocessor = joblib.load(_PREPROCESSOR_PATH)
        logger.info("Loaded preprocessor from %s", _PREPROCESSOR_PATH)
    else:
        _cached_preprocessor = None
        logger.warning("Preprocessor file not found: %s. Returning None.", _PREPROCESSOR_PATH)

    _is_loaded = True
    return _cached_model, _cached_preprocessor
