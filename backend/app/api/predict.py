from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ml.load_model import load_best_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["predict"])

# ── Ordered feature columns (must match training data) ───────────────

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── Load artifacts once at import time ───────────────────────────────

_bundle = load_best_model()

# ── Pydantic schema ─────────────────────────────────────────────────


class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# ── Prediction endpoint ─────────────────────────────────────────────


@router.post("/")
def predict(data: HeartInput) -> dict:
    """Return heart-disease prediction for a single patient record."""
    try:
        # Build DataFrame with explicit column order
        df = pd.DataFrame([data.model_dump()])
        df = df.loc[:, FEATURE_COLUMNS]

        # Pipeline already embeds preprocessing — plain estimator needs
        # the standalone preprocessor applied first.
        if _bundle.is_pipeline:
            print("Pipeline model detected — predicting directly with dataframe")
            X = df
        else:
            print("Separate preprocessing used — transforming with preprocessor")
            if _bundle.preprocessor is None:
                raise RuntimeError(
                    "Preprocessor not loaded but model is not a Pipeline"
                )
            X = _bundle.preprocessor.transform(df)

        # Predict — training labels are inverted (class 0 = disease,
        # class 1 = healthy), so we read proba[:,0] as the disease
        # probability and expose prediction=1 for "high risk".
        prob = float(_bundle.model.predict_proba(X)[0][0])
        prediction = int(prob >= 0.5)

        return {
            "prediction": prediction,
            "probability": round(prob, 6),
            "model_name": _bundle.model_name,
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
