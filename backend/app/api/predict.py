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


# ── Health-factor helpers ────────────────────────────────────────────


def _clinical_factors(
    raw: dict, risk_level: str,
) -> tuple[list[str], list[str]]:
    """Return (positive_factors, risk_factors) for the clinical model."""
    positives: list[str] = []
    risks: list[str] = []

    # Blood pressure
    bp = raw.get("trestbps", 999)
    if bp < 120:
        positives.append("Healthy Blood Pressure")
    elif bp >= 140:
        risks.append("Elevated Blood Pressure")

    # Cholesterol
    chol = raw.get("chol", 999)
    if chol < 200:
        positives.append("Good Cholesterol")
    elif chol >= 240:
        risks.append("High Cholesterol")

    # Age
    age = raw.get("age", 999)
    if age < 50:
        positives.append("Healthy Age Range")
    elif age >= 60:
        risks.append("Age-Related Risk")

    # Max heart rate (higher is generally better)
    thalach = raw.get("thalach", 0)
    if thalach >= 150:
        positives.append("Good Cardiovascular Fitness")
    elif thalach < 120:
        risks.append("Low Peak Heart Rate")

    # Fasting blood sugar
    if raw.get("fbs", 0) == 0:
        positives.append("Normal Blood Sugar")
    else:
        risks.append("Elevated Blood Sugar")

    # Exercise-induced angina
    if raw.get("exang", 0) == 0:
        positives.append("No Exercise Angina")
    else:
        risks.append("Exercise-Induced Angina")

    return positives, risks


def _recommendations(risk_level: str) -> list[str]:
    if risk_level == "Low Risk":
        return [
            "Continue daily walking.",
            "Maintain a healthy diet.",
            "Regular health check every year.",
        ]
    if risk_level == "Moderate Risk":
        return [
            "Increase physical activity.",
            "Monitor blood pressure monthly.",
            "Reduce salt and sugar intake.",
        ]
    # High Risk
    return [
        "Consult a healthcare professional.",
        "Schedule cardiovascular screening.",
        "Adopt immediate lifestyle modifications.",
    ]


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

        probability = round(prob, 6)

        if probability < 0.40:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # ── health factors & recommendations ─────────────────────
        raw = data.model_dump()
        positive_factors, risk_factors = _clinical_factors(raw, risk_level)
        recommendations = _recommendations(risk_level)

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "positive_factors": positive_factors,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "model_name": _bundle.model_name,
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
