"""Cardio-disease prediction endpoint.

Router prefix: ``/api/predict/cardio``

After including this router in the FastAPI app the endpoint is
reachable at **POST /api/predict/cardio/**.

Quick smoke-test (server must be running)::

    curl -X POST http://127.0.0.1:8000/api/predict/cardio/ \\
      -H "Content-Type: application/json" \\
      -d '{"age":50.4,"sex":2,"height":168,"weight":62,
           "systolic_bp":110,"diastolic_bp":80,
           "cholesterol":1,"gluc":1,
           "smoke":0,"alco":0,"active":1}'
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make the project root importable so ``ml.load_cardio_model`` resolves
# when this module is loaded inside the backend package.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.load_cardio_model import get_cardio_model  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict/cardio", tags=["predict-cardio"])

# ── Feature columns — must match cardio_cleaned.csv (minus "target") ──

FEATURE_COLUMNS = [
    "age",
    "sex",
    "height",
    "weight",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

# ── Pydantic input schema ────────────────────────────────────────────


class CardioInput(BaseModel):
    """Patient record for cardio-disease prediction.

    Field descriptions follow the Kaggle *Cardiovascular Disease* dataset
    after the renaming applied in ``prepare_cardio_dataset.py``.
    """

    age: float = Field(..., description="Age in years (e.g. 50.4)")
    sex: int = Field(..., description="Sex — 1 = female, 2 = male (dataset encoding)")
    height: float = Field(..., description="Height in cm")
    weight: float = Field(..., description="Weight in kg")
    systolic_bp: float = Field(..., description="Systolic blood pressure (ap_hi)")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure (ap_lo)")
    cholesterol: int = Field(..., description="Cholesterol level (1: normal, 2: above normal, 3: well above)")
    gluc: int = Field(..., description="Glucose level (1: normal, 2: above normal, 3: well above)")
    smoke: int = Field(..., description="Smoking (0/1)")
    alco: int = Field(..., description="Alcohol intake (0/1)")
    active: int = Field(..., description="Physical activity (0/1)")

    model_config = {"json_schema_extra": {"examples": [
        {
            "age": 50.4,
            "sex": 2,
            "height": 168,
            "weight": 62.0,
            "systolic_bp": 110,
            "diastolic_bp": 80,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1,
        }
    ]}}


# ── Health-factor helpers ────────────────────────────────────────────


def _cardio_factors(
    raw: dict, risk_level: str,
) -> tuple[list[str], list[str]]:
    """Return (positive_factors, risk_factors) for the cardio model."""
    positives: list[str] = []
    risks: list[str] = []

    # Blood pressure
    sbp = raw.get("systolic_bp", 999)
    if sbp < 120:
        positives.append("Healthy Blood Pressure")
    elif sbp >= 140:
        risks.append("Elevated Blood Pressure")

    # Cholesterol level (1 = normal)
    chol = raw.get("cholesterol", 1)
    if chol == 1:
        positives.append("Good Cholesterol")
    elif chol >= 3:
        risks.append("High Cholesterol")

    # Physical activity
    if raw.get("active", 0) == 1:
        positives.append("Active Lifestyle")
    else:
        risks.append("Physical Inactivity")

    # Age
    age = raw.get("age", 999)
    if age < 50:
        positives.append("Healthy Age Range")
    elif age >= 60:
        risks.append("Age-Related Risk")

    # BMI from height & weight
    h = raw.get("height", 0)
    w = raw.get("weight", 0)
    if h > 0 and w > 0:
        bmi = w / ((h / 100) ** 2)
        if bmi < 25:
            positives.append("Healthy Weight")
        elif bmi >= 30:
            risks.append("Obesity Risk")

    # Smoking
    if raw.get("smoke", 0) == 0:
        positives.append("Non-Smoker")
    else:
        risks.append("Smoking Habit")

    # Glucose
    gluc = raw.get("gluc", 1)
    if gluc == 1:
        positives.append("Normal Glucose")
    elif gluc >= 3:
        risks.append("Elevated Glucose")

    # Alcohol
    if raw.get("alco", 0) == 0:
        positives.append("No Alcohol Use")
    else:
        risks.append("Alcohol Consumption")

    return positives, risks


def _cardio_recommendations(risk_level: str) -> list[str]:
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
    return [
        "Consult a healthcare professional.",
        "Schedule cardiovascular screening.",
        "Adopt immediate lifestyle modifications.",
    ]


# ── Prediction handler ──────────────────────────────────────────────


@router.post("/")
def predict_cardio(data: CardioInput) -> dict:
    """Return cardio-disease prediction for a single patient record.

    Returns
    -------
    dict
        ``prediction`` (0 = healthy, 1 = disease),
        ``probability`` (float 0-1),
        ``model_name``.
    """
    try:
        bundle = get_cardio_model()

        # Build a single-row DataFrame with the correct column order
        df = pd.DataFrame([data.model_dump()])
        df = df[FEATURE_COLUMNS]

        # The cardio models are saved as full sklearn Pipelines
        # (preprocessor + estimator), so we can predict directly.
        if bundle.is_pipeline:
            proba = bundle.model.predict_proba(df)
        else:
            # Fallback: if a plain estimator was saved without a
            # preprocessor wrapper, pass the raw DataFrame.
            proba = bundle.model.predict_proba(df)

        # target encoding: 1 = disease → proba[:,1] is disease probability
        probability = float(proba[0][1])
        prediction = int(probability >= 0.5)
        probability = round(probability, 6)

        if probability < 0.40:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        raw = data.model_dump()
        positive_factors, risk_factors = _cardio_factors(raw, risk_level)
        recommendations = _cardio_recommendations(risk_level)

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "positive_factors": positive_factors,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "model_name": bundle.model_name,
        }

    except Exception as exc:
        logger.exception("Cardio prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
