"""Explainability endpoint — SHAP + rule-based recommendations.

Router prefix: ``/api/explain``

POST ``/api/explain/?dataset=clinical``  (or ``cardio``)

Returns prediction, probability, risk level, top SHAP features, and
human-readable recommendations for a single patient row.
"""

from __future__ import annotations

import logging
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# ── Make project root importable ──────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.ml.load_model import load_best_model          # noqa: E402
from ml.load_cardio_model import get_cardio_model       # noqa: E402
from ml.risk_recommender import recommend_from_prediction  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/explain", tags=["explain"])

# ── Feature columns (same order as training) ──────────────────────────

CLINICAL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

CARDIO_FEATURES = [
    "age", "sex", "height", "weight",
    "systolic_bp", "diastolic_bp", "cholesterol",
    "gluc", "smoke", "alco", "active",
]

# ── Pydantic schemas ─────────────────────────────────────────────────


class ExplainInput(BaseModel):
    """Superset of clinical + cardio fields.

    Clinical uses 13 fields; cardio uses 11.  Unused fields for the
    chosen dataset are simply ignored.
    """

    # Clinical (Cleveland) fields
    age: float
    sex: float
    cp: float | None = Field(None, description="Chest-pain type (clinical)")
    trestbps: float | None = Field(None, description="Resting BP (clinical)")
    chol: float | None = Field(None, description="Cholesterol (clinical)")
    fbs: float | None = Field(None, description="Fasting blood sugar >120 (clinical)")
    restecg: float | None = Field(None, description="Resting ECG (clinical)")
    thalach: float | None = Field(None, description="Max heart rate (clinical)")
    exang: float | None = Field(None, description="Exercise-induced angina (clinical)")
    oldpeak: float | None = Field(None, description="ST depression (clinical)")
    slope: float | None = Field(None, description="Slope of peak ST segment (clinical)")
    ca: float | None = Field(None, description="Major vessels coloured by fluoroscopy (clinical)")
    thal: float | None = Field(None, description="Thalassemia (clinical)")

    # Cardio fields (overlap: age, sex)
    height: float | None = Field(None, description="Height in cm (cardio)")
    weight: float | None = Field(None, description="Weight in kg (cardio)")
    systolic_bp: float | None = Field(None, description="Systolic BP (cardio)")
    diastolic_bp: float | None = Field(None, description="Diastolic BP (cardio)")
    cholesterol: int | None = Field(None, description="Cholesterol level 1/2/3 (cardio)")
    gluc: int | None = Field(None, description="Glucose level 1/2/3 (cardio)")
    smoke: int | None = Field(None, description="Smoking 0/1 (cardio)")
    alco: int | None = Field(None, description="Alcohol 0/1 (cardio)")
    active: int | None = Field(None, description="Physical activity 0/1 (cardio)")


class TopFeature(BaseModel):
    feature: str
    shap: float
    effect: str  # "increase" | "decrease"


class ExplainResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    top_features: list[TopFeature]
    recommendations: list[str]
    model_name: str


# ── SHAP helpers ──────────────────────────────────────────────────────

def _get_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        return model[len(model) - 1]
    return model


def _get_preprocessor(model: Any) -> Any | None:
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        return model.named_steps["preprocessor"]
    return None


def _preprocess(model: Any, X: pd.DataFrame) -> np.ndarray:
    preprocessor = _get_preprocessor(model)
    if preprocessor is not None:
        return preprocessor.transform(X)
    return X.values


def _compute_shap_for_row(
    model: Any,
    estimator: Any,
    row_df: pd.DataFrame,
    bg_df: pd.DataFrame,
) -> np.ndarray:
    """Return 1-D array of SHAP values for a single row."""
    import shap

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    X_bg = _preprocess(model, bg_df)
    X_row = _preprocess(model, row_df)

    # Pick the right explainer
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator, data=X_bg)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, X_bg)
    else:
        bg_sample = shap.sample(X_bg, min(50, len(X_bg)))
        explainer = shap.KernelExplainer(
            lambda x: estimator.predict_proba(x), bg_sample,
        )

    sv = explainer.shap_values(X_row)

    # Binary classifier may return list of two arrays
    if isinstance(sv, list):
        vals = np.array(sv[1]).flatten()
    else:
        vals = np.array(sv).flatten()

    return vals


# ── Small background samples (loaded lazily, once) ────────────────────

_BG_CACHE: dict[str, pd.DataFrame] = {}
_BG_MAX_ROWS = 100  # keep latency low


def _load_background(dataset: str) -> pd.DataFrame:
    if dataset in _BG_CACHE:
        return _BG_CACHE[dataset]

    base = Path(_PROJECT_ROOT) / "ml" / "data"
    if dataset == "clinical":
        path = base / "train.csv"
        cols = CLINICAL_FEATURES
    else:
        path = base / "cardio_cleaned.csv"
        cols = CARDIO_FEATURES

    df = pd.read_csv(path)[cols]
    if len(df) > _BG_MAX_ROWS:
        df = df.sample(n=_BG_MAX_ROWS, random_state=42)
    _BG_CACHE[dataset] = df
    return df


# ── Endpoint ──────────────────────────────────────────────────────────


@router.post("/", response_model=ExplainResponse)
def explain(
    data: ExplainInput,
    dataset: Literal["clinical", "cardio"] = Query(
        "clinical",
        description="Which model to explain: 'clinical' (Cleveland) or 'cardio' (Cardiovascular)",
    ),
) -> dict:
    """Predict, explain with SHAP, and return rule-based recommendations."""
    try:
        payload = data.model_dump()

        # ── Pick model & features ────────────────────────────────────
        if dataset == "clinical":
            bundle = load_best_model()
            feature_cols = CLINICAL_FEATURES
        else:
            bundle = get_cardio_model()
            feature_cols = CARDIO_FEATURES

        # ── Build single-row DataFrame ───────────────────────────────
        row_data = {col: payload[col] for col in feature_cols}
        df = pd.DataFrame([row_data])
        df = df[feature_cols]

        # ── Prediction ───────────────────────────────────────────────
        if bundle.is_pipeline:
            proba = bundle.model.predict_proba(df)
        else:
            if bundle.preprocessor is not None:
                X_pred = bundle.preprocessor.transform(df)
            else:
                X_pred = df
            proba = bundle.model.predict_proba(X_pred)

        # Clinical: class 0 = disease (inverted labels)
        # Cardio : class 1 = disease
        if dataset == "clinical":
            probability = float(proba[0][0])
        else:
            probability = float(proba[0][1])
        prediction = int(probability >= 0.5)
        probability = round(probability, 6)

        # risk level
        if probability < 0.40:
            risk_level = "Low Risk"
        elif probability < 0.70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        # ── SHAP explanation ─────────────────────────────────────────
        estimator = _get_estimator(bundle.model)
        bg = _load_background(dataset)
        shap_vals = _compute_shap_for_row(bundle.model, estimator, df, bg)

        # Top 5 features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_features = []
        for i in top_indices:
            top_features.append({
                "feature": feature_cols[i],
                "shap": round(float(shap_vals[i]), 6),
                "effect": "increase" if shap_vals[i] > 0 else "decrease",
            })

        # ── Recommendations ──────────────────────────────────────────
        top_for_rec = [(f["feature"], f["shap"]) for f in top_features]
        rec_result = recommend_from_prediction(payload, top_for_rec, probability)

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "top_features": top_features,
            "recommendations": rec_result["suggestions"],
            "model_name": bundle.model_name,
        }

    except Exception as exc:
        logger.exception("Explain endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
