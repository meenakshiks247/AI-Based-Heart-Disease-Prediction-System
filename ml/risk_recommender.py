"""Rule-based risk recommendations derived from patient features and SHAP contributions.

**Not medical advice** — every suggestion includes language such as
"consult a doctor" / "consider".

Usage
-----
    python ml/risk_recommender.py                       # built-in demo
    python -c "from ml.risk_recommender import recommend_from_prediction; ..."
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ── Rule table ────────────────────────────────────────────────────────
# Each rule is (condition_fn, severity, message).
# severity: 1 = urgent, 2 = important, 3 = advisory.

_RULES: list[tuple[Any, int, str]] = [
    # Chest-pain / angina
    (
        lambda r: r.get("exang") == 1,
        1,
        "Exercise-induced angina reported — immediate clinical evaluation recommended.",
    ),
    # Blood pressure (Cleveland or Cardio feature names)
    (
        lambda r: r.get("trestbps", 0) >= 140 or r.get("systolic_bp", 0) >= 140,
        1,
        "Elevated systolic blood pressure (≥140 mmHg). Monitor BP regularly and seek medical review.",
    ),
    # Cholesterol
    (
        lambda r: r.get("chol", 0) >= 240,
        2,
        "High cholesterol (≥240 mg/dL). Consider a lipid profile and dietary changes; consult a physician.",
    ),
    # Smoking
    (
        lambda r: r.get("smoke") == 1,
        2,
        "Smoking increases cardiovascular risk significantly. Seek smoking cessation support.",
    ),
    # Alcohol
    (
        lambda r: r.get("alco") == 1,
        3,
        "Regular alcohol intake noted. Discuss safe consumption limits with your doctor.",
    ),
    # BMI from height/weight (Cardio dataset)
    (
        lambda r: _bmi(r) is not None and _bmi(r) >= 30,
        2,
        "BMI indicates obesity (≥30). Weight management through diet and exercise is advised; consult a healthcare provider.",
    ),
    # Age
    (
        lambda r: r.get("age", 0) >= 60,
        3,
        "Age ≥ 60 increases cardiovascular risk; consider a routine cardiology checkup.",
    ),
    # Low physical activity
    (
        lambda r: r.get("active") == 0,
        3,
        "Physical inactivity reported. Regular moderate exercise can reduce heart disease risk — consult your doctor before starting.",
    ),
    # Fasting blood sugar
    (
        lambda r: r.get("fbs") == 1,
        2,
        "Fasting blood sugar > 120 mg/dL. Screen for diabetes and discuss management options with a physician.",
    ),
    # Low max heart rate (possible reduced fitness)
    (
        lambda r: r.get("thalach", 999) < 120,
        3,
        "Maximum heart rate below 120 bpm may suggest reduced cardiovascular fitness; discuss with your doctor.",
    ),
    # High glucose (Cardio feature)
    (
        lambda r: r.get("gluc", 1) >= 3,
        2,
        "Glucose level well above normal. Consider a comprehensive metabolic panel; consult a physician.",
    ),
]


def _bmi(r: dict) -> float | None:
    """Compute BMI from height (cm) and weight (kg) if both are present."""
    h = r.get("height")
    w = r.get("weight")
    if h and w and h > 0:
        return w / ((h / 100) ** 2)
    return None


# ── Core functions ────────────────────────────────────────────────────


def recommend_for_row(
    row: pd.Series | dict,
    top_features: list[tuple[str, float]] | None = None,
    *,
    shap_threshold: float = 0.05,
    max_suggestions: int = 5,
) -> list[str]:
    """Return up to *max_suggestions* human-readable recommendations.

    Parameters
    ----------
    row:
        Patient feature values (Series or dict).
    top_features:
        ``[(feature_name, shap_value), ...]`` — SHAP contributions for
        this patient.  Positive values mean the feature pushed the
        prediction toward *higher* risk.
    shap_threshold:
        Minimum absolute SHAP value to flag a feature as a key
        contributor.
    max_suggestions:
        Cap on the number of suggestions returned.
    """
    r = dict(row) if isinstance(row, pd.Series) else dict(row)

    # Collect (severity, message)
    hits: list[tuple[int, str]] = []

    # 1) Rule-based suggestions
    for condition, severity, message in _RULES:
        try:
            if condition(r):
                hits.append((severity, message))
        except Exception:
            pass  # skip rules that cannot be evaluated

    # 2) SHAP-based key-contributor notes
    if top_features:
        for feat, shap_val in top_features:
            if abs(shap_val) >= shap_threshold:
                direction = "increased risk" if shap_val > 0 else "decreased risk"
                hits.append(
                    (3, f"Key contributor: {feat} ({direction}). Discuss this factor with your doctor.")
                )

    # De-duplicate identical messages, sort by severity (1 = most urgent)
    seen: set[str] = set()
    unique: list[tuple[int, str]] = []
    for sev, msg in hits:
        if msg not in seen:
            seen.add(msg)
            unique.append((sev, msg))

    unique.sort(key=lambda t: t[0])
    return [msg for _, msg in unique[:max_suggestions]]


def recommend_from_prediction(
    payload: dict,
    top_features: list[tuple[str, float]] | None = None,
    probability: float = 0.0,
) -> dict:
    """High-level wrapper returning a JSON-friendly result.

    Returns
    -------
    dict
        ``suggestions``: list of recommendation strings.
        ``summary``: one-line risk summary.
    """
    suggestions = recommend_for_row(payload, top_features)

    if probability >= 0.70:
        summary = "High predicted risk — please consult a cardiologist promptly."
    elif probability >= 0.40:
        summary = "Moderate predicted risk — consider preventive screening."
    else:
        summary = "Low predicted risk — maintain healthy lifestyle habits."

    return {
        "summary": summary,
        "suggestions": suggestions,
    }


# ── CLI demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample = {
        "age": 65, "sex": 1, "cp": 2, "trestbps": 150, "chol": 260,
        "fbs": 1, "restecg": 0, "thalach": 110, "exang": 1,
        "oldpeak": 2.3, "slope": 2, "ca": 1, "thal": 2,
    }
    top = [("chol", 0.12), ("trestbps", 0.08), ("exang", 0.15)]

    result = recommend_from_prediction(sample, top, probability=0.78)
    print(json.dumps(result, indent=2))
