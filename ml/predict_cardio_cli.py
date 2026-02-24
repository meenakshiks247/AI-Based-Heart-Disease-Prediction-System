"""Command-line cardio-disease prediction using the best saved model.

Usage
-----
    python ml/predict_cardio_cli.py                        # use built-in sample
    python ml/predict_cardio_cli.py '{"age":55, ...}'      # pass JSON string
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.load_cardio_model import get_cardio_model  # noqa: E402

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

EXAMPLE_SAMPLE: dict = {
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


def predict(sample: dict) -> dict:
    """Run a single cardio prediction and return result dict."""
    bundle = get_cardio_model()

    df = pd.DataFrame([sample])[FEATURE_COLUMNS]
    proba = bundle.model.predict_proba(df)

    probability = float(proba[0][1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": round(probability, 6),
        "model_name": bundle.model_name,
    }


if __name__ == "__main__":
    # Accept optional JSON from the command line; otherwise use the example
    if len(sys.argv) > 1:
        sample = json.loads(sys.argv[1])
    else:
        sample = EXAMPLE_SAMPLE

    result = predict(sample)
    print(f"prediction: {result['prediction']}, "
          f"probability: {result['probability']}, "
          f"model: {result['model_name']}")
