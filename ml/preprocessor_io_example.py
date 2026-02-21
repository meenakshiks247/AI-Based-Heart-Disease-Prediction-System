"""
preprocessor_io_example.py
--------------------------
Short helper showing how to SAVE and LOAD the preprocessing pipeline
used during training so the backend applies identical transformations
at inference time.

NOTE: The trained models in ml/models/*.joblib already embed the
preprocessor inside a sklearn Pipeline, so you normally don't need a
separate file.  This script is useful when you want to ship the
preprocessor independently (e.g. for a lightweight API that only
needs to transform raw input before calling an external model service).
"""

from pathlib import Path

import joblib
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
MODEL_DIR = Path("ml/models")
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# =====================================================================
# 1.  SAVE  (run once, right after training)
# =====================================================================
# Assuming `preprocessor` is the fitted ColumnTransformer from
# train_models.build_preprocessor(), already .fit() on the training set:
#
#   from ml.train_models import build_preprocessor
#   preprocessor = build_preprocessor(X_train)
#   preprocessor.fit(X_train)
#
#   joblib.dump(preprocessor, PREPROCESSOR_PATH)
#   print(f"[INFO] Preprocessor saved → {PREPROCESSOR_PATH}")

# =====================================================================
# 2.  LOAD  (at inference / in the backend)
# =====================================================================
# pre = joblib.load(PREPROCESSOR_PATH)
# X_new_transformed = pre.transform(X_new)   # X_new is a DataFrame

# =====================================================================
# 3.  Quick runnable demo  (extracts the preprocessor from an existing
#     saved pipeline and round-trips it through joblib)
# =====================================================================
if __name__ == "__main__":
    # Pick any saved pipeline that contains the preprocessor step
    sample_pipeline_path = MODEL_DIR / "logistic_regression.joblib"
    pipeline = joblib.load(sample_pipeline_path)

    # Extract the fitted preprocessor from the pipeline
    preprocessor = pipeline.named_steps["preprocessor"]

    # Save it standalone
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[SAVE] Preprocessor saved → {PREPROCESSOR_PATH}")

    # Reload and verify with a single sample row
    pre = joblib.load(PREPROCESSOR_PATH)
    sample = pd.DataFrame([{
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
    }])
    transformed = pre.transform(sample)
    print(f"[LOAD] Preprocessor reloaded — transformed shape: {transformed.shape}")
    print(f"[LOAD] Transformed values:\n{transformed}\n")

    # ⚠️  IMPORTANT
    # The backend / inference service MUST use this exact preprocessor
    # so that feature ordering and scaling match what the model saw
    # during training.  A mismatch will silently produce wrong predictions.
