from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.models_api import router as models_router
from app.api.predict import router as predict_router
from app.api.cardio_predict import router as cardio_predict_router
from app.api.explain import router as explain_router


app = FastAPI(
    title="Heart Disease Prediction API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for SHAP plots & model artefacts ──────────────
_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "models"
if _MODELS_DIR.is_dir():
    app.mount("/static/models", StaticFiles(directory=str(_MODELS_DIR)), name="model-assets")

app.include_router(models_router)
app.include_router(predict_router)
app.include_router(cardio_predict_router)
app.include_router(explain_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Heart Disease Prediction API is running."}
