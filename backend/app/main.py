from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.models_api import router as models_router
from app.api.predict import router as predict_router
from app.api.cardio_predict import router as cardio_predict_router


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

app.include_router(models_router)
app.include_router(predict_router)
app.include_router(cardio_predict_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Heart Disease Prediction API is running."}
