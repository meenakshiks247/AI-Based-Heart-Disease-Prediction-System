from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/models", tags=["models"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "ml" / "models"
_RESULTS_CSV = _MODELS_DIR / "model_results.csv"
_BEST_INFO_JSON = _MODELS_DIR / "best_model_info.json"


@router.get("/")
def get_models() -> JSONResponse:
    """Return model results summary from model_results.csv."""
    if not _RESULTS_CSV.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"model results not found: {_RESULTS_CSV}"},
        )

    models: list[dict[str, object]] = []
    with _RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(
                {
                    "model_name": row.get("model_name"),
                    "roc_auc": float(row["roc_auc"]) if row.get("roc_auc") else None,
                }
            )

    return JSONResponse(content={"models": models})


@router.get("/best")
def get_best_model() -> JSONResponse:
    """Return best model metadata from best_model_info.json."""
    if not _BEST_INFO_JSON.exists():
        return JSONResponse(status_code=404, content={"error": "best model not found"})

    with _BEST_INFO_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse(content=data)
