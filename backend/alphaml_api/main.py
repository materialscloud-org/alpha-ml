"""FastAPI app: POST /api/predict, and the built frontend at / when present.

Run:  ALPHAML_DATA_DIR=<model data> uvicorn alphaml_api.main:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .schemas import PredictRequest, PredictResponse
from .service import InputError, compute

FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", Path(__file__).parents[2] / "frontend" / "dist"))


def load_model_data():
    """Load the model arrays once, only when the data directory is configured."""
    data_dir = os.environ.get("ALPHAML_DATA_DIR")
    if not data_dir:
        return None
    from .core import ModelData

    return ModelData(data_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_data()
    yield
    app.state.model = None


app = FastAPI(title="alpha-ml", version="2.0.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model data not loaded: set ALPHAML_DATA_DIR",
        )
    try:
        return compute(request.symbols, request.positions, request.model, app.state.model)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Last, so that /api routes take precedence.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
