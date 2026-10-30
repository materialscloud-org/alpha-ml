"""FastAPI app: POST /api/predict, and the built frontend at / when present.

Run:  ALPHAML_DATA_DIR=<tensors folder> uvicorn alphaml_api.main:app
(the settings: config.py)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import config
from .schemas import PredictRequest, PredictResponse
from .service import InputError, compute



def load_model_data():
    """Load the model arrays once, only when the data directory is configured."""
    if not config.DATA_DIR:
        return None
    from .core import ModelData

    return ModelData(config.DATA_DIR)


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
if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
