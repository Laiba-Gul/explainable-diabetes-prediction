"""FastAPI app: /health, /model/info and /predict (with optional SHAP explanation).

Privacy: request bodies, feature values and predictions are never logged; only
request id, path, status and latency.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from diabetes_xai import __version__
from diabetes_xai.api.schemas import HealthResponse, ModelInfoResponse, PatientIndicators, PredictionResponse
from diabetes_xai.inference import ModelBundle

logger = logging.getLogger("diabetes_xai.api")


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        for key in ("request_id", "path", "method", "status_code", "latency_ms", "model_version"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry)


def _configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    logging.getLogger("uvicorn.access").disabled = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    model_dir = os.environ.get("MODEL_DIR", "artifacts/brfss_serve")
    try:
        app.state.bundle = ModelBundle.load(model_dir)
        logger.info("model_loaded", extra={"model_version": app.state.bundle.version})
    except FileNotFoundError as exc:
        app.state.bundle = None
        logger.error(f"model_load_failed: {exc}")
    yield


app = FastAPI(
    title="Explainable Diabetes Risk API",
    version=__version__,
    description=(
        "Estimates diabetes/prediabetes risk from the 21 BRFSS 2015 health indicators and "
        "optionally explains the prediction with SHAP. Research/education use only - "
        "not a medical device."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    detail = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def _bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model is not loaded")
    return bundle


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_loaded": False})
    return HealthResponse(status="ok", model_loaded=True, model_version=bundle.version)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request):
    b = _bundle(request)
    m = b.metadata
    return ModelInfoResponse(
        model_version=b.version,
        model_name=m.get("model_name", ""),
        dataset=m.get("dataset", ""),
        features=b.features,
        threshold=b.threshold,
        threshold_policy=m.get("threshold_policy", ""),
        test_metrics=m.get("test_metrics_at_serving_threshold", {}),
        trained_at=m.get("trained_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(
    payload: PatientIndicators,
    request: Request,
    explain: bool = Query(False, description="Include top SHAP factors"),
):
    b = _bundle(request)
    row = payload.model_dump()
    result = b.predict(row)[0]
    if explain:
        result["top_factors"] = b.explain(row, top_k=5)[0]
    return PredictionResponse(**result)
