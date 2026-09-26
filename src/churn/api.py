"""Real-time scoring API (FastAPI).

* ``POST /v1/predict``: score feature vectors on demand (e.g. from a feature store or CRM event)
* ``GET  /v1/subscribers/{sub_id}``: latest batch score + reasons + action (call-centre / CRM lookup)
* ``GET  /health`` and ``GET /v1/model``: liveness and model provenance

Run: ``churn serve`` or ``uvicorn churn.api:app``. Configure with CHURN_MODEL_DIR and CHURN_SCORES.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.model import ChurnModel


class Subscriber(BaseModel):
    sub_id: int | None = None
    features: dict[str, Any] = Field(
        ..., description="Feature name -> value. Missing features are treated as unknown."
    )


class PredictRequest(BaseModel):
    subscribers: list[Subscriber] = Field(..., min_length=1, max_length=10_000)


class Reason(BaseModel):
    code: str
    description: str


class Prediction(BaseModel):
    sub_id: int | None
    churn_probability: float
    risk_band: str
    reasons: list[Reason]
    primary_driver: str
    recommended_action: str


class PredictResponse(BaseModel):
    model_version: str
    latency_ms: float
    predictions: list[Prediction]


def _reasons(*codes: str | None) -> list[Reason]:
    return [Reason(code=c, description=ChurnModel.describe_reason(c)) for c in codes if c]


def create_app(model_dir: str | Path | None = None, scores_path: str | Path | None = None) -> FastAPI:
    model_dir = model_dir or os.getenv("CHURN_MODEL_DIR", "artifacts/model")
    scores_path = scores_path or os.getenv("CHURN_SCORES")
    model = ChurnModel.load(model_dir)
    version = str(model.metadata.get("model_version", "unknown"))

    scores: pl.DataFrame | None = None
    if scores_path and Path(scores_path).exists():
        scores = pl.read_parquet(scores_path).sort("sub_id")
    sub_index = scores["sub_id"].to_numpy() if scores is not None else None

    app = FastAPI(title="Telco Churn Scoring API", version=version)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_version": version, "batch_scores_loaded": scores is not None}

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        keys = [
            "model_version",
            "trained_at",
            "label_definition",
            "train_cutoffs",
            "valid_cutoff",
            "test_cutoff",
            "best_iteration",
            "train_rows",
            "test_metrics",
            "risk_thresholds",
        ]
        return {k: model.metadata.get(k) for k in keys} | {
            "n_features": len(model.features),
            "features": model.features,
        }

    @app.post("/v1/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        t0 = time.perf_counter()
        X = model.matrix([s.features for s in req.subscribers])
        out = model.score_frame(X)
        preds = [
            Prediction(
                sub_id=s.sub_id,
                churn_probability=round(float(out["churn_probability"][i]), 6),
                risk_band=str(out["risk_band"][i]),
                reasons=_reasons(out["reason_1"][i], out["reason_2"][i]),
                primary_driver=str(out["primary_driver"][i]),
                recommended_action=str(out["recommended_action"][i]),
            )
            for i, s in enumerate(req.subscribers)
        ]
        return PredictResponse(
            model_version=version, latency_ms=round((time.perf_counter() - t0) * 1e3, 2), predictions=preds
        )

    @app.get("/v1/subscribers/{sub_id}")
    def subscriber(sub_id: int) -> dict[str, Any]:
        if scores is None:
            raise HTTPException(503, "No batch scores loaded. Run `churn score` and set CHURN_SCORES.")
        i = int(np.searchsorted(sub_index, sub_id))
        if i >= len(sub_index) or sub_index[i] != sub_id:
            raise HTTPException(404, f"Subscriber {sub_id} not in the active scored base")
        row = {
            k: None if isinstance(v, float) and v != v else v for k, v in scores.row(i, named=True).items()
        }
        row["reasons"] = [r.model_dump() for r in _reasons(row.pop("reason_1"), row.pop("reason_2"))]
        return row

    return app


def __getattr__(name: str):
    # lazily build `app` so importing the module (e.g. in tests) doesn't require a trained model
    if name == "app":
        return create_app()
    raise AttributeError(name)
