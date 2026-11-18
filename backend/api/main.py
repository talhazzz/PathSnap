"""PathSnap FastAPI Backend Service (Laptop 1 — Model, HMM/Viterbi & API).

Supports the Two-Laptop Distributed Architecture:
- Laptop 1 hosts trained ML classifiers, HMM/Viterbi decoding, road-network data,
  and server-side evaluation (never exposing ground-truth labels to model or UI).
- Laptop 2 runs the evaluator UI, fetches the 10 evaluation trips via GET /trips,
  triggers inference via POST /predict/trip/{trip_id}, and visualizes returned routes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.src.train_models import (
    MODEL_DIR,
    RESULT_DIR,
    list_available_models,
    load_model,
    predict,
    predict_proba,
    score_candidates,
)
from backend.src.trip_pipeline import (
    DEFAULT_DATA_ROOT,
    generate_folium_map,
    load_trip_arrays,
    project_coordinates,
    run_trip_inference,
    track_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap Backend API",
    description=(
        "Distributed ML Map-Matching & HMM/Viterbi Route Reconstruction Service "
        "(Laptop 1 API Server)"
    ),
    version="2.0.0",
)

# Allow Laptop 2 UI (any LAN IP / Vite dev port) to call Laptop 1 API cleanly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_partition_dir(partition: str | None = None) -> tuple[str, Path]:
    """Resolve the active evaluation dataset folder.

    Priority:
    1. Explicit `partition` query/body param ('evaluator_test', 'student_test', 'train').
    2. `PATHSNAP_EVAL_DIR` environment variable if set and exists.
    3. `evaluator_test` folder under Student folder if evaluators placed it there.
    4. Fallback to `student_test` (10 development evaluation trips).
    """
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if partition is None:
        if env_dir and Path(env_dir).is_dir():
            return "evaluator_test", Path(env_dir)
        eval_default = DEFAULT_DATA_ROOT / "evaluator_test"
        if eval_default.is_dir():
            return "evaluator_test", eval_default
        return "student_test", DEFAULT_DATA_ROOT / "student_test"

    if partition == "evaluator_test":
        if env_dir and Path(env_dir).is_dir():
            return "evaluator_test", Path(env_dir)
        candidate = DEFAULT_DATA_ROOT / "evaluator_test"
        if candidate.is_dir():
            return "evaluator_test", candidate
        raise HTTPException(
            status_code=404,
            detail="evaluator_test directory not found; set PATHSNAP_EVAL_DIR or place trips in backend/data/raw/Student folder/evaluator_test",
        )

    if partition in {"student_test", "train"}:
        candidate = DEFAULT_DATA_ROOT / partition
        if candidate.is_dir():
            return partition, candidate
        raise HTTPException(status_code=404, detail=f"Partition directory not found: {candidate}")

    raise HTTPException(
        status_code=400,
        detail="partition must be one of: evaluator_test, student_test, train",
    )


@lru_cache(maxsize=8)
def _get_cached_model(model_name: str) -> dict[str, Any]:
    """Load and cache trained .joblib artifacts in memory for low-latency serving."""
    logger.info("Loading model artifact into memory: %s", model_name)
    return load_model(model_name)


def _default_model_name() -> str:
    """Pick best available trained model across backend/models and artifacts/models."""
    available = [m["name"] for m in list_available_models() if m["available"]]
    for candidate in ("random_forest_tuned", "hist_gradient_boosting", "random_forest", "decision_tree"):
        if candidate in available:
            return candidate
    return available[0] if available else "random_forest"


class PredictionRequest(BaseModel):
    model_name: str = Field(default="random_forest_tuned")
    features: list[dict[str, float]]


class TripPredictionRequest(BaseModel):
    model_name: str | None = Field(default=None)
    use_hmm: bool = Field(default=True)
    partition: str | None = Field(default=None)


@app.get("/")
def root(request: Request):
    part_name, part_dir = _resolve_partition_dir(None)
    trip_count = len([p for p in part_dir.iterdir() if p.is_dir()]) if part_dir.is_dir() else 0
    logger.info("GET / from %s", request.client.host if request.client else "unknown")
    return {
        "project": "PathSnap",
        "service": "Distributed Map-Matching Backend API (Laptop 1)",
        "status": "running",
        "default_model": _default_model_name(),
        "active_eval_partition": part_name,
        "available_eval_trips": trip_count,
    }


@app.get("/health")
def health(request: Request):
    models = list_available_models()
    ready_models = [m["name"] for m in models if m["available"]]
    logger.info("GET /health from %s | ready_models=%s", request.client.host if request.client else "unknown", ready_models)
    return {
        "status": "healthy",
        "ready_models": ready_models,
        "default_model": _default_model_name(),
    }


@app.get("/models")
def get_models(request: Request):
    models = list_available_models()
    logger.info("GET /models from %s", request.client.host if request.client else "unknown")
    return {
        "default_model": _default_model_name(),
        "models": models,
    }


@lru_cache(maxsize=4)
def _load_trips_metadata_cached(part_name: str, part_dir_str: str) -> tuple[dict[str, Any], ...]:
    """Load and cache non-leaky trip telemetry for fast GET /trips responses."""
    part_dir = Path(part_dir_str)
    trip_dirs = sorted(p for p in part_dir.iterdir() if p.is_dir())
    trips_out = []
    for tdir in trip_dirs:
        tid = tdir.name
        try:
            trip = load_trip_arrays(tid, part_dir, include_route=False)
            track = trip["track"]
            origin_lon = float(np.median(track[:, 0]))
            origin_lat = float(np.median(track[:, 1]))
            track_xy = project_coordinates(track[:, :2], origin_lon, origin_lat)
            step_dist, _, _, _, _ = track_context(track_xy, track[:, 2])
            duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0
            trips_out.append(
                {
                    "trip_id": tid,
                    "partition": part_name,
                    "gps_points": int(len(track)),
                    "nodes": int(len(trip["nodes"])),
                    "arcs": int(len(trip["arcs"])),
                    "duration_seconds": round(duration_s, 1),
                    "approx_distance_km": round(float(np.sum(step_dist)) / 1000.0, 3),
                }
            )
        except Exception as exc:
            logger.warning("Skipping unreadable trip %s: %s", tid, exc)
    return tuple(trips_out)


@app.get("/trips")
def list_trips(
    request: Request,
    partition: str | None = Query(default=None, description="evaluator_test, student_test, or train"),
):
    """List available trips with basic non-leaky telemetry (NO ground-truth route exposed)."""
    part_name, part_dir = _resolve_partition_dir(partition)
    trips_out = list(_load_trips_metadata_cached(part_name, str(part_dir.resolve())))

    logger.info(
        "GET /trips from %s | partition=%s count=%d",
        request.client.host if request.client else "unknown",
        part_name,
        len(trips_out),
    )
    return {
        "partition": part_name,
        "count": len(trips_out),
        "trips": trips_out,
    }


@app.post("/predict")
def make_prediction(payload: PredictionRequest, request: Request):
    """Backward-compatible candidate-level batch prediction endpoint."""
    if not payload.features:
        raise HTTPException(
            status_code=400,
            detail="At least one feature row is required",
        )

    try:
        artifact = _get_cached_model(payload.model_name)
        columns = artifact["feature_columns"]
        frame = pd.DataFrame(payload.features)

        if set(frame.columns) != set(columns):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Feature columns do not match trained model",
                    "expected": columns,
                    "received": list(frame.columns),
                },
            )

        start = time.perf_counter()
        scored = score_candidates(payload.model_name, frame)
        elapsed = time.perf_counter() - start

        logger.info(
            "POST /predict from %s | model=%s type=%s rows=%d time=%.4fs",
            request.client.host if request.client else "unknown",
            payload.model_name,
            scored["model_type"],
            len(frame),
            elapsed,
        )

        if scored["model_type"] == "regression_scorer":
            scores = scored["scores"]
            ranked_indices = np.argsort(-scores).tolist()
            return {
                "model": payload.model_name,
                "model_type": "regression_scorer",
                "scores": [round(float(s), 6) for s in scores],
                "ranked_indices": ranked_indices,
                "count": len(scores),
                "inference_time_seconds": round(elapsed, 5),
            }

        predictions = scored["predictions"]
        probabilities = scored["probabilities"]
        return {
            "model": payload.model_name,
            "model_type": "classifier",
            "predictions": predictions.tolist(),
            "probabilities": [round(float(p), 6) for p in probabilities],
            "count": len(predictions),
            "inference_time_seconds": round(elapsed, 5),
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/trip/{trip_id}")
def predict_trip(
    trip_id: str,
    request: Request,
    body: TripPredictionRequest | None = None,
    model_name: str | None = Query(default=None),
    use_hmm: bool | None = Query(default=None),
    partition: str | None = Query(default=None),
):
    """Run complete end-to-end map-matching pipeline on a trip (feature extraction -> ML -> HMM/Viterbi).

    Ground-truth .route is NEVER passed to feature extraction, ML scoring, or HMM decoding.
    If reference .route exists server-side, accuracy metrics are computed strictly after
    prediction completes, and raw reference arc labels are NEVER returned to the UI.
    """
    chosen_model = (
        model_name
        or (body.model_name if body and body.model_name else None)
        or _default_model_name()
    )
    chosen_hmm = use_hmm if use_hmm is not None else (body.use_hmm if body is not None else True)
    chosen_partition = partition or (body.partition if body else None)
    part_name, part_dir = _resolve_partition_dir(chosen_partition)

    try:
        artifact = _get_cached_model(chosen_model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        result = run_trip_inference(
            trip_id=trip_id,
            model_artifact=artifact,
            partition_dir=part_dir,
            use_hmm=chosen_hmm,
            evaluate_server_side=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Trip inference failed for trip=%s", trip_id)
        raise HTTPException(status_code=500, detail=str(exc))

    # Strip non-serializable internal arrays before responding
    result.pop("_internal", None)
    result["partition"] = part_name

    ev = result.get("server_evaluation") or {}
    hmm_acc = ev.get("hmm_viterbi", {}).get("top1_accuracy") if ev else None
    clf_acc = ev.get("classifier_only", {}).get("top1_accuracy") if ev else None

    logger.info(
        "POST /predict/trip/%s from %s | partition=%s model=%s hmm=%s | "
        "gps=%d pred_arcs=%d continuity=%.4f clf_top1=%s hmm_top1=%s total_time=%.3fs",
        trip_id,
        request.client.host if request.client else "unknown",
        part_name,
        chosen_model,
        chosen_hmm,
        result["gps_points_count"],
        result["predicted_arc_count"],
        result["route_continuity"],
        f"{clf_acc:.4f}" if clf_acc is not None else "N/A",
        f"{hmm_acc:.4f}" if hmm_acc is not None else "N/A",
        result["timings_seconds"]["total"],
    )
    return result


@app.get("/trips/{trip_id}/map", response_class=HTMLResponse)
def get_trip_map(
    trip_id: str,
    request: Request,
    model_name: str | None = Query(default=None),
    use_hmm: bool = Query(default=True),
    partition: str | None = Query(default=None),
):
    """Return an interactive Folium HTML map for the reconstructed trip route."""
    chosen_model = model_name or _default_model_name()
    part_name, part_dir = _resolve_partition_dir(partition)

    try:
        artifact = _get_cached_model(chosen_model)
        result = run_trip_inference(
            trip_id=trip_id,
            model_artifact=artifact,
            partition_dir=part_dir,
            use_hmm=use_hmm,
            evaluate_server_side=False,  # Never expose ground truth on live UI map!
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    internal = result["_internal"]
    html = generate_folium_map(
        trip_id=trip_id,
        track=internal["track"],
        nodes=internal["nodes"],
        arcs=internal["arcs"],
        hmm_route_arcs=internal["hmm_route_arcs"],
        clf_point_arcs=internal["clf_point_arcs"],
        route_ref=None,  # Strict rule: ground truth never sent to UI
        corridor_arc_ids=internal["corridor_arc_ids"],
    )
    logger.info(
        "GET /trips/%s/map from %s | partition=%s model=%s",
        trip_id,
        request.client.host if request.client else "unknown",
        part_name,
        chosen_model,
    )
    return HTMLResponse(content=html)


@app.get("/metrics/summary")
def get_metrics_summary(request: Request):
    """Return recorded training, Bayesian tuning, and HMM evaluation summaries."""
    summary: dict[str, Any] = {}
    comp_path = RESULT_DIR / "model_comparison_mithun.json"
    tuning_path = RESULT_DIR / "rf_tuning_summary.json"
    if comp_path.exists():
        summary["baseline_models"] = json.loads(comp_path.read_text(encoding="utf-8"))
    if tuning_path.exists():
        summary["bayesian_tuning"] = json.loads(tuning_path.read_text(encoding="utf-8"))
    logger.info("GET /metrics/summary from %s", request.client.host if request.client else "unknown")
    return summary
