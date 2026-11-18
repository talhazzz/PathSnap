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

from backend.src.gemini_explainer import explain_trip_summary, is_gemini_configured
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
    align_feature_frame,
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
    """Pick best available trained model, prioritizing Medha's Bayesian-tuned HGB."""
    available = [m["name"] for m in list_available_models() if m["available"]]
    for candidate in (
        "hist_gradient_boosting_bayesian",
        "hist_gradient_boosting",
        "hist_gradient_boosting_tuned",
        "random_forest",
        "decision_tree",
        "knn",
        "linear_regression",
    ):
        if candidate in available:
            return candidate
    return available[0] if available else "random_forest"


class PredictionRequest(BaseModel):
    model_name: str = Field(default="hist_gradient_boosting_bayesian")
    features: list[dict[str, float]]


class TripPredictionRequest(BaseModel):
    model_name: str | None = Field(default=None)
    use_hmm: bool = Field(default=True)
    partition: str | None = Field(default=None)


class AIExplainTripRequest(BaseModel):
    trip_id: str
    model_name: str | None = Field(default=None)
    use_hmm: bool = Field(default=True)
    partition: str | None = Field(default=None)
    # Optional precomputed non-leaky trip summary from POST /predict/trip/{trip_id}
    summary: dict[str, Any] | None = Field(default=None)


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
    gemini_status = is_gemini_configured()
    logger.info("GET /health from %s | ready_models=%s", request.client.host if request.client else "unknown", ready_models)
    return {
        "status": "healthy",
        "ready_models": ready_models,
        "default_model": _default_model_name(),
        "ai_analyst_ready": gemini_status["ready"],
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

        try:
            frame = align_feature_frame(frame, columns)
        except ValueError:
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


@app.get("/trips/{trip_id}/geometry")
def get_trip_geometry(
    trip_id: str,
    request: Request,
    partition: str | None = Query(default=None),
):
    """Return raw GPS trajectory coordinates for a trip (NO ground-truth route exposed)."""
    part_name, part_dir = _resolve_partition_dir(partition)
    try:
        trip = load_trip_arrays(trip_id, part_dir, include_route=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    track = trip["track"]
    gps_coords = [[float(r[1]), float(r[0])] for r in track]
    logger.info(
        "GET /trips/%s/geometry from %s | partition=%s gps=%d",
        trip_id,
        request.client.host if request.client else "unknown",
        part_name,
        len(gps_coords),
    )
    return {
        "trip_id": trip_id,
        "partition": part_name,
        "gps_points_count": len(gps_coords),
        "gps_coordinates": gps_coords,
    }


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
    """Return recorded training, Bayesian tuning, and 5-model evaluation summaries."""
    summary: dict[str, Any] = {}
    for key, filename in [
        ("baseline_dt_rf", "model_comparison_mithun.json"),
        ("extra_models", "extra_models_metrics.json"),
        ("bayesian_tuning_hgb", "hgb_tuning_summary.json"),
        ("all_models_student_test", "all_models_student_test_summary.json"),
    ]:
        path = RESULT_DIR / filename
        if path.exists():
            summary[key] = json.loads(path.read_text(encoding="utf-8"))
    logger.info("GET /metrics/summary from %s", request.client.host if request.client else "unknown")
    return summary


@app.post("/ai/explain-trip")
def ai_explain_trip(payload: AIExplainTripRequest, request: Request):
    """Optional Gemini-powered AI Route Analyst.

    Accepts either a precomputed non-leaky trip prediction summary or runs the
    selected pipeline (evaluate_server_side=False so ground truth NEVER enters prompts).
    Always returns HTTP 200 with structured graceful degradation if GEMINI_API_KEY
    is unconfigured or Gemini is unreachable; core map matching is unaffected.
    """
    chosen_model = payload.model_name or _default_model_name()

    if payload.summary:
        # Whitelist ONLY safe summary statistics (never coordinates or reference routes)
        safe_keys = {
            "trip_id",
            "model_name",
            "model_type",
            "use_hmm",
            "gps_points_count",
            "candidate_rows_count",
            "predicted_arc_count",
            "approx_distance_km",
            "duration_seconds",
            "route_continuity",
            "classifier_only_route_continuity",
            "timings_seconds",
        }
        summary = {k: payload.summary[k] for k in safe_keys if k in payload.summary}
        summary.setdefault("trip_id", payload.trip_id)
        summary.setdefault("model_name", chosen_model)
        summary.setdefault("use_hmm", payload.use_hmm)
    else:
        part_name, part_dir = _resolve_partition_dir(payload.partition)
        try:
            artifact = _get_cached_model(chosen_model)
            res = run_trip_inference(
                trip_id=payload.trip_id,
                model_artifact=artifact,
                partition_dir=part_dir,
                use_hmm=payload.use_hmm,
                evaluate_server_side=False,  # Ground truth NEVER enters AI prompts!
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        summary = {
            "trip_id": res["trip_id"],
            "model_name": res["model_name"],
            "model_type": res["model_type"],
            "use_hmm": res["use_hmm"],
            "gps_points_count": res["gps_points_count"],
            "candidate_rows_count": res["candidate_rows_count"],
            "predicted_arc_count": res["predicted_arc_count"],
            "approx_distance_km": res["approx_distance_km"],
            "duration_seconds": res["duration_seconds"],
            "route_continuity": res["route_continuity"],
            "classifier_only_route_continuity": res["classifier_only_route_continuity"],
            "timings_seconds": res["timings_seconds"],
        }

    ai_result = explain_trip_summary(summary)
    logger.info(
        "POST /ai/explain-trip from %s | trip=%s model=%s ai_status=%s",
        request.client.host if request.client else "unknown",
        summary.get("trip_id"),
        summary.get("model_name"),
        ai_result.get("status"),
    )
    return {
        "trip_id": summary.get("trip_id"),
        "model_name": summary.get("model_name"),
        **ai_result,
    }
