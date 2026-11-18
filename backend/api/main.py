import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.src.model_registry import (
    MODEL_NAMES,
    MODEL_DIR,
    load_model,
    get_feature_columns,
    predict_model,
)

app = FastAPI(
    title="PathSnap Backend API",
    description="Four-model GPS candidate classification service",
    version="1.1.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathsnap")


class PredictionRequest(BaseModel):
    model_name: str = Field(
        default="random_forest",
        pattern="^(logistic_regression|knn|decision_tree|random_forest)$",
    )
    features: list[dict[str, float]]


@app.get("/")
def root():
    return {
        "project": "PathSnap",
        "service": "Backend API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def get_models():
    return {
        "models": [
            {
                "name": name,
                "available": (MODEL_DIR / f"{name}.joblib").exists(),
            }
            for name in MODEL_NAMES
        ]
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="No feature rows supplied")

    try:
        artifact = load_model(request.model_name)
        columns = get_feature_columns(request.model_name, artifact)
        frame = pd.DataFrame(request.features)

        missing = set(columns) - set(frame.columns)
        extra = set(columns) - set(frame.columns)

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Feature columns do not match trained model",
                    "expected": columns,
                    "missing": sorted(missing),
                    "unexpected": sorted(extra),
                },
            )

        start = time.perf_counter()
        result = predict_model(request.model_name, frame)
        elapsed = time.perf_counter() - start

        logger.info(
            "Prediction completed: model=%s rows=%d time=%.4fs",
            request.model_name,
            len(frame),
            elapsed,
        )

        return {
            "model": request.model_name,
            "predictions": result["predictions"],
            "probabilities": result["probabilities"],
            "count": len(result["predictions"]),
            "inference_time_seconds": elapsed,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
