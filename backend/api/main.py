from pathlib import Path
import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.src.train_models import load_model, predict

app = FastAPI(
    title="PathSnap Backend API",
    description="Decision Tree and Random Forest inference service",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathsnap")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

class PredictionRequest(BaseModel):
    model_name: str = Field(
        default="random_forest",
        pattern="^(decision_tree|random_forest)$"
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
            for name in ["decision_tree", "random_forest"]
        ]
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    if not request.features:
        raise HTTPException(
            status_code=400,
            detail="At least one feature row is required",
        )

    try:
        artifact = load_model(request.model_name)
        columns = artifact["feature_columns"]

        frame = pd.DataFrame(request.features)

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
        predictions = predict(request.model_name, frame)
        elapsed = time.perf_counter() - start

        logger.info(
            "Prediction completed: model=%s rows=%d time=%.4fs",
            request.model_name,
            len(frame),
            elapsed,
        )

        return {
            "model": request.model_name,
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "inference_time_seconds": elapsed,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
