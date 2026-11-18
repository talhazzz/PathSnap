"""Train and evaluate the owned HistGradientBoosting candidate ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

from ml.preprocessing.build_shared_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "artifacts/configs/hist_gradient_boosting.yaml"


def pointwise_predictions(
    frame: pd.DataFrame, scores: np.ndarray, score_column: str
) -> pd.DataFrame:
    ranked = frame[["trip_id", "point_idx", "candidate_arc_id", "label"]].copy()
    ranked[score_column] = scores
    return (
        ranked.sort_values(
            ["trip_id", "point_idx", score_column, "candidate_arc_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["trip_id", "point_idx"])
        .reset_index(drop=True)
    )


def ranking_metrics(ranked: pd.DataFrame) -> dict[str, Any]:
    per_trip = ranked.groupby("trip_id")["is_correct"].mean()
    return {
        "point_count": int(len(ranked)),
        "top1_accuracy": float(ranked["is_correct"].mean()),
        "macro_trip_top1_accuracy": float(per_trip.mean()),
        "per_trip_top1_accuracy": {
            str(trip_id): float(value) for trip_id, value in per_trip.items()
        },
    }


def train(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = ROOT / settings["paths"]["dataset"]
    model_path = ROOT / settings["paths"]["model"]
    metrics_path = ROOT / settings["paths"]["metrics"]
    dtypes: dict[str, Any] = {
        "trip_id": "string",
        "point_idx": "int32",
        "candidate_arc_id": "int32",
        "split": "category",
        "label": "int8",
        **{column: "float32" for column in FEATURE_COLUMNS},
    }
    data = pd.read_csv(dataset_path, dtype=dtypes)
    train_frame = data.loc[data["split"] == "train"].reset_index(drop=True)
    validation_frame = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Both trip-level train and validation partitions must be nonempty")
    train_trips = set(train_frame["trip_id"])
    validation_trips = set(validation_frame["trip_id"])
    if train_trips & validation_trips:
        raise ValueError("Trip leakage detected between train and validation")

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    x_validation = validation_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_validation = validation_frame["label"].to_numpy(dtype=np.int8)
    model_options = dict(settings["model"])
    model = HistGradientBoostingClassifier(**model_options)
    sample_weight = None
    if settings["training"]["balance_classes"]:
        positive_rate = float(y_train.mean())
        sample_weight = np.where(
            y_train == 1,
            0.5 / positive_rate,
            0.5 / (1.0 - positive_rate),
        ).astype(np.float32)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    probabilities = model.predict_proba(x_validation)[:, 1]
    row_predictions = (probabilities >= 0.5).astype(np.int8)
    validation_frame["_score"] = probabilities
    ranked = pointwise_predictions(validation_frame, probabilities, "_score")
    ranked["is_correct"] = ranked["label"].astype(bool)

    nearest = pointwise_predictions(
        validation_frame,
        -validation_frame["distance_m"].to_numpy(),
        "_score",
    )
    nearest["is_correct"] = nearest["label"].astype(bool)
    importance_settings = settings.get("feature_importance", {})
    importance_sample = validation_frame.sample(
        n=min(
            int(importance_settings.get("validation_sample_size", 20000)),
            len(validation_frame),
        ),
        random_state=int(model_options.get("random_state", 42)),
    )
    importance = permutation_importance(
        model,
        importance_sample[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        importance_sample["label"].to_numpy(dtype=np.int8),
        scoring="average_precision",
        n_repeats=int(importance_settings.get("repeats", 3)),
        random_state=int(model_options.get("random_state", 42)),
        n_jobs=1,
    )
    metrics = {
        "model": "HistGradientBoostingClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "model_parameters": model_options,
        "class_balancing": bool(settings["training"]["balance_classes"]),
        "data_source": "artifacts/shared/candidate_features.csv; train split only for fitting",
        "student_test_used": False,
        "train": {
            "rows": int(len(train_frame)),
            "trips": int(len(train_trips)),
            "positive_rows": int(y_train.sum()),
            "negative_rows": int(len(y_train) - y_train.sum()),
        },
        "validation": {
            "rows": int(len(validation_frame)),
            "trips": int(len(validation_trips)),
            "positive_rows": int(y_validation.sum()),
            "negative_rows": int(len(y_validation) - y_validation.sum()),
            "candidate_row_metrics_at_threshold_0_5": {
                "accuracy": float(accuracy_score(y_validation, row_predictions)),
                "precision": float(precision_score(y_validation, row_predictions, zero_division=0)),
                "recall": float(recall_score(y_validation, row_predictions, zero_division=0)),
                "f1": float(f1_score(y_validation, row_predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_validation, probabilities)),
                "average_precision": float(average_precision_score(y_validation, probabilities)),
            },
            "classifier_only_candidate_ranking": ranking_metrics(ranked),
            "nearest_road_baseline": ranking_metrics(nearest),
        },
        "feature_importance": {
            "method": "permutation importance on held-out validation rows",
            "scoring": "average_precision",
            "sample_rows": int(len(importance_sample)),
            "repeats": int(importance_settings.get("repeats", 3)),
            "values": [
                {
                    "feature": feature,
                    "mean": float(mean),
                    "std": float(std),
                }
                for feature, mean, std in sorted(
                    zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS}, model_path, compress=3
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path.relative_to(ROOT)), "metrics": metrics}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    train(args.config.resolve())


if __name__ == "__main__":
    main()