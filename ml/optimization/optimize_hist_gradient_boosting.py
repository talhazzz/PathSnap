"""Tune HistGradientBoosting with trip-grouped CV on training trips only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from skopt import Optimizer
from skopt.space import Integer, Real
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from ml.models.train_hist_gradient_boosting import pointwise_predictions
from ml.preprocessing.build_shared_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "artifacts/configs/bayesian_optimization.yaml"
SEARCH_SPACE = [
    Integer(80, 160, name="max_iter"),
    Real(0.03, 0.15, prior="log-uniform", name="learning_rate"),
    Integer(15, 63, name="max_leaf_nodes"),
    Integer(20, 60, name="min_samples_leaf"),
    Real(0.05, 8.0, prior="log-uniform", name="l2_regularization"),
]


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy and path values to standard JSON types."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtypes: dict[str, Any] = {
        "trip_id": "string",
        "point_idx": "int32",
        "candidate_arc_id": "int32",
        "split": "category",
        "label": "int8",
        **{column: "float32" for column in FEATURE_COLUMNS},
    }
    data = pd.read_csv(dataset_path, dtype=dtypes)
    required = {"trip_id", "point_idx", "candidate_arc_id", "split", "label", *FEATURE_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Candidate dataset is missing schema columns: {sorted(missing)}")
    training = data.loc[data["split"] == "train"].reset_index(drop=True)
    validation = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError("Training and validation splits must both be populated")
    overlap = set(training["trip_id"]) & set(validation["trip_id"])
    if overlap:
        raise ValueError(f"Trip leakage detected: {sorted(overlap)}")
    return training, validation


def balance_weights(labels: np.ndarray) -> np.ndarray:
    positive_rate = float(labels.mean())
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("Every training fold must contain both label classes")
    return np.where(labels == 1, 0.5 / positive_rate, 0.5 / (1.0 - positive_rate)).astype(
        np.float32
    )


def evaluate(
    model: HistGradientBoostingClassifier, validation: pd.DataFrame
) -> dict[str, Any]:
    features = validation[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = validation["label"].to_numpy(dtype=np.int8)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    ranked = validation[["trip_id", "point_idx", "candidate_arc_id", "label"]].copy()
    ranked["probability"] = probabilities
    ranked = pointwise_predictions(ranked, probabilities, "probability")
    ranked["is_correct"] = ranked["label"].astype(bool)
    per_trip = ranked.groupby("trip_id", observed=True)["is_correct"].mean()
    return {
        "candidate_row_metrics_at_threshold_0_5": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
        },
        "classifier_only_candidate_ranking": {
            "point_count": int(len(ranked)),
            "top1_accuracy": float(ranked["is_correct"].mean()),
            "macro_trip_top1_accuracy": float(per_trip.mean()),
            "per_trip_top1_accuracy": {
                str(trip_id): float(score) for trip_id, score in per_trip.items()
            },
        },
    }


def sample_by_trip(training: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for index, (_, trip) in enumerate(training.groupby("trip_id", sort=True, observed=True)):
        if len(trip) > cap:
            trip = trip.sample(n=cap, random_state=seed + index)
        pieces.append(trip)
    sampled = pd.concat(pieces, ignore_index=True)
    if sampled["trip_id"].nunique() != training["trip_id"].nunique():
        raise ValueError("Optimization sampling unexpectedly dropped a training trip")
    return sampled


def optimize(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: ROOT / path for name, path in settings["paths"].items()}
    search = settings["search"]
    training, validation = load_data(paths["dataset"])

    baseline_bundle = joblib.load(paths["baseline_model"])
    baseline_features = baseline_bundle["feature_columns"]
    if list(baseline_features) != FEATURE_COLUMNS:
        raise ValueError("Saved baseline model feature schema does not match the dataset")
    baseline = baseline_bundle["model"]
    baseline_metrics = evaluate(baseline, validation)

    optimization_data = sample_by_trip(
        training,
        int(search["max_rows_per_trip"]),
        int(search["random_seed"]),
    )
    x = optimization_data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = optimization_data["label"].to_numpy(dtype=np.int8)
    groups = optimization_data["trip_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(search["cv_splits"]))
    folds = list(splitter.split(x, y, groups))
    for train_index, validation_index in folds:
        if set(groups[train_index]) & set(groups[validation_index]):
            raise ValueError("Grouped cross-validation split contains trip leakage")

    optimizer = Optimizer(
        dimensions=SEARCH_SPACE,
        random_state=int(search["random_seed"]),
        n_initial_points=min(4, int(search["iterations"])),
        acq_func="EI",
    )
    results: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_cv_score = -np.inf
    for iteration in range(int(search["iterations"])):
        point = optimizer.ask()
        parameters = {
            dimension.name: (
                value.item() if isinstance(value, np.generic) else value
            )
            for dimension, value in zip(SEARCH_SPACE, point)
        }
        fold_scores: list[float] = []
        for train_index, validation_index in folds:
            model = HistGradientBoostingClassifier(
                **parameters,
                random_state=int(search["random_seed"]),
            )
            model.fit(
                x[train_index],
                y[train_index],
                sample_weight=balance_weights(y[train_index]),
            )
            probability = model.predict_proba(x[validation_index])[:, 1]
            fold_scores.append(
                float(average_precision_score(y[validation_index], probability))
            )
        mean_score = float(np.mean(fold_scores))
        optimizer.tell(point, -mean_score)
        row = {
            "iteration": iteration + 1,
            "cv_average_precision_mean": mean_score,
            "cv_average_precision_std": float(np.std(fold_scores)),
            **parameters,
        }
        results.append(row)
        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_parameters = parameters
        print(
            f"[{iteration + 1}/{search['iterations']}] grouped_cv_average_precision="
            f"{mean_score:.6f} params={parameters}"
        )

    if best_parameters is None:
        raise RuntimeError("Bayesian optimization produced no candidate parameters")

    tuned_model = HistGradientBoostingClassifier(
        **best_parameters,
        random_state=int(search["random_seed"]),
    )
    train_features = training[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_labels = training["label"].to_numpy(dtype=np.int8)
    tuned_model.fit(
        train_features,
        train_labels,
        sample_weight=balance_weights(train_labels),
    )
    tuned_metrics = evaluate(tuned_model, validation)

    paths["tuned_model"].parent.mkdir(parents=True, exist_ok=True)
    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": tuned_model, "feature_columns": FEATURE_COLUMNS},
        paths["tuned_model"],
        compress=3,
    )
    pd.DataFrame(results).to_csv(paths["results"], index=False)
    best_parameters_report = {
        "best_cv_average_precision": best_cv_score,
        "evaluations": int(search["iterations"]),
        "parameters": best_parameters,
    }
    metrics = {
        "method": "scikit-optimize Optimizer with expected improvement",
        "objective": "mean grouped-fold average precision",
        "student_test_used": False,
        "training_trips_for_search": int(optimization_data["trip_id"].nunique()),
        "training_rows_for_search": int(len(optimization_data)),
        "max_rows_per_trip": int(search["max_rows_per_trip"]),
        "cv_splits": int(search["cv_splits"]),
        "evaluations": int(search["iterations"]),
        "best_cv_average_precision": best_cv_score,
        "best_parameters": best_parameters,
        "baseline_validation": baseline_metrics,
        "tuned_validation": tuned_metrics,
        "results_csv": str(paths["results"].relative_to(ROOT)),
        "tuned_model": str(paths["tuned_model"].relative_to(ROOT)),
        "best_parameters_json": str(paths["best_parameters"].relative_to(ROOT)),
    }
    paths["best_parameters"].write_text(
        json.dumps(json_safe(best_parameters_report), indent=2), encoding="utf-8"
    )
    paths["metrics"].write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    optimize(args.config.resolve())


if __name__ == "__main__":
    main()