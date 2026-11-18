"""Train ONLY the 3 newly added PathSnap models (HistGradientBoosting, KNN, Linear Regression)
on the canonical 17-feature shared dataset without touching Decision Tree or Random Forest.

- HistGradientBoosting: reuses exact hyperparameters & class balancing from
  origin/ml-dev:artifacts/configs/hist_gradient_boosting.yaml.
- KNN: StandardScaler + distance-weighted KD-tree KNeighborsClassifier fit on a
  deterministic trip-stratified subsample for sub-second live inference.
- Linear Regression: StandardScaler + OLS LinearRegression implemented strictly as
  a continuous candidate scoring/ranking model (model_type="regression_scorer").
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train_models import (
    MODEL_DIR,
    RESULT_DIR,
    compute_top1_accuracy,
    evaluate_predictions,
    load_shared_dataset,
    resolve_shared_dataset_path,
)
from src.trip_pipeline import FEATURE_COLUMNS

# Exact config from origin/ml-dev:artifacts/configs/hist_gradient_boosting.yaml
HGB_PARAMS: dict[str, Any] = {
    "max_iter": 150,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}

KNN_PARAMS: dict[str, Any] = {
    "n_neighbors": 15,
    "weights": "distance",
    "algorithm": "kd_tree",
    "leaf_size": 40,
    "n_jobs": -1,
}
KNN_MAX_TRAIN_ROWS = 60_000


def _trip_stratified_sample(
    frame: pd.DataFrame, max_rows: int, seed: int = 42
) -> pd.DataFrame:
    """Deterministically sample up to max_rows proportionally across trips."""
    if len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    frac = max_rows / len(frame)
    parts = []
    for _, grp in frame.groupby("trip_id", sort=True):
        k = max(2, int(round(len(grp) * frac)))
        k = min(k, len(grp))
        idx = rng.choice(len(grp), size=k, replace=False)
        parts.append(grp.iloc[idx])
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return sampled


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    pos_rate = float(y.mean())
    return np.where(y == 1, 0.5 / pos_rate, 0.5 / (1.0 - pos_rate)).astype(np.float32)


def train_three_models() -> dict[str, Any]:
    dataset_path = resolve_shared_dataset_path()
    print(f"Loading canonical shared dataset: {dataset_path}", flush=True)
    data = load_shared_dataset(dataset_path)

    train_frame = data.loc[data["split"] == "train"].reset_index(drop=True)
    val_frame = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if set(train_frame["trip_id"]) & set(val_frame["trip_id"]):
        raise ValueError("Trip leakage detected between train and validation splits")

    X_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    X_val = val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_frame["label"].to_numpy(dtype=np.int8)
    X_all = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_all = data["label"].to_numpy(dtype=np.int8)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "feature_columns": FEATURE_COLUMNS,
        "models": {},
    }

    # ------------------------------------------------------------------
    # 1. HistGradientBoosting (exact Medha ml-dev config)
    # ------------------------------------------------------------------
    print("\n[1/3] Training HistGradientBoosting (ml-dev config)...", flush=True)
    hgb_val_model = HistGradientBoostingClassifier(**HGB_PARAMS)
    t0 = time.perf_counter()
    hgb_val_model.fit(X_train, y_train, sample_weight=_balanced_weights(y_train))
    hgb_split_s = time.perf_counter() - t0
    hgb_val_probs = hgb_val_model.predict_proba(X_val)[:, 1]
    hgb_val_metrics = evaluate_predictions(val_frame, y_val, hgb_val_probs)

    hgb_final = HistGradientBoostingClassifier(**HGB_PARAMS)
    t1 = time.perf_counter()
    hgb_final.fit(X_all, y_all, sample_weight=_balanced_weights(y_all))
    hgb_full_s = time.perf_counter() - t1

    hgb_path = MODEL_DIR / "hist_gradient_boosting.joblib"
    joblib.dump(
        {
            "model": hgb_final,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": "hist_gradient_boosting",
            "display_name": "HistGradientBoosting",
            "model_type": "classifier",
            "algorithm_family": "Boosting",
            "hyperparameters": {**HGB_PARAMS, "balance_classes": True},
            "config_source": "origin/ml-dev:artifacts/configs/hist_gradient_boosting.yaml",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        hgb_path,
        compress=3,
    )
    print(
        f"  Saved {hgb_path.name} ({hgb_path.stat().st_size / (1024*1024):.2f} MB) | "
        f"Val Macro F1={hgb_val_metrics['f1_macro']:.4f} | Val Top-1={hgb_val_metrics['top1_accuracy']:.4f} | "
        f"Val LogLoss={hgb_val_metrics['log_loss']:.4f} | time={hgb_full_s:.2f}s",
        flush=True,
    )
    report["models"]["hist_gradient_boosting"] = {
        "model_type": "classifier",
        "algorithm_family": "Boosting",
        "hyperparameters": {**HGB_PARAMS, "balance_classes": True},
        "validation_split_metrics": hgb_val_metrics,
        "split_train_time_s": round(hgb_split_s, 4),
        "full_80trip_train_time_s": round(hgb_full_s, 4),
        "model_size_mb": round(hgb_path.stat().st_size / (1024 * 1024), 2),
    }

    # ------------------------------------------------------------------
    # 2. K-Nearest Neighbors (Scaled distance-weighted KD-tree)
    # ------------------------------------------------------------------
    print("\n[2/3] Training K-Nearest Neighbors (KNN)...", flush=True)
    knn_train_sub = _trip_stratified_sample(train_frame, KNN_MAX_TRAIN_ROWS, seed=42)
    X_knn_tr = knn_train_sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_knn_tr = knn_train_sub["label"].to_numpy(dtype=np.int8)

    knn_val_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(**KNN_PARAMS)),
        ]
    )
    t0 = time.perf_counter()
    knn_val_pipe.fit(X_knn_tr, y_knn_tr)
    knn_split_s = time.perf_counter() - t0
    knn_val_probs = knn_val_pipe.predict_proba(X_val)[:, 1]
    knn_val_metrics = evaluate_predictions(val_frame, y_val, knn_val_probs)

    knn_all_sub = _trip_stratified_sample(data, KNN_MAX_TRAIN_ROWS, seed=42)
    X_knn_all = knn_all_sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_knn_all = knn_all_sub["label"].to_numpy(dtype=np.int8)
    knn_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(**KNN_PARAMS)),
        ]
    )
    t1 = time.perf_counter()
    knn_final.fit(X_knn_all, y_knn_all)
    knn_full_s = time.perf_counter() - t1

    knn_path = MODEL_DIR / "knn.joblib"
    joblib.dump(
        {
            "model": knn_final,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": "knn",
            "display_name": "K-Nearest Neighbors (KNN)",
            "model_type": "classifier",
            "algorithm_family": "Instance-based",
            "hyperparameters": {
                **KNN_PARAMS,
                "scaling": "StandardScaler",
                "trip_stratified_reference_rows": int(len(knn_all_sub)),
            },
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        knn_path,
        compress=3,
    )
    print(
        f"  Saved {knn_path.name} ({knn_path.stat().st_size / (1024*1024):.2f} MB) | "
        f"Val Macro F1={knn_val_metrics['f1_macro']:.4f} | Val Top-1={knn_val_metrics['top1_accuracy']:.4f} | "
        f"Val LogLoss={knn_val_metrics['log_loss']:.4f} | time={knn_full_s:.2f}s",
        flush=True,
    )
    report["models"]["knn"] = {
        "model_type": "classifier",
        "algorithm_family": "Instance-based",
        "hyperparameters": {
            **KNN_PARAMS,
            "scaling": "StandardScaler",
            "trip_stratified_reference_rows": int(len(knn_all_sub)),
        },
        "validation_split_metrics": knn_val_metrics,
        "split_train_time_s": round(knn_split_s, 4),
        "full_80trip_train_time_s": round(knn_full_s, 4),
        "model_size_mb": round(knn_path.stat().st_size / (1024 * 1024), 2),
    }

    # ------------------------------------------------------------------
    # 3. Linear Regression (Continuous Candidate Scoring / Ranking Model)
    # ------------------------------------------------------------------
    print("\n[3/3] Training Linear Regression (Candidate Scorer / Ranker)...", flush=True)
    lr_val_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    t0 = time.perf_counter()
    lr_val_pipe.fit(X_train, y_train.astype(np.float32))
    lr_split_s = time.perf_counter() - t0
    lr_val_scores = lr_val_pipe.predict(X_val).astype(np.float64)
    lr_top1 = compute_top1_accuracy(val_frame, lr_val_scores)
    mse = float(np.mean((y_val.astype(np.float64) - lr_val_scores) ** 2))
    r2 = float(lr_val_pipe.score(X_val, y_val.astype(np.float32)))

    lr_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    t1 = time.perf_counter()
    lr_final.fit(X_all, y_all.astype(np.float32))
    lr_full_s = time.perf_counter() - t1

    lr_path = MODEL_DIR / "linear_regression.joblib"
    joblib.dump(
        {
            "model": lr_final,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": "linear_regression",
            "display_name": "Linear Regression (Candidate Scorer)",
            "model_type": "regression_scorer",
            "algorithm_family": "Linear (OLS Candidate Ranking)",
            "hyperparameters": {
                "scaling": "StandardScaler",
                "fit_intercept": True,
                "hmm_softmax_temperature": 5.0,
            },
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        lr_path,
        compress=3,
    )
    print(
        f"  Saved {lr_path.name} ({lr_path.stat().st_size / (1024*1024):.2f} MB) | "
        f"Val Top-1={lr_top1['top1_accuracy']:.4f} | Val MSE={mse:.5f} | Val R2={r2:.4f} | time={lr_full_s:.2f}s",
        flush=True,
    )
    report["models"]["linear_regression"] = {
        "model_type": "regression_scorer",
        "algorithm_family": "Linear (OLS Candidate Ranking)",
        "hyperparameters": {
            "scaling": "StandardScaler",
            "fit_intercept": True,
            "hmm_softmax_temperature": 5.0,
        },
        "validation_split_metrics": {
            "rows": int(len(y_val)),
            "top1_accuracy": lr_top1["top1_accuracy"],
            "macro_trip_top1_accuracy": lr_top1["macro_trip_top1_accuracy"],
            "evaluated_points": lr_top1["point_count"],
            "mse": mse,
            "r2": r2,
            "log_loss": None,
        },
        "split_train_time_s": round(lr_split_s, 4),
        "full_80trip_train_time_s": round(lr_full_s, 4),
        "model_size_mb": round(lr_path.stat().st_size / (1024 * 1024), 2),
    }

    out_path = RESULT_DIR / "extra_models_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved extra model metrics -> {out_path}", flush=True)
    return report


if __name__ == "__main__":
    train_three_models()
