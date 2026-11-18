
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "artifacts" / "shared" / "candidate_features.csv"
OUTPUT_DIR = ROOT / "artifacts" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "distance_m",
    "signed_lateral_offset_m",
    "projection_fraction",
    "distance_to_arc_start_m",
    "distance_to_arc_end_m",
    "arc_length_m",
    "candidate_rank_distance",
    "candidate_count_at_point",
    "is_nearest_candidate",
    "is_within_search_radius",
    "gps_step_distance_m",
    "gps_speed_mps",
    "time_delta_s",
    "gps_bearing_valid",
    "heading_alignment_cos",
    "heading_alignment_sin",
    "is_self_loop",
]

TARGET = "label"
GROUP_COLUMNS = ["trip_id", "point_idx"]


def top1_accuracy(frame, probabilities):
    results = frame[GROUP_COLUMNS].copy()
    results["label"] = frame[TARGET].to_numpy()
    results["probability"] = probabilities

    chosen = results.loc[
        results.groupby(GROUP_COLUMNS)["probability"].idxmax()
    ]
    return float(chosen["label"].mean())


def main():
    print("Loading candidate dataset...")
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET, "split"] + GROUP_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET, "split"]).copy()

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "validation"].copy()

    if train.empty or valid.empty:
        raise ValueError("Training or validation data is empty.")

    X_train = train[FEATURES].astype("float32")
    y_train = train[TARGET].astype("int8")
    X_valid = valid[FEATURES].astype("float32")
    y_valid = valid[TARGET].astype("int8")

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    print(f"Training positive rate: {y_train.mean():.4f}")

    # Scalable logistic regression using stochastic gradient descent.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            max_iter=30,
            tol=0.001,
            class_weight="balanced",
            random_state=42,
            average=True,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("Predicting validation probabilities...")
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "Logistic Regression (SGD)",
        "training_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "top1_accuracy": top1_accuracy(valid, probabilities),
    }

    # Nearest-distance baseline for comparison.
    nearest = valid.loc[
        valid.groupby(GROUP_COLUMNS)["distance_m"].idxmin()
    ]
    metrics["nearest_candidate_baseline_top1"] = float(
        nearest[TARGET].mean()
    )

    model_path = OUTPUT_DIR / "logistic_regression.joblib"
    report_path = OUTPUT_DIR / "logistic_regression_metrics.json"

    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(metrics, indent=4))

    print("\n========== LOGISTIC REGRESSION RESULTS ==========")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    main()
