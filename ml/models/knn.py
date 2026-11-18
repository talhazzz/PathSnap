
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "artifacts/shared/candidate_features.csv"
MODEL_PATH = ROOT / "artifacts/models/logistic_regression.joblib"
OUTPUT_DIR = ROOT / "artifacts/models"
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

GROUPS = ["trip_id", "point_idx"]
RANDOM_SEED = 42


def calculate_metrics(frame, probabilities, name):
    y = frame["label"].to_numpy()
    predictions = (probabilities >= 0.5).astype(int)

    grouped = frame[GROUPS].copy()
    grouped["label"] = y
    grouped["probability"] = probabilities

    best = grouped.loc[
        grouped.groupby(GROUPS)["probability"].idxmax()
    ]

    return {
        "model": name,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "top1_accuracy": float(best["label"].mean()),
    }


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + ["label", "split"])

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "validation"].copy()

    # Sample training candidates with equal representation of both classes.
    samples_per_class = 10_000
    train_sample = (
        train.groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), samples_per_class),
                random_state=RANDOM_SEED,
            ),
            include_groups=True,
        )
        .reset_index(drop=True)
    )

    # Sample complete GPS observations, not individual candidate rows.
    rng = np.random.default_rng(RANDOM_SEED)
    unique_groups = valid[GROUPS].drop_duplicates()
    selected_indices = rng.choice(
        len(unique_groups),
        size=min(2000, len(unique_groups)),
        replace=False,
    )
    selected_groups = unique_groups.iloc[selected_indices]

    valid_sample = valid.merge(selected_groups, on=GROUPS, how="inner")
    valid_sample = valid_sample.reset_index(drop=True)

    print(f"KNN training candidates: {len(train_sample):,}")
    print(f"Validation GPS observations: {len(selected_groups):,}")
    print(f"Validation candidate rows: {len(valid_sample):,}")

    X_train = train_sample[FEATURES].astype("float32")
    y_train = train_sample["label"].astype(int)
    X_valid = valid_sample[FEATURES].astype("float32")

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    print("Training KNN...")
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        algorithm="brute",
        n_jobs=-1,
    )
    knn.fit(X_train_scaled, y_train)

    print("Evaluating KNN...")
    knn_probabilities = knn.predict_proba(X_valid_scaled)[:, 1]
    knn_metrics = calculate_metrics(
        valid_sample, knn_probabilities, "KNN (k=7)"
    )

    print("Evaluating Logistic Regression on the same observations...")
    logistic = joblib.load(MODEL_PATH)
    logistic_probabilities = logistic.predict_proba(
        X_valid[FEATURES]
    )[:, 1]
    logistic_metrics = calculate_metrics(
        valid_sample, logistic_probabilities, "Logistic Regression"
    )

    report = {
        "experiment": "KNN vs Logistic Regression on shared validation sample",
        "training_candidates_for_knn": int(len(train_sample)),
        "validation_gps_observations": int(len(selected_groups)),
        "validation_candidate_rows": int(len(valid_sample)),
        "results": [logistic_metrics, knn_metrics],
    }

    joblib.dump(
        {"scaler": scaler, "model": knn, "features": FEATURES},
        OUTPUT_DIR / "knn.joblib",
    )
    (OUTPUT_DIR / "knn_comparison.json").write_text(
        json.dumps(report, indent=4)
    )

    print("\n========== SAME-SAMPLE COMPARISON ==========")
    for result in report["results"]:
        print(f"\n{result['model']}")
        for metric, value in result.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

    print("\nSaved KNN model and comparison report.")


if __name__ == "__main__":
    main()
