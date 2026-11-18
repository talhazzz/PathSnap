"""Decision Tree and Random Forest training, GroupKFold validation, student-test
evaluation, and model persistence module for PathSnap (Mithun — Backend & ML).

Consumes the canonical 17-feature shared candidate dataset generated from the
80 training trips (zero leakage from student_test or evaluator partitions).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from src.trip_pipeline import (
    DEFAULT_DATA_ROOT,
    FEATURE_COLUMNS,
    align_feature_frame,
    run_trip_inference,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"
RESULT_DIR = BACKEND_DIR / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DT_PARAMS: dict[str, Any] = {
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}


def resolve_shared_dataset_path() -> Path:
    """Locate the canonical shared candidate_features.csv generated from train/."""
    candidates = [
        REPO_ROOT / "artifacts" / "shared" / "candidate_features.csv",
        BACKEND_DIR / "data" / "processed" / "candidate_features.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Shared candidate_features.csv not found in artifacts/shared/ or backend/data/processed/."
    )


def load_shared_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the canonical candidate dataset with compact numeric dtypes."""
    dataset_path = path or resolve_shared_dataset_path()
    dtypes: dict[str, Any] = {
        "trip_id": "string",
        "point_idx": "int32",
        "candidate_arc_id": "int32",
        "split": "category",
        "label": "int8",
        **{col: "float32" for col in FEATURE_COLUMNS},
    }
    return pd.read_csv(dataset_path, dtype=dtypes)


def compute_top1_accuracy(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute observation-level Top-1 accuracy (highest-scoring candidate per GPS fix)."""
    ranked = frame[["trip_id", "point_idx", "candidate_arc_id", "label"]].copy()
    ranked["_prob"] = probabilities
    best = (
        ranked.sort_values(
            ["trip_id", "point_idx", "_prob", "candidate_arc_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["trip_id", "point_idx"])
    )
    per_trip = best.groupby("trip_id")["label"].mean()
    return {
        "point_count": int(len(best)),
        "top1_accuracy": float(best["label"].mean()),
        "macro_trip_top1_accuracy": float(per_trip.mean()),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive candidate-level and observation-level classification metrics."""
    probs = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    preds = (probs >= 0.5).astype(np.int8)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    top1 = compute_top1_accuracy(frame, probs)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "precision_pos": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "top1_accuracy": top1["top1_accuracy"],
        "macro_trip_top1_accuracy": top1["macro_trip_top1_accuracy"],
        "evaluated_points": top1["point_count"],
        "confusion_matrix": cm.tolist(),
    }


def run_group_kfold_cv(
    data: pd.DataFrame,
    model_builder,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run leak-free GroupKFold cross-validation grouped strictly by trip_id."""
    gkf = GroupKFold(n_splits=n_splits)
    X_all = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_all = data["label"].to_numpy(dtype=np.int8)
    groups = data["trip_id"].to_numpy()

    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
        model = model_builder()
        t0 = time.perf_counter()
        model.fit(X_all[tr_idx], y_all[tr_idx])
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = model.predict_proba(X_all[va_idx])[:, 1]
        inf_s = time.perf_counter() - t1

        va_frame = data.iloc[va_idx]
        m = evaluate_predictions(va_frame, y_all[va_idx], probs)
        m["fold"] = fold_idx
        m["train_time_s"] = fit_s
        m["inference_time_s"] = inf_s
        fold_metrics.append(m)

    summary = {
        "n_splits": n_splits,
        "cv_macro_f1_mean": float(np.mean([f["f1_macro"] for f in fold_metrics])),
        "cv_macro_f1_std": float(np.std([f["f1_macro"] for f in fold_metrics])),
        "cv_top1_accuracy_mean": float(np.mean([f["top1_accuracy"] for f in fold_metrics])),
        "cv_top1_accuracy_std": float(np.std([f["top1_accuracy"] for f in fold_metrics])),
        "cv_log_loss_mean": float(np.mean([f["log_loss"] for f in fold_metrics])),
        "cv_log_loss_std": float(np.std([f["log_loss"] for f in fold_metrics])),
        "cv_precision_macro_mean": float(np.mean([f["precision_macro"] for f in fold_metrics])),
        "cv_recall_macro_mean": float(np.mean([f["recall_macro"] for f in fold_metrics])),
        "cv_roc_auc_mean": float(np.mean([f["roc_auc"] for f in fold_metrics])),
        "mean_train_time_s": float(np.mean([f["train_time_s"] for f in fold_metrics])),
        "mean_inference_time_s": float(np.mean([f["inference_time_s"] for f in fold_metrics])),
        "folds": fold_metrics,
    }
    return summary


def evaluate_on_student_test(
    model_artifact: dict[str, Any],
    student_test_dir: Path | None = None,
    use_hmm: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained model on the 10 held-out student_test trips (evaluation only)."""
    part_dir = student_test_dir or (DEFAULT_DATA_ROOT / "student_test")
    trip_ids = sorted(p.name for p in part_dir.iterdir() if p.is_dir())
    per_trip: list[dict[str, Any]] = []

    for tid in trip_ids:
        res = run_trip_inference(
            trip_id=tid,
            model_artifact=model_artifact,
            partition_dir=part_dir,
            use_hmm=use_hmm,
            evaluate_server_side=True,
        )
        ev = res["server_evaluation"] or {}
        per_trip.append(
            {
                "trip_id": tid,
                "gps_points": res["gps_points_count"],
                "candidate_rows": res["candidate_rows_count"],
                "candidate_miss_rate": ev.get("candidate_miss_rate", 0.0),
                "candidate_macro_f1": ev.get("candidate_macro_f1", 0.0),
                "candidate_precision_macro": ev.get("candidate_precision_macro", 0.0),
                "candidate_recall_macro": ev.get("candidate_recall_macro", 0.0),
                "candidate_log_loss": ev.get("candidate_log_loss", 0.0),
                "clf_top1_accuracy": ev.get("classifier_only", {}).get("top1_accuracy", 0.0),
                "clf_route_continuity": ev.get("classifier_only", {}).get("route_continuity", 0.0),
                "clf_route_f1": ev.get("classifier_only", {}).get("route_f1", 0.0),
                "hmm_top1_accuracy": ev.get("hmm_viterbi", {}).get("top1_accuracy", 0.0),
                "hmm_route_continuity": ev.get("hmm_viterbi", {}).get("route_continuity", 0.0),
                "hmm_route_f1": ev.get("hmm_viterbi", {}).get("route_f1", 0.0),
                "hmm_route_jaccard": ev.get("hmm_viterbi", {}).get("route_jaccard", 0.0),
                "clf_inference_time_s": res["timings_seconds"]["classifier_inference"],
                "hmm_time_s": res["timings_seconds"]["hmm_viterbi_decoding"],
                "total_time_s": res["timings_seconds"]["total"],
            }
        )

    df = pd.DataFrame(per_trip)
    return {
        "trip_count": len(per_trip),
        "student_test_macro_f1": float(df["candidate_macro_f1"].mean()),
        "student_test_precision_macro": float(df["candidate_precision_macro"].mean()),
        "student_test_recall_macro": float(df["candidate_recall_macro"].mean()),
        "student_test_log_loss": float(df["candidate_log_loss"].mean()),
        "student_test_candidate_miss_rate": float(df["candidate_miss_rate"].mean()),
        "classifier_only_top1_accuracy": float(df["clf_top1_accuracy"].mean()),
        "classifier_only_route_continuity": float(df["clf_route_continuity"].mean()),
        "classifier_only_route_f1": float(df["clf_route_f1"].mean()),
        "hmm_viterbi_top1_accuracy": float(df["hmm_top1_accuracy"].mean()),
        "hmm_viterbi_route_continuity": float(df["hmm_route_continuity"].mean()),
        "hmm_viterbi_route_f1": float(df["hmm_route_f1"].mean()),
        "hmm_viterbi_route_jaccard": float(df["hmm_route_jaccard"].mean()),
        "mean_clf_inference_time_s": float(df["clf_inference_time_s"].mean()),
        "total_clf_inference_time_s": float(df["clf_inference_time_s"].sum()),
        "mean_total_trip_time_s": float(df["total_time_s"].mean()),
        "per_trip": per_trip,
    }


def train_models(X, y, random_state=42):
    """Backward-compatible helper to train Decision Tree and Random Forest on a DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths")
    if len(X) == 0:
        raise ValueError("Training dataset is empty")
    if X.isnull().any().any():
        raise ValueError("X contains missing feature values")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("All model features must be numeric")
    if len(pd.Series(y).unique()) < 2:
        raise ValueError("Training requires at least two classes")

    feature_columns = list(X.columns)
    dt_params = {**DEFAULT_DT_PARAMS, "random_state": random_state}
    rf_params = {**DEFAULT_RF_PARAMS, "random_state": random_state}
    models = {
        "decision_tree": DecisionTreeClassifier(**dt_params),
        "random_forest": RandomForestClassifier(**rf_params),
    }

    results = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X, y)
        training_time = time.perf_counter() - start
        predictions = model.predict(X)
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y, predictions),
            "precision_macro": precision_score(y, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
            "training_time_seconds": training_time,
        }
        artifact = {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": name,
        }
        joblib.dump(artifact, MODEL_DIR / f"{name}.joblib", compress=3)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_DIR / "decision_tree_random_forest_training.csv", index=False)
    return results_df


SHARED_MODEL_DIR = REPO_ROOT / "artifacts" / "models"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "hist_gradient_boosting_bayesian": {
        "display_name": "HistGradientBoosting (Bayesian Tuned)",
        "model_type": "classifier",
        "algorithm_family": "Boosting (Bayesian Optimized — ml-dev)",
    },
    "hist_gradient_boosting": {
        "display_name": "HistGradientBoosting (Baseline)",
        "model_type": "classifier",
        "algorithm_family": "Boosting (Baseline — ml-dev)",
    },
    "hist_gradient_boosting_tuned": {
        "display_name": "HistGradientBoosting (Optuna Tuned)",
        "model_type": "classifier",
        "algorithm_family": "Boosting (Optuna TPE Tuned)",
    },
    "random_forest": {
        "display_name": "Random Forest",
        "model_type": "classifier",
        "algorithm_family": "Ensemble (Bagging)",
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "model_type": "classifier",
        "algorithm_family": "Tree-based",
    },
    "knn": {
        "display_name": "K-Nearest Neighbors (KNN)",
        "model_type": "classifier",
        "algorithm_family": "Instance-based",
    },
    "linear_regression": {
        "display_name": "Linear Regression (Candidate Scorer)",
        "model_type": "regression_scorer",
        "algorithm_family": "Linear (OLS Candidate Ranking)",
    },
}


def _resolve_model_file(safe_name: str) -> Path | None:
    """Resolve a .joblib artifact, prioritizing shared artifacts/models/ (ml-dev) then backend/models/."""
    for directory in (SHARED_MODEL_DIR, MODEL_DIR):
        candidate = directory / f"{safe_name}.joblib"
        if candidate.exists():
            return candidate
    return None


def list_available_models() -> list[dict[str, Any]]:
    """List all registered models plus any additional discovered .joblib models."""
    discovered_set: set[str] = set()
    for directory in (SHARED_MODEL_DIR, MODEL_DIR):
        if directory.is_dir():
            discovered_set.update(p.stem for p in directory.glob("*.joblib"))

    ordered_names = list(MODEL_REGISTRY.keys())
    for extra in sorted(discovered_set):
        if extra not in ordered_names:
            ordered_names.append(extra)

    out = []
    for name in ordered_names:
        path = _resolve_model_file(name)
        meta = MODEL_REGISTRY.get(
            name,
            {
                "display_name": name.replace("_", " ").title(),
                "model_type": "classifier",
                "algorithm_family": "Custom",
            },
        )
        rel_path = None
        if path is not None:
            try:
                rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            except ValueError:
                rel_path = str(path)
        out.append(
            {
                "name": name,
                "display_name": meta["display_name"],
                "model_type": meta["model_type"],
                "algorithm_family": meta["algorithm_family"],
                "available": path is not None,
                "artifact_path": rel_path,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path is not None else None,
            }
        )
    return out


def load_model(model_name: str) -> dict[str, Any]:
    """Load a saved model artifact and its expected feature columns."""
    safe_name = Path(model_name).name
    if safe_name != model_name or not safe_name:
        raise ValueError(f"Invalid model name: {model_name}")
    path = _resolve_model_file(safe_name)
    if path is None:
        raise FileNotFoundError(
            f"Model '{safe_name}' not found in {SHARED_MODEL_DIR} or {MODEL_DIR}"
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        artifact = {"model": artifact, "feature_columns": list(FEATURE_COLUMNS)}
    if "model_name" not in artifact:
        artifact["model_name"] = safe_name
    artifact["artifact_path"] = str(path)
    reg_meta = MODEL_REGISTRY.get(safe_name, {})
    if "model_type" not in artifact:
        artifact["model_type"] = (
            reg_meta.get("model_type")
            or ("classifier" if hasattr(artifact["model"], "predict_proba") else "regression_scorer")
        )
    if "display_name" not in artifact:
        artifact["display_name"] = reg_meta.get("display_name", safe_name.replace("_", " ").title())
    return artifact


def score_candidates(model_name: str, features: pd.DataFrame) -> dict[str, Any]:
    """Return candidate scores/probabilities respecting classifier vs regression_scorer semantics."""
    artifact = load_model(model_name)
    model = artifact["model"]
    columns = artifact["feature_columns"]
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    aligned = align_feature_frame(features, columns)
    X = aligned.to_numpy(dtype=np.float32)
    is_regression = (
        artifact.get("model_type") == "regression_scorer"
        or not hasattr(model, "predict_proba")
    )
    if is_regression:
        scores = model.predict(X).astype(np.float64)
        return {
            "model_type": "regression_scorer",
            "scores": scores,
            "predictions": None,
            "probabilities": None,
        }
    probs = model.predict_proba(X)[:, 1].astype(np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "model_type": "classifier",
        "scores": probs,
        "predictions": preds,
        "probabilities": probs,
    }


def predict(model_name: str, features: pd.DataFrame) -> np.ndarray:
    """Generate class predictions using a saved model with strict feature validation."""
    artifact = load_model(model_name)
    model = artifact["model"]
    columns = artifact["feature_columns"]
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    aligned = align_feature_frame(features, columns)
    return model.predict(aligned.to_numpy(dtype=np.float32))


def predict_proba(model_name: str, features: pd.DataFrame) -> np.ndarray:
    """Generate positive-class probability predictions with strict feature validation."""
    artifact = load_model(model_name)
    model = artifact["model"]
    columns = artifact["feature_columns"]
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    aligned = align_feature_frame(features, columns)
    X = aligned.to_numpy(dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).astype(np.float64)


def run_full_training_pipeline() -> dict[str, Any]:
    """Train, cross-validate, and evaluate Decision Tree and Random Forest on genuine data."""
    dataset_path = resolve_shared_dataset_path()
    print(f"Loading canonical shared dataset from: {dataset_path}")
    data = load_shared_dataset(dataset_path)

    train_frame = data.loc[data["split"] == "train"].reset_index(drop=True)
    val_frame = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if set(train_frame["trip_id"]) & set(val_frame["trip_id"]):
        raise ValueError("Trip leakage detected between train and validation splits")

    print(
        f"Loaded {len(data):,} candidate rows across {data['trip_id'].nunique()} train trips "
        f"(train split: {len(train_frame):,} rows / {train_frame['trip_id'].nunique()} trips; "
        f"val split: {len(val_frame):,} rows / {val_frame['trip_id'].nunique()} trips)"
    )

    X_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    X_val = val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_frame["label"].to_numpy(dtype=np.int8)
    X_all = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_all = data["label"].to_numpy(dtype=np.int8)

    specs = [
        (
            "decision_tree",
            "Tree-based",
            DEFAULT_DT_PARAMS,
            lambda: DecisionTreeClassifier(**DEFAULT_DT_PARAMS),
        ),
        (
            "random_forest",
            "Ensemble (Bagging)",
            DEFAULT_RF_PARAMS,
            lambda: RandomForestClassifier(**DEFAULT_RF_PARAMS),
        ),
    ]

    full_report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "feature_columns": FEATURE_COLUMNS,
        "train_split_rows": int(len(train_frame)),
        "validation_split_rows": int(len(val_frame)),
        "total_training_trips": int(data["trip_id"].nunique()),
        "models": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for name, family, params, builder in specs:
        print(f"\n{'=' * 70}\nTRAINING & EVALUATING: {name.upper()} ({family})\n{'=' * 70}")

        # 1. 5-Fold GroupKFold CV across all 80 training trips (strictly grouped by trip_id)
        cv_summary = run_group_kfold_cv(data, builder, n_splits=5)
        print(
            f"  [5-Fold GroupKFold CV] Macro F1: {cv_summary['cv_macro_f1_mean']:.4f} +/- {cv_summary['cv_macro_f1_std']:.4f} | "
            f"Top-1 Acc: {cv_summary['cv_top1_accuracy_mean']:.4f} | LogLoss: {cv_summary['cv_log_loss_mean']:.4f}"
        )

        # 2. Held-out validation split (64 train trips -> 16 validation trips, identical to Medha's split)
        val_model = builder()
        t0 = time.perf_counter()
        val_model.fit(X_train, y_train)
        split_train_s = time.perf_counter() - t0

        train_probs = val_model.predict_proba(X_train)[:, 1]
        train_eval = evaluate_predictions(train_frame, y_train, train_probs)

        t1 = time.perf_counter()
        val_probs = val_model.predict_proba(X_val)[:, 1]
        val_inf_s = time.perf_counter() - t1
        val_eval = evaluate_predictions(val_frame, y_val, val_probs)
        print(
            f"  [Train Split]          Macro F1: {train_eval['f1_macro']:.4f} | Top-1 Acc: {train_eval['top1_accuracy']:.4f} | LogLoss: {train_eval['log_loss']:.4f}"
        )
        print(
            f"  [Validation Split]     Macro F1: {val_eval['f1_macro']:.4f} | Top-1 Acc: {val_eval['top1_accuracy']:.4f} | LogLoss: {val_eval['log_loss']:.4f}"
        )

        # 3. Refit final production artifact on all 80 training trips (zero student_test leakage)
        final_model = builder()
        t2 = time.perf_counter()
        final_model.fit(X_all, y_all)
        full_train_s = time.perf_counter() - t2

        artifact = {
            "model": final_model,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": name,
            "algorithm_family": family,
            "hyperparameters": params,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_trips": int(data["trip_id"].nunique()),
            "training_rows": int(len(data)),
        }
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(artifact, model_path, compress=3)
        print(f"  Saved artifact -> {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

        # 4. Evaluate on the 10 student_test trips (Classifier-only vs Classifier + HMM/Viterbi)
        st_eval = evaluate_on_student_test(artifact, use_hmm=True)
        print(
            f"  [Student-Test 10 trips] Macro F1: {st_eval['student_test_macro_f1']:.4f} | "
            f"LogLoss: {st_eval['student_test_log_loss']:.4f} | "
            f"Clf-Only Top-1: {st_eval['classifier_only_top1_accuracy']:.4f} (Cont: {st_eval['classifier_only_route_continuity']:.4f}) | "
            f"HMM Top-1: {st_eval['hmm_viterbi_top1_accuracy']:.4f} (Cont: {st_eval['hmm_viterbi_route_continuity']:.4f})"
        )

        full_report["models"][name] = {
            "algorithm_family": family,
            "hyperparameters": params,
            "cv_5fold_group": cv_summary,
            "train_split_metrics": train_eval,
            "validation_split_metrics": val_eval,
            "split_train_time_s": round(split_train_s, 4),
            "val_inference_time_s": round(val_inf_s, 4),
            "full_80trip_train_time_s": round(full_train_s, 4),
            "student_test_metrics": st_eval,
            "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        }

        csv_rows.append(
            {
                "model": name,
                "algorithm_family": family,
                "cv_macro_f1_mean": round(cv_summary["cv_macro_f1_mean"], 4),
                "cv_macro_f1_std": round(cv_summary["cv_macro_f1_std"], 4),
                "cv_top1_accuracy": round(cv_summary["cv_top1_accuracy_mean"], 4),
                "val_macro_f1": round(val_eval["f1_macro"], 4),
                "val_top1_accuracy": round(val_eval["top1_accuracy"], 4),
                "val_log_loss": round(val_eval["log_loss"], 4),
                "student_test_macro_f1": round(st_eval["student_test_macro_f1"], 4),
                "student_test_top1_clf": round(st_eval["classifier_only_top1_accuracy"], 4),
                "student_test_top1_hmm": round(st_eval["hmm_viterbi_top1_accuracy"], 4),
                "student_test_continuity_clf": round(st_eval["classifier_only_route_continuity"], 4),
                "student_test_continuity_hmm": round(st_eval["hmm_viterbi_route_continuity"], 4),
                "student_test_log_loss": round(st_eval["student_test_log_loss"], 4),
                "student_test_miss_rate": round(st_eval["student_test_candidate_miss_rate"], 4),
                "train_time_s": round(full_train_s, 4),
                "mean_clf_inference_time_s": round(st_eval["mean_clf_inference_time_s"], 4),
            }
        )

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(RESULT_DIR / "decision_tree_random_forest_training.csv", index=False)
    (RESULT_DIR / "model_comparison_mithun.json").write_text(
        json.dumps(full_report, indent=2), encoding="utf-8"
    )
    print("\nSaved metrics to backend/experiments/decision_tree_random_forest_training.csv and model_comparison_mithun.json")
    return full_report


if __name__ == "__main__":
    run_full_training_pipeline()
