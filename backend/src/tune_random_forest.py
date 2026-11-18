"""Bayesian Hyperparameter Optimization (Optuna TPE) for Random Forest Classifier.

Strict leakage-prevention protocol:
- Uses ONLY the 80 training trips (backend/data/processed/candidate_features.csv).
- Objective optimizes trip-grouped GroupKFold Macro F1-score (never random row splits).
- Never reads or touches student_test or evaluator trips during search.
- After selecting the best hyperparameters, evaluates on the exact same 5-fold GroupKFold,
  validation split, and 10 student_test trips as the untuned model for a fair comparison.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from backend.src.train_models import (
    BACKEND_DIR,
    DEFAULT_RF_PARAMS,
    MODEL_DIR,
    RESULT_DIR,
    evaluate_on_student_test,
    evaluate_predictions,
    load_shared_dataset,
    resolve_shared_dataset_path,
    run_group_kfold_cv,
)
from backend.src.trip_pipeline import FEATURE_COLUMNS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_tuning(n_trials: int = 10, cv_folds_search: int = 3, seed: int = 42) -> dict[str, Any]:
    """Run Optuna TPE Bayesian optimization for Random Forest and compare untuned vs tuned."""
    dataset_path = resolve_shared_dataset_path()
    data = load_shared_dataset(dataset_path)

    train_frame = data.loc[data["split"] == "train"].reset_index(drop=True)
    val_frame = data.loc[data["split"] == "validation"].reset_index(drop=True)

    X_all = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_all = data["label"].to_numpy(dtype=np.int8)
    groups_all = data["trip_id"].to_numpy()

    X_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    X_val = val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_frame["label"].to_numpy(dtype=np.int8)

    gkf_search = GroupKFold(n_splits=cv_folds_search)
    search_splits = list(gkf_search.split(X_all, y_all, groups=groups_all))

    trial_records: list[dict[str, Any]] = []
    search_start = time.perf_counter()

    print(f"Starting Optuna TPE Bayesian Optimization ({n_trials} trials, {cv_folds_search}-fold GroupKFold on 80 train trips)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 140, step=20),
            "max_depth": trial.suggest_int("max_depth", 10, 20, step=2),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 12),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "max_samples": trial.suggest_float("max_samples", 0.15, 0.45, step=0.05),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
            "n_jobs": -1,
            "random_state": seed,
        }

        t0 = time.perf_counter()
        fold_f1s: list[float] = []
        for tr_idx, va_idx in search_splits:
            clf = RandomForestClassifier(**params)
            clf.fit(X_all[tr_idx], y_all[tr_idx])
            preds = clf.predict(X_all[va_idx])
            fold_f1s.append(float(f1_score(y_all[va_idx], preds, average="macro", zero_division=0)))

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        elapsed = time.perf_counter() - t0

        record = {
            "trial_number": trial.number,
            **{k: ("none" if v is None else v) for k, v in params.items() if k not in {"n_jobs", "random_state"}},
            "cv_macro_f1_mean": round(mean_f1, 5),
            "cv_macro_f1_std": round(std_f1, 5),
            "duration_seconds": round(elapsed, 2),
        }
        trial_records.append(record)
        print(
            f"  Trial {trial.number + 1:02d}/{n_trials:02d} | CV Macro F1 = {mean_f1:.4f} (+/- {std_f1:.4f}) | "
            f"time = {elapsed:.1f}s | params = { {k: params[k] for k in ['n_estimators','max_depth','min_samples_leaf','max_features','max_samples','class_weight']} }"
        )
        return mean_f1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_rf_bayes_opt")
    study.optimize(objective, n_trials=n_trials)
    total_search_time = time.perf_counter() - search_start

    best_params = {
        **study.best_params,
        "n_jobs": -1,
        "random_state": seed,
    }
    print(f"\nBayesian Optimization complete in {total_search_time:.1f}s!")
    print(f"Best 3-fold GroupKFold Macro F1: {study.best_value:.4f}")
    print(f"Best parameters: {json.dumps(best_params, indent=2)}")

    # Save trial history CSV
    trials_df = pd.DataFrame(trial_records)
    trials_csv_path = RESULT_DIR / "rf_optuna_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    # Full 5-fold GroupKFold evaluation of tuned RF (identical protocol to untuned RF)
    print("\nEvaluating tuned Random Forest with 5-fold GroupKFold CV...")
    tuned_cv = run_group_kfold_cv(
        data, lambda: RandomForestClassifier(**best_params), n_splits=5
    )

    # Held-out validation split evaluation (64 train -> 16 val)
    val_clf = RandomForestClassifier(**best_params)
    t_split = time.perf_counter()
    val_clf.fit(X_train, y_train)
    split_train_s = time.perf_counter() - t_split
    val_probs = val_clf.predict_proba(X_val)[:, 1]
    val_eval = evaluate_predictions(val_frame, y_val, val_probs)
    train_probs = val_clf.predict_proba(X_train)[:, 1]
    train_eval = evaluate_predictions(train_frame, y_train, train_probs)

    # Refit final tuned model on all 80 training trips
    final_clf = RandomForestClassifier(**best_params)
    t_full = time.perf_counter()
    final_clf.fit(X_all, y_all)
    full_train_s = time.perf_counter() - t_full

    tuned_artifact = {
        "model": final_clf,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "random_forest_tuned",
        "algorithm_family": "Ensemble (Bagging — Optuna TPE Tuned)",
        "hyperparameters": best_params,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_trips": int(data["trip_id"].nunique()),
        "training_rows": int(len(data)),
    }
    model_path = MODEL_DIR / "random_forest_tuned.joblib"
    joblib.dump(tuned_artifact, model_path, compress=3)
    print(f"Saved tuned model -> {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

    # Evaluate tuned model on the 10 student_test trips
    st_eval = evaluate_on_student_test(tuned_artifact, use_hmm=True)
    print(
        f"[Tuned RF | Student-Test 10 trips] Macro F1: {st_eval['student_test_macro_f1']:.4f} | "
        f"LogLoss: {st_eval['student_test_log_loss']:.4f} | "
        f"Clf Top-1: {st_eval['classifier_only_top1_accuracy']:.4f} | "
        f"HMM Top-1: {st_eval['hmm_viterbi_top1_accuracy']:.4f} (Cont: {st_eval['hmm_viterbi_route_continuity']:.4f})"
    )

    # Load untuned RF comparison if available
    untuned_summary = None
    comp_path = RESULT_DIR / "model_comparison_mithun.json"
    if comp_path.exists():
        comp_data = json.loads(comp_path.read_text(encoding="utf-8"))
        untuned_summary = comp_data.get("models", {}).get("random_forest")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "Optuna TPESampler (Bayesian Tree-structured Parzen Estimator)",
        "objective": f"{cv_folds_search}-fold GroupKFold Macro F1-score (grouped by trip_id, 80 train trips only)",
        "n_trials": n_trials,
        "total_search_time_seconds": round(total_search_time, 2),
        "search_space": {
            "n_estimators": "int [60, 140], step=20",
            "max_depth": "int [10, 20], step=2",
            "min_samples_split": "int [2, 20]",
            "min_samples_leaf": "int [2, 12]",
            "max_features": "categorical ['sqrt', 'log2', 0.5]",
            "max_samples": "float [0.15, 0.45], step=0.05",
            "class_weight": "categorical ['balanced', 'balanced_subsample', None]",
        },
        "best_trial_number": int(study.best_trial.number),
        "best_search_cv_macro_f1": round(float(study.best_value), 5),
        "best_hyperparameters": best_params,
        "untuned_hyperparameters": DEFAULT_RF_PARAMS,
        "tuned_model": {
            "cv_5fold_group": tuned_cv,
            "train_split_metrics": train_eval,
            "validation_split_metrics": val_eval,
            "split_train_time_s": round(split_train_s, 4),
            "full_80trip_train_time_s": round(full_train_s, 4),
            "student_test_metrics": st_eval,
            "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        },
        "untuned_vs_tuned_comparison": {
            "untuned": {
                "cv_macro_f1": round(untuned_summary["cv_5fold_group"]["cv_macro_f1_mean"], 4)
                if untuned_summary
                else None,
                "student_test_macro_f1": round(
                    untuned_summary["student_test_metrics"]["student_test_macro_f1"], 4
                )
                if untuned_summary
                else None,
                "student_test_top1_clf": round(
                    untuned_summary["student_test_metrics"]["classifier_only_top1_accuracy"], 4
                )
                if untuned_summary
                else None,
                "student_test_top1_hmm": round(
                    untuned_summary["student_test_metrics"]["hmm_viterbi_top1_accuracy"], 4
                )
                if untuned_summary
                else None,
                "student_test_log_loss": round(
                    untuned_summary["student_test_metrics"]["student_test_log_loss"], 4
                )
                if untuned_summary
                else None,
                "mean_clf_inference_time_s": round(
                    untuned_summary["student_test_metrics"]["mean_clf_inference_time_s"], 4
                )
                if untuned_summary
                else None,
            },
            "tuned": {
                "cv_macro_f1": round(tuned_cv["cv_macro_f1_mean"], 4),
                "student_test_macro_f1": round(st_eval["student_test_macro_f1"], 4),
                "student_test_top1_clf": round(st_eval["classifier_only_top1_accuracy"], 4),
                "student_test_top1_hmm": round(st_eval["hmm_viterbi_top1_accuracy"], 4),
                "student_test_log_loss": round(st_eval["student_test_log_loss"], 4),
                "mean_clf_inference_time_s": round(st_eval["mean_clf_inference_time_s"], 4),
            },
        },
    }

    out_path = RESULT_DIR / "rf_tuning_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tuning summary -> {out_path}")
    return summary


if __name__ == "__main__":
    run_bayesian_tuning()
