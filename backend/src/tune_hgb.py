"""Bayesian Hyperparameter Optimization (Optuna TPE) for HistGradientBoostingClassifier.

Strict leakage-prevention protocol:
- Uses ONLY the 80 training trips (canonical 17-feature shared dataset).
- Optimizes held-out trip-level validation Macro F1-score (64 train trips -> 16 validation trips;
  no GPS points from the same trip ever cross splits).
- Never reads or touches student_test or evaluator partitions during optimization.
- Records every trial to backend/experiments/hgb_optuna_trials.csv.
- Refits the winning configuration on all 80 training trips, saves
  backend/models/hist_gradient_boosting_tuned.joblib, and evaluates baseline vs tuned
  on both the held-out validation split and all 10 student_test trips.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from src.train_extra_models import HGB_PARAMS, _balanced_weights
from src.train_models import (
    MODEL_DIR,
    RESULT_DIR,
    evaluate_on_student_test,
    evaluate_predictions,
    load_model,
    load_shared_dataset,
    resolve_shared_dataset_path,
)
from src.trip_pipeline import FEATURE_COLUMNS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hgb_bayesian_tuning(n_trials: int = 10, seed: int = 42) -> dict[str, Any]:
    dataset_path = resolve_shared_dataset_path()
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

    w_train_bal = _balanced_weights(y_train)
    w_all_bal = _balanced_weights(y_all)

    # 1. Baseline HistGradientBoosting on exact same validation split
    print("Evaluating baseline HistGradientBoosting on held-out 16-trip validation split...", flush=True)
    base_val_clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    t_b0 = time.perf_counter()
    base_val_clf.fit(X_train, y_train, sample_weight=w_train_bal)
    base_fit_s = time.perf_counter() - t_b0
    base_val_probs = base_val_clf.predict_proba(X_val)[:, 1]
    base_val_metrics = evaluate_predictions(val_frame, y_val, base_val_probs)
    print(
        f"  [Baseline Val] Macro F1={base_val_metrics['f1_macro']:.4f} | "
        f"Top-1={base_val_metrics['top1_accuracy']:.4f} | "
        f"LogLoss={base_val_metrics['log_loss']:.4f} | time={base_fit_s:.1f}s",
        flush=True,
    )

    # 2. Optuna TPE Bayesian Search
    trial_records: list[dict[str, Any]] = []
    search_start = time.perf_counter()
    print(
        f"\nStarting Optuna TPE Bayesian Optimization ({n_trials} trials, trip-grouped 64/16 train/val split)...",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        balance_classes = trial.suggest_categorical("balance_classes", [True, False])
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 95, step=16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 100, step=5),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
            "random_state": seed,
        }

        t0 = time.perf_counter()
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=w_train_bal if balance_classes else None)
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(np.int8)
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        elapsed = time.perf_counter() - t0

        val_m = evaluate_predictions(val_frame, y_val, probs)
        rec = {
            "trial_number": int(trial.number),
            "balance_classes": bool(balance_classes),
            "max_iter": int(params["max_iter"]),
            "learning_rate": round(float(params["learning_rate"]), 5),
            "max_leaf_nodes": int(params["max_leaf_nodes"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "l2_regularization": round(float(params["l2_regularization"]), 5),
            "max_bins": int(params["max_bins"]),
            "val_macro_f1": round(macro_f1, 5),
            "val_top1_accuracy": round(float(val_m["top1_accuracy"]), 5),
            "val_log_loss": round(float(val_m["log_loss"]), 5),
            "duration_seconds": round(elapsed, 2),
        }
        trial_records.append(rec)
        print(
            f"  Trial {trial.number + 1:02d}/{n_trials:02d} | Val Macro F1={macro_f1:.4f} | "
            f"Top-1={val_m['top1_accuracy']:.4f} | LogLoss={val_m['log_loss']:.4f} | "
            f"bal={balance_classes} lr={params['learning_rate']:.4f} leaves={params['max_leaf_nodes']} iter={params['max_iter']} ({elapsed:.1f}s)",
            flush=True,
        )
        return macro_f1

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_hgb_bayesian_opt",
    )
    study.optimize(objective, n_trials=n_trials)
    total_search_s = time.perf_counter() - search_start

    best_raw = dict(study.best_params)
    best_balance = bool(best_raw.pop("balance_classes"))
    best_model_params = {**best_raw, "random_state": seed}

    trials_df = pd.DataFrame(trial_records)
    trials_csv = RESULT_DIR / "hgb_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    # 3. Re-evaluate best config on held-out validation split
    tuned_val_clf = HistGradientBoostingClassifier(**best_model_params)
    t_tv = time.perf_counter()
    tuned_val_clf.fit(X_train, y_train, sample_weight=w_train_bal if best_balance else None)
    tuned_split_s = time.perf_counter() - t_tv
    tuned_val_probs = tuned_val_clf.predict_proba(X_val)[:, 1]
    tuned_val_metrics = evaluate_predictions(val_frame, y_val, tuned_val_probs)

    # 4. Refit best tuned model on all 80 training trips (zero student_test leakage)
    tuned_final = HistGradientBoostingClassifier(**best_model_params)
    t_tf = time.perf_counter()
    tuned_final.fit(X_all, y_all, sample_weight=w_all_bal if best_balance else None)
    tuned_full_s = time.perf_counter() - t_tf

    tuned_artifact = {
        "model": tuned_final,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": "hist_gradient_boosting_tuned",
        "display_name": "HistGradientBoosting (Optuna Tuned)",
        "model_type": "classifier",
        "algorithm_family": "Boosting (Optuna TPE Tuned)",
        "hyperparameters": {**best_model_params, "balance_classes": best_balance},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_trips": int(data["trip_id"].nunique()),
        "training_rows": int(len(data)),
    }
    tuned_path = MODEL_DIR / "hist_gradient_boosting_tuned.joblib"
    joblib.dump(tuned_artifact, tuned_path, compress=3)
    print(
        f"\nSaved tuned model -> {tuned_path} ({tuned_path.stat().st_size / (1024*1024):.2f} MB) in {tuned_full_s:.1f}s",
        flush=True,
    )

    # 5. Evaluate baseline vs tuned on all 10 student_test trips
    print("Evaluating baseline vs tuned HistGradientBoosting on all 10 student_test trips...", flush=True)
    base_artifact = load_model("hist_gradient_boosting")
    base_st = evaluate_on_student_test(base_artifact, use_hmm=True)
    tuned_st = evaluate_on_student_test(tuned_artifact, use_hmm=True)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_optimized": "HistGradientBoostingClassifier",
        "optimizer": "Optuna TPESampler (Tree-structured Parzen Estimator)",
        "objective": "Maximize trip-level held-out validation Macro F1-score (64 train / 16 val trips; zero student_test/evaluator leakage)",
        "random_seed": seed,
        "n_trials": n_trials,
        "total_search_time_seconds": round(total_search_s, 2),
        "search_space": {
            "balance_classes": "categorical [True, False]",
            "max_iter": "int [100, 300], step=50",
            "learning_rate": "log-uniform [0.03, 0.20]",
            "max_leaf_nodes": "int [15, 95], step=16",
            "min_samples_leaf": "int [15, 100], step=5",
            "l2_regularization": "log-uniform [1e-3, 10.0]",
            "max_bins": "categorical [128, 255]",
        },
        "best_trial_number": int(study.best_trial.number),
        "baseline_hyperparameters": {**HGB_PARAMS, "balance_classes": True},
        "best_hyperparameters": {**best_model_params, "balance_classes": best_balance},
        "validation_comparison_16_trips": {
            "baseline": {
                "macro_f1": round(base_val_metrics["f1_macro"], 4),
                "top1_accuracy": round(base_val_metrics["top1_accuracy"], 4),
                "log_loss": round(base_val_metrics["log_loss"], 4),
                "roc_auc": round(base_val_metrics["roc_auc"], 4),
                "train_time_s": round(base_fit_s, 2),
                "confusion_matrix": base_val_metrics["confusion_matrix"],
            },
            "tuned": {
                "macro_f1": round(tuned_val_metrics["f1_macro"], 4),
                "top1_accuracy": round(tuned_val_metrics["top1_accuracy"], 4),
                "log_loss": round(tuned_val_metrics["log_loss"], 4),
                "roc_auc": round(tuned_val_metrics["roc_auc"], 4),
                "train_time_s": round(tuned_split_s, 2),
                "confusion_matrix": tuned_val_metrics["confusion_matrix"],
            },
            "delta_tuned_minus_baseline": {
                "macro_f1": round(tuned_val_metrics["f1_macro"] - base_val_metrics["f1_macro"], 4),
                "top1_accuracy": round(tuned_val_metrics["top1_accuracy"] - base_val_metrics["top1_accuracy"], 4),
                "log_loss": round(tuned_val_metrics["log_loss"] - base_val_metrics["log_loss"], 4),
            },
        },
        "student_test_comparison_10_trips": {
            "baseline": {
                "macro_f1": round(base_st["student_test_macro_f1"], 4),
                "log_loss": round(base_st["student_test_log_loss"], 4),
                "clf_top1_accuracy": round(base_st["classifier_only_top1_accuracy"], 4),
                "hmm_top1_accuracy": round(base_st["hmm_viterbi_top1_accuracy"], 4),
                "clf_route_continuity": round(base_st["classifier_only_route_continuity"], 4),
                "hmm_route_continuity": round(base_st["hmm_viterbi_route_continuity"], 4),
                "hmm_route_f1": round(base_st["hmm_viterbi_route_f1"], 4),
                "mean_clf_inference_time_s": round(base_st["mean_clf_inference_time_s"], 4),
                "mean_total_trip_time_s": round(base_st["mean_total_trip_time_s"], 4),
            },
            "tuned": {
                "macro_f1": round(tuned_st["student_test_macro_f1"], 4),
                "log_loss": round(tuned_st["student_test_log_loss"], 4),
                "clf_top1_accuracy": round(tuned_st["classifier_only_top1_accuracy"], 4),
                "hmm_top1_accuracy": round(tuned_st["hmm_viterbi_top1_accuracy"], 4),
                "clf_route_continuity": round(tuned_st["classifier_only_route_continuity"], 4),
                "hmm_route_continuity": round(tuned_st["hmm_viterbi_route_continuity"], 4),
                "hmm_route_f1": round(tuned_st["hmm_viterbi_route_f1"], 4),
                "mean_clf_inference_time_s": round(tuned_st["mean_clf_inference_time_s"], 4),
                "mean_total_trip_time_s": round(tuned_st["mean_total_trip_time_s"], 4),
            },
            "delta_tuned_minus_baseline": {
                "macro_f1": round(tuned_st["student_test_macro_f1"] - base_st["student_test_macro_f1"], 4),
                "log_loss": round(tuned_st["student_test_log_loss"] - base_st["student_test_log_loss"], 4),
                "clf_top1_accuracy": round(tuned_st["classifier_only_top1_accuracy"] - base_st["classifier_only_top1_accuracy"], 4),
                "hmm_top1_accuracy": round(tuned_st["hmm_viterbi_top1_accuracy"] - base_st["hmm_viterbi_top1_accuracy"], 4),
            },
        },
        "tuned_student_test_per_trip": tuned_st["per_trip"],
        "full_80trip_train_time_s": round(tuned_full_s, 2),
        "model_size_mb": round(tuned_path.stat().st_size / (1024 * 1024), 2),
    }

    out_json = RESULT_DIR / "hgb_tuning_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved Bayesian tuning summary -> {out_json}", flush=True)
    print(
        f"RESULT | Val Macro F1: {base_val_metrics['f1_macro']:.4f} -> {tuned_val_metrics['f1_macro']:.4f} "
        f"({tuned_val_metrics['f1_macro'] - base_val_metrics['f1_macro']:+.4f}) | "
        f"Student-Test Macro F1: {base_st['student_test_macro_f1']:.4f} -> {tuned_st['student_test_macro_f1']:.4f} | "
        f"Student-Test HMM Top-1: {base_st['hmm_viterbi_top1_accuracy']:.4f} -> {tuned_st['hmm_viterbi_top1_accuracy']:.4f} | "
        f"Student-Test LogLoss: {base_st['student_test_log_loss']:.4f} -> {tuned_st['student_test_log_loss']:.4f}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run_hgb_bayesian_tuning(n_trials=10, seed=42)
