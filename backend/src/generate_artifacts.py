"""Generate publication-quality evaluation plots and interactive Folium HTML maps
from genuine PathSnap experiment outputs for Deliverables.pdf (Sections 3.3, 4, 5, 6).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.src.train_models import MODEL_DIR, RESULT_DIR, load_model
from backend.src.trip_pipeline import (
    DEFAULT_DATA_ROOT,
    generate_folium_map,
    load_trip_arrays,
    run_trip_inference,
)

PLOTS_DIR = RESULT_DIR / "plots"
MAPS_DIR = RESULT_DIR / "maps"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred: Non-Route (0)", "Pred: Route (1)"])
    ax.set_yticks([0, 1], labels=["True: Non-Route (0)", "True: Route (1)"])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    thresh = arr.max() / 2.0
    total = arr.sum()
    for i in range(2):
        for j in range(2):
            pct = 100.0 * arr[i, j] / total if total else 0.0
            ax.text(
                j,
                i,
                f"{arr[i, j]:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "#0f172a",
                fontsize=10,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_all_plots() -> list[str]:
    """Generate real charts from model_comparison_mithun.json and rf_tuning_summary.json."""
    created: list[str] = []
    comp_path = RESULT_DIR / "model_comparison_mithun.json"
    tuning_path = RESULT_DIR / "rf_tuning_summary.json"
    trials_path = RESULT_DIR / "rf_optuna_trials.csv"

    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}; run train_models.py first.")

    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    models_data = comp["models"]

    rows = []
    for key, label in [
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest\n(Untuned)"),
    ]:
        if key not in models_data:
            continue
        m = models_data[key]
        rows.append(
            {
                "label": label,
                "cv_macro_f1": m["cv_5fold_group"]["cv_macro_f1_mean"],
                "cv_macro_f1_std": m["cv_5fold_group"]["cv_macro_f1_std"],
                "st_macro_f1": m["student_test_metrics"]["student_test_macro_f1"],
                "st_top1_clf": m["student_test_metrics"]["classifier_only_top1_accuracy"],
                "st_top1_hmm": m["student_test_metrics"]["hmm_viterbi_top1_accuracy"],
                "st_cont_clf": m["student_test_metrics"]["classifier_only_route_continuity"],
                "st_cont_hmm": m["student_test_metrics"]["hmm_viterbi_route_continuity"],
                "st_log_loss": m["student_test_metrics"]["student_test_log_loss"],
            }
        )
        cm_path = PLOTS_DIR / f"confusion_matrix_{key}.png"
        _plot_confusion_matrix(
            m["validation_split_metrics"]["confusion_matrix"],
            f"Validation Confusion Matrix — {label.replace(chr(10), ' ')}",
            cm_path,
        )
        created.append(str(cm_path))

    if tuning_path.exists():
        tun = json.loads(tuning_path.read_text(encoding="utf-8"))
        tm = tun["tuned_model"]
        rows.append(
            {
                "label": "Random Forest\n(Optuna Tuned)",
                "cv_macro_f1": tm["cv_5fold_group"]["cv_macro_f1_mean"],
                "cv_macro_f1_std": tm["cv_5fold_group"]["cv_macro_f1_std"],
                "st_macro_f1": tm["student_test_metrics"]["student_test_macro_f1"],
                "st_top1_clf": tm["student_test_metrics"]["classifier_only_top1_accuracy"],
                "st_top1_hmm": tm["student_test_metrics"]["hmm_viterbi_top1_accuracy"],
                "st_cont_clf": tm["student_test_metrics"]["classifier_only_route_continuity"],
                "st_cont_hmm": tm["student_test_metrics"]["hmm_viterbi_route_continuity"],
                "st_log_loss": tm["student_test_metrics"]["student_test_log_loss"],
            }
        )
        cm_path = PLOTS_DIR / "confusion_matrix_random_forest_tuned.png"
        _plot_confusion_matrix(
            tm["validation_split_metrics"]["confusion_matrix"],
            "Validation Confusion Matrix — Random Forest (Optuna Tuned)",
            cm_path,
        )
        created.append(str(cm_path))

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    colors = ["#3b82f6", "#10b981", "#8b5cf6"][: len(df)]

    # 1. CV Macro F1 & Student-Test Macro F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.34
    b1 = ax.bar(
        x - w / 2,
        df["cv_macro_f1"],
        w,
        yerr=df["cv_macro_f1_std"],
        capsize=4,
        label="5-Fold GroupKFold CV Macro F1",
        color="#2563eb",
    )
    b2 = ax.bar(
        x + w / 2,
        df["st_macro_f1"],
        w,
        label="Student-Test (10 Trips) Macro F1",
        color="#16a34a",
    )
    ax.set_xticks(x, df["label"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Model Comparison — GroupKFold CV vs Student-Test Macro F1", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    p1 = PLOTS_DIR / "model_comparison_macro_f1.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    created.append(str(p1))

    # 2. Student-Test Log Loss Comparison
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(x, df["st_log_loss"], width=0.5, color=colors)
    ax.set_xticks(x, df["label"])
    ax.set_ylabel("Log Loss (Lower is Better)")
    ax.set_title("Student-Test (10 Trips) Log Loss Comparison", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    p2 = PLOTS_DIR / "model_comparison_log_loss.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    created.append(str(p2))

    # 3. Classifier-Only vs Classifier + HMM/Viterbi Comparison (Top-1 Acc & Route Continuity)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col_clf, col_hmm, ylabel, title in [
        (axes[0], "st_top1_clf", "st_top1_hmm", "Observation-Level Top-1 Accuracy", "Top-1 Accuracy: Classifier vs HMM/Viterbi"),
        (axes[1], "st_cont_clf", "st_cont_hmm", "Directed Route Continuity", "Route Continuity: Classifier vs HMM/Viterbi"),
    ]:
        b1 = ax.bar(x - w / 2, df[col_clf], w, label="Classifier Only", color="#f59e0b")
        b2 = ax.bar(x + w / 2, df[col_hmm], w, label="Classifier + HMM/Viterbi", color="#dc2626")
        ax.set_xticks(x, df["label"])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
    fig.tight_layout()
    p3 = PLOTS_DIR / "hmm_vs_classifier_comparison.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    created.append(str(p3))

    # 4. Optuna Bayesian Optimization Trial History
    if trials_path.exists():
        tdf = pd.read_csv(trials_path)
        tdf["best_so_far"] = tdf["cv_macro_f1_mean"].cummax()
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ax.plot(tdf["trial_number"] + 1, tdf["cv_macro_f1_mean"], "o-", color="#64748b", alpha=0.75, label="Trial CV Macro F1")
        ax.plot(tdf["trial_number"] + 1, tdf["best_so_far"], "s-", color="#2563eb", linewidth=2.2, label="Best So Far")
        ax.set_xlabel("Optuna TPE Trial Number")
        ax.set_ylabel("GroupKFold CV Macro F1-score")
        ax.set_title("Bayesian Hyperparameter Optimization History (Random Forest)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p4 = PLOTS_DIR / "optuna_rf_optimization_history.png"
        fig.savefig(p4, dpi=180)
        plt.close(fig)
        created.append(str(p4))

    return created


def generate_sample_folium_maps() -> list[str]:
    """Generate interactive Folium HTML maps for representative student_test trips."""
    best_model = (
        "random_forest_tuned"
        if (MODEL_DIR / "random_forest_tuned.joblib").exists()
        else "random_forest"
    )
    artifact = load_model(best_model)
    st_dir = DEFAULT_DATA_ROOT / "student_test"
    trip_ids = sorted(p.name for p in st_dir.iterdir() if p.is_dir())[:3]
    created: list[str] = []

    for tid in trip_ids:
        res = run_trip_inference(
            trip_id=tid,
            model_artifact=artifact,
            partition_dir=st_dir,
            use_hmm=True,
            evaluate_server_side=True,
        )
        trip_with_ref = load_trip_arrays(tid, st_dir, include_route=True)
        internal = res["_internal"]
        out_html = MAPS_DIR / f"trip_{tid}_{best_model}_hmm_map.html"
        generate_folium_map(
            trip_id=tid,
            track=internal["track"],
            nodes=internal["nodes"],
            arcs=internal["arcs"],
            hmm_route_arcs=internal["hmm_route_arcs"],
            clf_point_arcs=internal["clf_point_arcs"],
            route_ref=trip_with_ref["route"],
            corridor_arc_ids=internal["corridor_arc_ids"],
            output_path=out_html,
        )
        created.append(str(out_html))
    return created


def main() -> None:
    plots = generate_all_plots()
    maps = generate_sample_folium_maps()
    print("Generated Plots:")
    for p in plots:
        print(" ", p)
    print("Generated Folium Maps:")
    for m in maps:
        print(" ", m)


if __name__ == "__main__":
    main()
