"""Generate publication-quality evaluation plots and interactive Folium HTML maps
from genuine PathSnap experiment outputs (all 5 baseline models + Optuna-tuned HGB).
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
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
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
                fontsize=9.5,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_all_plots() -> list[str]:
    created: list[str] = []

    # 1. Confusion matrices from genuine held-out 16-trip validation splits
    dt_rf_path = RESULT_DIR / "model_comparison_mithun.json"
    extra_path = RESULT_DIR / "extra_models_metrics.json"
    hgb_tun_path = RESULT_DIR / "hgb_tuning_summary.json"
    all_st_path = RESULT_DIR / "all_models_student_test_summary.json"
    hgb_trials_path = RESULT_DIR / "hgb_optuna_trials.csv"

    val_f1_map: dict[str, float] = {}

    if dt_rf_path.exists():
        d = json.loads(dt_rf_path.read_text(encoding="utf-8"))
        for k, label in [("decision_tree", "Decision Tree"), ("random_forest", "Random Forest")]:
            if k in d.get("models", {}):
                vm = d["models"][k]["validation_split_metrics"]
                val_f1_map[k] = vm["f1_macro"]
                p = PLOTS_DIR / f"confusion_matrix_{k}.png"
                _plot_confusion_matrix(vm["confusion_matrix"], f"Validation Confusion Matrix — {label}", p)
                created.append(str(p))

    if extra_path.exists():
        d = json.loads(extra_path.read_text(encoding="utf-8"))
        for k, label in [("hist_gradient_boosting", "HistGradientBoosting"), ("knn", "KNN")]:
            if k in d.get("models", {}):
                vm = d["models"][k]["validation_split_metrics"]
                val_f1_map[k] = vm["f1_macro"]
                p = PLOTS_DIR / f"confusion_matrix_{k}.png"
                _plot_confusion_matrix(vm["confusion_matrix"], f"Validation Confusion Matrix — {label}", p)
                created.append(str(p))

    if hgb_tun_path.exists():
        tun = json.loads(hgb_tun_path.read_text(encoding="utf-8"))
        tm = tun["validation_comparison_16_trips"]["tuned"]
        val_f1_map["hist_gradient_boosting_tuned"] = tm["macro_f1"]
        p = PLOTS_DIR / "confusion_matrix_hist_gradient_boosting_tuned.png"
        _plot_confusion_matrix(
            tm["confusion_matrix"],
            "Validation Confusion Matrix — HistGradientBoosting (Optuna Tuned)",
            p,
        )
        created.append(str(p))

    # 2. All-Model Comparison Charts (from 10-trip student_test evaluation)
    if all_st_path.exists():
        all_data = json.loads(all_st_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(all_data["comparison_table"])
        short_names = {
            "hist_gradient_boosting_tuned": "HGB\n(Optuna Tuned)",
            "hist_gradient_boosting": "HGB\n(Baseline)",
            "random_forest": "Random\nForest",
            "decision_tree": "Decision\nTree",
            "knn": "KNN",
            "linear_regression": "Linear Reg.\n(Scorer)",
        }
        df["short"] = df["model"].map(lambda k: short_names.get(k, k))
        x = np.arange(len(df))
        w = 0.36

        # Chart A: Student-Test Top-1 Accuracy & Route Continuity (Raw/Clf-Only vs HMM/Viterbi)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        for ax, c1, c2, ylabel, title in [
            (axes[0], "clf_top1_accuracy", "hmm_top1_accuracy", "Observation Top-1 Accuracy", "Top-1 Accuracy: Raw Model vs HMM/Viterbi (10 Student-Test Trips)"),
            (axes[1], "clf_route_continuity", "hmm_route_continuity", "Directed Route Continuity", "Route Continuity: Raw Model vs HMM/Viterbi (10 Student-Test Trips)"),
        ]:
            b1 = ax.bar(x - w / 2, df[c1], w, label="Model Only (No HMM)", color="#f59e0b")
            b2 = ax.bar(x + w / 2, df[c2], w, label="Model + HMM/Viterbi", color="#2563eb")
            ax.set_xticks(x, df["short"], fontsize=8.5)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8.5)
            for bars in (b1, b2):
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f"{h*100:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
        fig.tight_layout()
        p_hmm = PLOTS_DIR / "all_models_hmm_vs_raw_comparison.png"
        fig.savefig(p_hmm, dpi=180)
        plt.close(fig)
        created.append(str(p_hmm))

        # Chart B: Macro F1 Comparison (16-Trip Validation vs 10-Trip Student-Test)
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        val_vals = [val_f1_map.get(m, np.nan) for m in df["model"]]
        b1 = ax.bar(x - w / 2, [0 if np.isnan(v) else v for v in val_vals], w, label="16-Trip Held-Out Validation Macro F1", color="#6366f1")
        b2 = ax.bar(x + w / 2, df["student_test_macro_f1"], w, label="10-Trip Student-Test Macro F1", color="#10b981")
        ax.set_xticks(x, df["short"], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Macro F1-score")
        ax.set_title("All Models — Validation vs Student-Test Macro F1-score", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()
        for idx_b, bar in enumerate(b1):
            v = val_vals[idx_b]
            lbl = "N/A" if np.isnan(v) else f"{v:.3f}"
            ax.annotate(lbl, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        for bar in b2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        fig.tight_layout()
        p_f1 = PLOTS_DIR / "all_models_macro_f1_comparison.png"
        fig.savefig(p_f1, dpi=180)
        plt.close(fig)
        created.append(str(p_f1))

        # Chart C: Probabilistic Log Loss Comparison (excluding Linear Regression = N/A)
        prob_df = df.dropna(subset=["student_test_log_loss"]).reset_index(drop=True)
        if not prob_df.empty:
            fig, ax = plt.subplots(figsize=(8.2, 4.4))
            px = np.arange(len(prob_df))
            bars = ax.bar(px, prob_df["student_test_log_loss"], width=0.52, color=["#7c3aed", "#2563eb", "#10b981", "#f59e0b", "#0ea5e9"][: len(prob_df)])
            ax.set_xticks(px, prob_df["short"], fontsize=9)
            ax.set_ylabel("Log Loss (Lower is Better)")
            ax.set_title("Student-Test Log Loss (Probabilistic Models; Linear Regression = N/A)", fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
            fig.tight_layout()
            p_ll = PLOTS_DIR / "all_models_log_loss_comparison.png"
            fig.savefig(p_ll, dpi=180)
            plt.close(fig)
            created.append(str(p_ll))

    # 3. Optuna TPE Bayesian Optimization History (HistGradientBoosting)
    if hgb_trials_path.exists():
        tdf = pd.read_csv(hgb_trials_path)
        tdf["best_so_far"] = tdf["val_macro_f1"].cummax()
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        ax.plot(tdf["trial_number"] + 1, tdf["val_macro_f1"], "o-", color="#64748b", alpha=0.8, label="Trial Validation Macro F1")
        ax.plot(tdf["trial_number"] + 1, tdf["best_so_far"], "s-", color="#2563eb", linewidth=2.2, label="Best So Far (TPE)")
        if hgb_tun_path.exists():
            tun = json.loads(hgb_tun_path.read_text(encoding="utf-8"))
            base_f1 = tun["validation_comparison_16_trips"]["baseline"]["macro_f1"]
            ax.axhline(base_f1, color="#dc2626", linestyle="--", linewidth=1.4, label=f"Untuned HGB Baseline ({base_f1:.4f})")
        ax.set_xlabel("Optuna TPE Trial Number")
        ax.set_ylabel("16-Trip Held-Out Validation Macro F1")
        ax.set_title("Bayesian Hyperparameter Optimization History — HistGradientBoosting", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        p_opt = PLOTS_DIR / "optuna_hgb_optimization_history.png"
        fig.savefig(p_opt, dpi=180)
        plt.close(fig)
        created.append(str(p_opt))

    return created


def generate_sample_folium_maps() -> list[str]:
    best_model = (
        "hist_gradient_boosting_tuned"
        if (MODEL_DIR / "hist_gradient_boosting_tuned.joblib").exists()
        else "hist_gradient_boosting"
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
    print("Generated Plots:", len(plots))
    for p in plots:
        print(" ", p)
    print("Generated Folium Maps:", len(maps))
    for m in maps:
        print(" ", m)


if __name__ == "__main__":
    main()
