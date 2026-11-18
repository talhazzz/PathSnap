"""Comprehensive 10-trip evaluation across all 5 baseline models + Optuna-tuned HistGradientBoosting.

Strictly uses the 10 held-out student_test trips ONLY for evaluation (zero leakage).
Produces:
- backend/experiments/all_models_student_test_summary.json
- backend/experiments/all_models_comparison.csv
- backend/experiments/student_test_10trips_final_model.csv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train_models import (
    MODEL_REGISTRY,
    RESULT_DIR,
    list_available_models,
    load_model,
)
from .trip_pipeline import DEFAULT_DATA_ROOT, run_trip_inference


def evaluate_all_models(partition: str = "student_test") -> dict[str, Any]:
    part_dir = DEFAULT_DATA_ROOT / partition
    trip_ids = sorted(p.name for p in part_dir.iterdir() if p.is_dir())
    available_models = [m["name"] for m in list_available_models() if m["available"]]

    print(
        f"Evaluating {len(available_models)} models across {len(trip_ids)} '{partition}' trips: {available_models}",
        flush=True,
    )

    summary_rows: list[dict[str, Any]] = []
    per_model_details: dict[str, Any] = {}

    for model_name in available_models:
        art = load_model(model_name)
        meta = MODEL_REGISTRY.get(model_name, {})
        per_trip_rows: list[dict[str, Any]] = []
        print(f"\n--- Model: {model_name} ({art.get('model_type', 'classifier')}) ---", flush=True)

        for tid in trip_ids:
            try:
                res = run_trip_inference(
                    trip_id=tid,
                    model_artifact=art,
                    partition_dir=part_dir,
                    use_hmm=True,
                    evaluate_server_side=True,
                )
                ev = res["server_evaluation"] or {}
                row = {
                    "trip_id": tid,
                    "model": model_name,
                    "model_type": res["model_type"],
                    "status": "OK",
                    "hmm_status": "OK",
                    "gps_points": res["gps_points_count"],
                    "candidates": res["candidate_rows_count"],
                    "predicted_arcs": res["predicted_arc_count"],
                    "candidate_miss_rate": ev.get("candidate_miss_rate"),
                    "candidate_macro_f1": ev.get("candidate_macro_f1"),
                    "candidate_log_loss": ev.get("candidate_log_loss"),
                    "clf_top1_accuracy": ev.get("classifier_only", {}).get("top1_accuracy"),
                    "hmm_top1_accuracy": ev.get("hmm_viterbi", {}).get("top1_accuracy"),
                    "clf_route_continuity": res["classifier_only_route_continuity"],
                    "hmm_route_continuity": res["route_continuity"],
                    "clf_route_f1": ev.get("classifier_only", {}).get("route_f1"),
                    "hmm_route_f1": ev.get("hmm_viterbi", {}).get("route_f1"),
                    "hmm_route_jaccard": ev.get("hmm_viterbi", {}).get("route_jaccard"),
                    "ml_inference_s": res["timings_seconds"]["classifier_inference"],
                    "hmm_time_s": res["timings_seconds"]["hmm_viterbi_decoding"],
                    "total_time_s": res["timings_seconds"]["total"],
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "trip_id": tid,
                    "model": model_name,
                    "model_type": art.get("model_type", "classifier"),
                    "status": "ERROR",
                    "hmm_status": "ERROR",
                    "gps_points": 0,
                    "candidates": 0,
                    "predicted_arcs": 0,
                    "candidate_miss_rate": None,
                    "candidate_macro_f1": None,
                    "candidate_log_loss": None,
                    "clf_top1_accuracy": None,
                    "hmm_top1_accuracy": None,
                    "clf_route_continuity": None,
                    "hmm_route_continuity": None,
                    "clf_route_f1": None,
                    "hmm_route_f1": None,
                    "hmm_route_jaccard": None,
                    "ml_inference_s": None,
                    "hmm_time_s": None,
                    "total_time_s": None,
                    "error": str(exc),
                }
            per_trip_rows.append(row)
            print(
                f"  trip={tid} status={row['status']} gps={row['gps_points']} cands={row['candidates']} "
                f"arcs={row['predicted_arcs']} clf_top1={row['clf_top1_accuracy']} hmm_top1={row['hmm_top1_accuracy']} "
                f"cont={row['hmm_route_continuity']} t={row['total_time_s']}s",
                flush=True,
            )

        tdf = pd.DataFrame(per_trip_rows)
        ok_df = tdf.loc[tdf["status"] == "OK"]
        ll_vals = ok_df["candidate_log_loss"].dropna()
        mean_ll = round(float(ll_vals.mean()), 4) if len(ll_vals) > 0 else None

        agg = {
            "model": model_name,
            "display_name": meta.get("display_name", model_name),
            "algorithm_family": meta.get("algorithm_family", "Unknown"),
            "model_type": art.get("model_type", "classifier"),
            "trips_passed": int(len(ok_df)),
            "trips_total": int(len(trip_ids)),
            "student_test_macro_f1": round(float(ok_df["candidate_macro_f1"].mean()), 4),
            "student_test_log_loss": mean_ll,
            "clf_top1_accuracy": round(float(ok_df["clf_top1_accuracy"].mean()), 4),
            "hmm_top1_accuracy": round(float(ok_df["hmm_top1_accuracy"].mean()), 4),
            "clf_route_continuity": round(float(ok_df["clf_route_continuity"].mean()), 4),
            "hmm_route_continuity": round(float(ok_df["hmm_route_continuity"].mean()), 4),
            "clf_route_f1": round(float(ok_df["clf_route_f1"].mean()), 4),
            "hmm_route_f1": round(float(ok_df["hmm_route_f1"].mean()), 4),
            "hmm_route_jaccard": round(float(ok_df["hmm_route_jaccard"].mean()), 4),
            "candidate_miss_rate": round(float(ok_df["candidate_miss_rate"].mean()), 4),
            "mean_ml_inference_s": round(float(ok_df["ml_inference_s"].mean()), 4),
            "mean_total_trip_s": round(float(ok_df["total_time_s"].mean()), 4),
        }
        summary_rows.append(agg)
        per_model_details[model_name] = {
            "aggregate": agg,
            "per_trip": per_trip_rows,
        }

    # Save consolidated CSV & JSON
    comp_df = pd.DataFrame(summary_rows)
    comp_csv = RESULT_DIR / "all_models_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    # Save per-trip table for the best/final model
    final_model = (
        "hist_gradient_boosting_tuned"
        if "hist_gradient_boosting_tuned" in per_model_details
        else available_models[0]
    )
    final_trip_df = pd.DataFrame(per_model_details[final_model]["per_trip"])
    final_trip_csv = RESULT_DIR / "student_test_10trips_final_model.csv"
    final_trip_df.to_csv(final_trip_csv, index=False)

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": partition,
        "trip_count": len(trip_ids),
        "final_selected_model": final_model,
        "models_evaluated": available_models,
        "comparison_table": summary_rows,
        "details": per_model_details,
    }
    out_json = RESULT_DIR / "all_models_student_test_summary.json"
    out_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"\nSaved:\n  {comp_csv}\n  {final_trip_csv}\n  {out_json}", flush=True)
    print("\nCONSOLIDATED 10-TRIP STUDENT-TEST COMPARISON:")
    print(comp_df.to_string(index=False), flush=True)
    return out_payload


if __name__ == "__main__":
    evaluate_all_models("student_test")
