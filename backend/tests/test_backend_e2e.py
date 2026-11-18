"""Comprehensive automated test suite for PathSnap backend (runs via unittest or pytest).

Covers:
1. Model registry & artifact loading for all 7 models (including Medha's ml-dev HGB baseline & Bayesian-tuned).
2. Canonical 17-feature schema parity with ml/preprocessing/build_shared_dataset.py and legacy alias alignment.
3. Edge cases: NaN GPS rows, invalid trip IDs (404), missing models (503), invalid feature columns (422), empty inputs (400).
4. Complete real-trip map-matching pipeline (GPS -> candidates -> 17 features -> ML -> HMM/Viterbi -> geometry).
5. FastAPI endpoints including optional Gemini AI Route Analyst graceful degradation & zero ground-truth leakage.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.api.main import (
    AIExplainTripRequest,
    PredictionRequest,
    TripPredictionRequest,
    ai_explain_trip,
    app,
    get_metrics_summary,
    get_models,
    get_trip_geometry,
    get_trip_map,
    health,
    list_trips,
    make_prediction,
    predict_trip,
    root,
)
from src.data_loader import load_partition
from src.train_models import list_available_models, load_model, score_candidates
from src.trip_pipeline import (
    DEFAULT_DATA_ROOT,
    FEATURE_COLUMNS,
    align_feature_frame,
    extract_trip_candidates,
    load_trip_arrays,
)

EXPECTED_MODELS = [
    "hist_gradient_boosting_bayesian",
    "hist_gradient_boosting",
    "hist_gradient_boosting_tuned",
    "random_forest",
    "decision_tree",
    "knn",
    "linear_regression",
]

FAKE_REQ = SimpleNamespace(client=SimpleNamespace(host="10.0.15.200"))


class TestPathSnapBackend(unittest.TestCase):
    def test_01_data_loader_partitions(self):
        train_trips = load_partition("train")
        test_trips = load_partition("student_test")
        self.assertEqual(len(train_trips), 80)
        self.assertEqual(len(test_trips), 10)

    def test_02_canonical_feature_schema_matches_medha(self):
        from ml.preprocessing.build_shared_dataset import FEATURE_COLUMNS as MEDHA_COLS

        self.assertEqual(len(FEATURE_COLUMNS), 17)
        self.assertEqual(FEATURE_COLUMNS, list(MEDHA_COLS))
        self.assertEqual(FEATURE_COLUMNS[11], "gps_speed_m_per_time_unit")
        self.assertEqual(FEATURE_COLUMNS[12], "time_delta_units")

        # Verify legacy alias bridging works bidirectionally without corrupting schema
        legacy_row = {col: 1.0 for col in FEATURE_COLUMNS}
        legacy_row["gps_speed_mps"] = legacy_row.pop("gps_speed_m_per_time_unit")
        legacy_row["time_delta_s"] = legacy_row.pop("time_delta_units")
        aligned = align_feature_frame(pd.DataFrame([legacy_row]), FEATURE_COLUMNS)
        self.assertEqual(list(aligned.columns), FEATURE_COLUMNS)

        # Strictly reject unknown columns
        with self.assertRaises(ValueError):
            align_feature_frame(pd.DataFrame([{"unknown_col": 1.0}]), FEATURE_COLUMNS)

    def test_03_nan_gps_row_handled_gracefully(self):
        trip = load_trip_arrays("00000052", DEFAULT_DATA_ROOT / "student_test", include_route=False)
        noisy_track = trip["track"].copy()
        noisy_track[0, 0] = np.nan
        cands, _ = extract_trip_candidates("00000052", noisy_track, trip["nodes"], trip["arcs"])
        self.assertFalse(cands.empty)
        self.assertTrue(np.isfinite(cands[FEATURE_COLUMNS].to_numpy()).all())

    def test_04_all_7_models_load_and_score(self):
        models = {m["name"]: m for m in list_available_models()}
        sample_row = {col: 0.0 for col in FEATURE_COLUMNS}
        sample_row["distance_m"] = 2.5
        sample_row["is_nearest_candidate"] = 1.0
        sample_row["is_within_search_radius"] = 1.0
        sample_df = pd.DataFrame([sample_row])

        for name in EXPECTED_MODELS:
            self.assertIn(name, models)
            self.assertTrue(models[name]["available"], f"Model {name} not available")
            art = load_model(name)
            self.assertEqual(len(art["feature_columns"]), 17)
            scored = score_candidates(name, sample_df)
            self.assertEqual(len(scored["scores"]), 1)
            if name == "linear_regression":
                self.assertEqual(scored["model_type"], "regression_scorer")
                self.assertIsNone(scored["predictions"])
            else:
                self.assertEqual(scored["model_type"], "classifier")
                self.assertEqual(len(scored["probabilities"]), 1)

        # Verify Medha's official artifacts are prioritized
        self.assertEqual(
            models["hist_gradient_boosting_bayesian"]["artifact_path"],
            "artifacts/models/hist_gradient_boosting_bayesian.joblib",
        )
        self.assertEqual(
            models["hist_gradient_boosting"]["artifact_path"],
            "artifacts/models/hist_gradient_boosting.joblib",
        )

    def test_05_cors_and_health_and_models_endpoints(self):
        cors_mw = [m for m in app.user_middleware if "CORS" in m.cls.__name__]
        self.assertEqual(len(cors_mw), 1)
        self.assertEqual(cors_mw[0].kwargs["allow_origins"], ["*"])

        r_root = root(FAKE_REQ)
        self.assertEqual(r_root["status"], "running")

        r_health = health(FAKE_REQ)
        self.assertEqual(r_health["status"], "healthy")
        self.assertEqual(r_health["default_model"], "hist_gradient_boosting_bayesian")
        self.assertIn("ai_analyst_ready", r_health)

        r_models = get_models(FAKE_REQ)
        self.assertEqual(r_models["default_model"], "hist_gradient_boosting_bayesian")
        self.assertGreaterEqual(len(r_models["models"]), 7)

    def test_06_trips_and_geometry_no_ground_truth_leakage(self):
        r_trips = list_trips(FAKE_REQ, partition="student_test")
        self.assertEqual(r_trips["count"], 10)
        for t in r_trips["trips"]:
            self.assertNotIn("route", t)
            self.assertNotIn("arc_id", t)

        r_geom = get_trip_geometry("00000052", FAKE_REQ, partition="student_test")
        self.assertEqual(r_geom["gps_points_count"], 262)
        self.assertNotIn("route", r_geom)

    def test_07_real_trip_prediction_hgb_baseline_and_bayesian(self):
        for model_name in ("hist_gradient_boosting_bayesian", "hist_gradient_boosting"):
            res = predict_trip(
                trip_id="00000052",
                request=FAKE_REQ,
                body=TripPredictionRequest(model_name=model_name, use_hmm=True, partition="student_test"),
                model_name=None,
                use_hmm=None,
                partition=None,
            )
            self.assertEqual(res["model_name"], model_name)
            self.assertGreater(res["predicted_arc_count"], 0)
            self.assertGreaterEqual(res["route_continuity"], 0.95)
            self.assertGreater(res["server_evaluation"]["hmm_viterbi"]["top1_accuracy"], 0.90)
            self.assertNotIn("route", res)
            self.assertNotIn("reference_arc_ids", res)
            # Ensure JSON serializability
            json.dumps(res)

    def test_08_error_handling_400_404_422_503(self):
        # Empty features -> 400
        with self.assertRaises(HTTPException) as ctx:
            make_prediction(PredictionRequest(model_name="hist_gradient_boosting", features=[]), FAKE_REQ)
        self.assertEqual(ctx.exception.status_code, 400)

        # Invalid feature schema -> 422
        with self.assertRaises(HTTPException) as ctx:
            make_prediction(
                PredictionRequest(model_name="hist_gradient_boosting", features=[{"bad": 1.0}]),
                FAKE_REQ,
            )
        self.assertEqual(ctx.exception.status_code, 422)

        # Missing model -> 503
        with self.assertRaises(HTTPException) as ctx:
            make_prediction(
                PredictionRequest(model_name="nonexistent_model_xyz", features=[{"a": 1.0}]),
                FAKE_REQ,
            )
        self.assertEqual(ctx.exception.status_code, 503)

        # Invalid trip ID -> 404
        with self.assertRaises(HTTPException) as ctx:
            predict_trip(
                trip_id="99999999",
                request=FAKE_REQ,
                body=TripPredictionRequest(model_name="hist_gradient_boosting_bayesian", partition="student_test"),
                model_name=None,
                use_hmm=None,
                partition=None,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    def test_09_folium_map_and_metrics_summary(self):
        m_resp = get_trip_map(
            trip_id="00000052",
            request=FAKE_REQ,
            model_name="hist_gradient_boosting_bayesian",
            use_hmm=True,
            partition="student_test",
        )
        self.assertEqual(m_resp.status_code, 200)
        self.assertIn("leaflet", m_resp.body.decode("utf-8").lower())

        summary = get_metrics_summary(FAKE_REQ)
        self.assertIsInstance(summary, dict)
        json.dumps(summary)

    def test_10_gemini_ai_analyst_graceful_fallback(self):
        # Verify graceful non-crashing response when GEMINI_API_KEY is unset
        prev_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            resp = ai_explain_trip(
                AIExplainTripRequest(
                    trip_id="00000052",
                    model_name="hist_gradient_boosting_bayesian",
                    use_hmm=True,
                    summary={
                        "trip_id": "00000052",
                        "model_name": "hist_gradient_boosting_bayesian",
                        "gps_points_count": 262,
                        "candidate_rows_count": 2771,
                        "predicted_arc_count": 93,
                        "route_continuity": 1.0,
                        "classifier_only_route_continuity": 0.79,
                    },
                ),
                FAKE_REQ,
            )
            self.assertEqual(resp["status"], "unavailable")
            self.assertIn("GEMINI_API_KEY", resp["message"])
            json.dumps(resp)
        finally:
            if prev_key is not None:
                os.environ["GEMINI_API_KEY"] = prev_key


if __name__ == "__main__":
    unittest.main(verbosity=2)
