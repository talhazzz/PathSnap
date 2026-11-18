"""End-to-end backend test suite for PathSnap (data loader, trained models, HMM/Viterbi, FastAPI)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.src.data_loader import load_partition
from backend.src.train_models import MODEL_DIR, load_model
from backend.src.trip_pipeline import FEATURE_COLUMNS


client = TestClient(app)


def test_data_loader_partitions():
    train_trips = load_partition("train")
    test_trips = load_partition("student_test")
    assert len(train_trips) == 80
    assert len(test_trips) == 10


def test_health_and_root():
    r_root = client.get("/")
    assert r_root.status_code == 200
    body = r_root.json()
    assert body["project"] == "PathSnap"
    assert body["status"] == "running"

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


def test_models_endpoint_and_artifacts():
    r = client.get("/models")
    assert r.status_code == 200
    models = {m["name"]: m["available"] for m in r.json()["models"]}
    assert models["decision_tree"] is True
    assert models["random_forest"] is True
    for name in ("decision_tree", "random_forest"):
        art = load_model(name)
        assert art["feature_columns"] == FEATURE_COLUMNS


def test_trips_listing_no_ground_truth_leakage():
    r = client.get("/trips?partition=student_test")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 10
    for item in data["trips"]:
        assert "route" not in item
        assert "arc_id" not in item
        assert item["gps_points"] > 0


def test_batch_predict_and_schema_validation():
    sample_row = {col: 0.0 for col in FEATURE_COLUMNS}
    sample_row["distance_m"] = 3.5
    sample_row["is_nearest_candidate"] = 1.0
    sample_row["is_within_search_radius"] = 1.0

    r_ok = client.post(
        "/predict",
        json={"model_name": "random_forest", "features": [sample_row]},
    )
    assert r_ok.status_code == 200
    body = r_ok.json()
    assert body["count"] == 1
    assert len(body["probabilities"]) == 1

    r_bad = client.post(
        "/predict",
        json={"model_name": "random_forest", "features": [{"wrong_col": 1.0}]},
    )
    assert r_bad.status_code == 422


def test_end_to_end_trip_prediction_and_map():
    r_trips = client.get("/trips?partition=student_test")
    first_trip = r_trips.json()["trips"][0]["trip_id"]

    r_pred = client.post(
        f"/predict/trip/{first_trip}",
        json={"model_name": "random_forest", "use_hmm": True, "partition": "student_test"},
    )
    assert r_pred.status_code == 200
    payload = r_pred.json()
    assert payload["trip_id"] == first_trip
    assert payload["predicted_arc_count"] > 0
    assert payload["route_continuity"] >= 0.85
    assert "server_evaluation" in payload
    assert payload["server_evaluation"]["hmm_viterbi"]["top1_accuracy"] > 0.5
    # Verify raw ground-truth arcs are never leaked in response
    assert "reference_arc_ids" not in payload
    assert "route" not in payload

    r_map = client.get(f"/trips/{first_trip}/map?partition=student_test&model_name=random_forest")
    assert r_map.status_code == 200
    assert "leaflet" in r_map.text.lower()
