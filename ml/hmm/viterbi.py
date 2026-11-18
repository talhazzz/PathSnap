"""Minimal Viterbi decoding over candidate directed road segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.preprocessing.build_shared_dataset import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "trip_id",
    "point_idx",
    "candidate_arc_id",
    "projection_fraction",
    "arc_length_m",
    "distance_to_arc_start_m",
    "distance_to_arc_end_m",
    "gps_step_distance_m",
}


def _transition_score(
    previous: pd.Series,
    current: pd.Series,
    arc_endpoints: np.ndarray,
    movement_sigma_m: float,
) -> float:
    """Score only same-arc or directly connected directed-arc transitions."""
    previous_arc = int(previous["candidate_arc_id"])
    current_arc = int(current["candidate_arc_id"])
    if (
        previous_arc < 0
        or current_arc < 0
        or previous_arc >= len(arc_endpoints)
        or current_arc >= len(arc_endpoints)
    ):
        raise ValueError("Candidate arc ID is outside the supplied road graph")

    if previous_arc == current_arc:
        movement_distance = abs(
            float(current["projection_fraction"])
            - float(previous["projection_fraction"])
        ) * float(previous["arc_length_m"])
    else:
        previous_to = int(arc_endpoints[previous_arc, 1])
        current_from = int(arc_endpoints[current_arc, 0])
        if previous_to != current_from:
            return -np.inf
        movement_distance = float(previous["distance_to_arc_end_m"]) + float(
            current["distance_to_arc_start_m"]
        )

    observed_distance = max(float(current["gps_step_distance_m"]), 0.0)
    sigma = max(float(movement_sigma_m), 0.35 * observed_distance + 10.0)
    residual = (movement_distance - observed_distance) / sigma
    return -0.5 * residual * residual


def _decode_segment(
    frames: list[pd.DataFrame],
    arc_endpoints: np.ndarray,
    probability_column: str,
    movement_sigma_m: float,
    segment_id: int,
) -> pd.DataFrame:
    emissions = [
        np.log(np.clip(frame[probability_column].to_numpy(dtype=float), 1e-12, 1.0))
        for frame in frames
    ]
    scores = emissions[0]
    backpointers: list[np.ndarray] = []

    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        transition = np.full((len(previous), len(current)), -np.inf, dtype=float)
        for previous_index in range(len(previous)):
            previous_row = previous.iloc[previous_index]
            for current_index in range(len(current)):
                transition[previous_index, current_index] = _transition_score(
                    previous_row,
                    current.iloc[current_index],
                    arc_endpoints,
                    movement_sigma_m,
                )

        joint_scores = scores[:, None] + transition
        pointers = np.argmax(joint_scores, axis=0)
        next_scores = joint_scores[pointers, np.arange(len(current))] + emissions[index]
        if not np.isfinite(next_scores).any():
            raise RuntimeError("Decoder segment contains no feasible state transition")
        backpointers.append(pointers)
        scores = next_scores

    state_indices = np.empty(len(frames), dtype=np.int64)
    state_indices[-1] = int(np.argmax(scores))
    for index in range(len(frames) - 1, 0, -1):
        state_indices[index - 1] = backpointers[index - 1][state_indices[index]]

    selected = [frames[index].iloc[state].copy() for index, state in enumerate(state_indices)]
    result = pd.DataFrame(selected).reset_index(drop=True)
    result["segment_id"] = segment_id
    result["reset_before_point"] = False
    result.loc[0, "reset_before_point"] = True
    return result


def decode_viterbi(
    candidates: pd.DataFrame,
    arc_endpoints: np.ndarray,
    probability_column: str = "candidate_probability",
    movement_sigma_m: float = 35.0,
) -> pd.DataFrame:
    """Decode one trip's candidate table with explicit connectivity resets.

    Candidate rows must contain one trip and one or more states per point. A
    transition is allowed only for the same directed arc or when the previous
    arc's destination node equals the next arc's source node. If observations
    are nonconsecutive or no candidate pair is connected, decoding starts a new
    segment at that point and marks `reset_before_point`; disconnected roads are
    never silently treated as connected.
    """
    missing = REQUIRED_COLUMNS.union({probability_column}).difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if candidates.empty:
        return candidates.copy()
    if candidates["trip_id"].nunique() != 1:
        raise ValueError("decode_viterbi expects candidate rows for exactly one trip")
    if arc_endpoints.ndim != 2 or arc_endpoints.shape[1] != 2:
        raise ValueError("arc_endpoints must be an N-by-2 directed endpoint array")
    if movement_sigma_m <= 0:
        raise ValueError("movement_sigma_m must be positive")

    ordered = candidates.sort_values(
        ["point_idx", "candidate_arc_id"], kind="stable"
    ).copy()
    probabilities = ordered[probability_column].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("Candidate emission probabilities must be finite and in [0, 1]")
    point_frames: list[pd.DataFrame] = []
    for _, frame in ordered.groupby("point_idx", sort=True, observed=True):
        if frame["candidate_arc_id"].duplicated().any():
            raise ValueError("Duplicate candidate arc IDs found at one GPS point")
        point_frames.append(frame.reset_index(drop=True))

    segments: list[list[pd.DataFrame]] = []
    current_segment = [point_frames[0]]
    reachable_states = np.ones(len(point_frames[0]), dtype=bool)
    for frame in point_frames[1:]:
        previous = current_segment[-1]
        consecutive_points = int(frame["point_idx"].iloc[0]) == int(
            previous["point_idx"].iloc[0]
        ) + 1
        next_reachable = np.zeros(len(frame), dtype=bool)
        if consecutive_points:
            for previous_index in np.flatnonzero(reachable_states):
                for current_index in range(len(frame)):
                    next_reachable[current_index] |= np.isfinite(
                        _transition_score(
                            previous.iloc[previous_index],
                            frame.iloc[current_index],
                            arc_endpoints,
                            movement_sigma_m,
                        )
                    )
        if consecutive_points and next_reachable.any():
            current_segment.append(frame)
            reachable_states = next_reachable
        else:
            segments.append(current_segment)
            current_segment = [frame]
            reachable_states = np.ones(len(frame), dtype=bool)
    segments.append(current_segment)

    decoded = [
        _decode_segment(
            frames,
            arc_endpoints,
            probability_column,
            movement_sigma_m,
            segment_id,
        )
        for segment_id, frames in enumerate(segments)
    ]
    return pd.concat(decoded, ignore_index=True)


def _read_trip_candidates(dataset_path: Path, trip_id: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = [
        "trip_id",
        "point_idx",
        "candidate_arc_id",
        "label",
        *FEATURE_COLUMNS,
    ]
    dtypes: dict[str, Any] = {
        "trip_id": "string",
        "point_idx": "int32",
        "candidate_arc_id": "int32",
        "label": "int8",
        **{column: "float32" for column in FEATURE_COLUMNS},
    }
    for chunk in pd.read_csv(dataset_path, usecols=columns, dtype=dtypes, chunksize=50_000):
        selected = chunk.loc[chunk["trip_id"] == trip_id]
        if not selected.empty:
            pieces.append(selected)
        if pieces and str(chunk["trip_id"].iloc[-1]) > trip_id:
            break
    if not pieces:
        raise ValueError(f"No candidate rows found for trip {trip_id}")
    return pd.concat(pieces, ignore_index=True)


def run_smoke_test(trip_id: str = "00000005") -> dict[str, Any]:
    """Exercise the saved baseline on one held-out trip from the train partition."""
    trip_folder = ROOT / "data/raw/Student folder/train" / trip_id
    if not trip_folder.is_dir():
        raise FileNotFoundError(f"Trip {trip_id} is not in the train partition")
    candidates = _read_trip_candidates(
        ROOT / "artifacts/shared/candidate_features.csv", trip_id
    )
    bundle = joblib.load(ROOT / "artifacts/models/hist_gradient_boosting.joblib")
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("Saved baseline model schema does not match shared features")
    probabilities = bundle["model"].predict_proba(
        candidates[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    )[:, 1]
    candidates["candidate_probability"] = probabilities
    classifier_only = candidates.sort_values(
        ["point_idx", "candidate_probability", "candidate_arc_id"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("point_idx")
    arc_endpoints = np.loadtxt(
        trip_folder / f"{trip_id}.arcs", dtype=np.int64, ndmin=2
    )
    decoded = decode_viterbi(candidates, arc_endpoints)
    reference_path = trip_folder / f"{trip_id}.route"
    route_arc_ids = np.loadtxt(reference_path, dtype=np.int64, ndmin=1)
    split_table = pd.read_csv(ROOT / "artifacts/shared/trip_split.csv", dtype={"trip_id": "string"})
    internal_split = str(
        split_table.loc[split_table["trip_id"] == trip_id, "split"].iloc[0]
    )
    same_segment = decoded["segment_id"].iloc[1:].to_numpy() == decoded[
        "segment_id"
    ].iloc[:-1].to_numpy()
    connected_transitions = 0
    connectivity_violations = 0
    arc_ids = decoded["candidate_arc_id"].to_numpy()
    for index, (previous_arc, current_arc) in enumerate(zip(arc_ids[:-1], arc_ids[1:])):
        if not same_segment[index]:
            continue
        if previous_arc == current_arc or arc_endpoints[int(previous_arc), 1] == arc_endpoints[int(current_arc), 0]:
            connected_transitions += 1
        else:
            connectivity_violations += 1
    if connectivity_violations:
        raise RuntimeError("Decoded consecutive states contain disconnected arcs")
    return {
        "trip_id": trip_id,
        "partition": "train",
        "internal_split": internal_split,
        "points_decoded": int(len(decoded)),
        "reference_route_arc_count": int(len(route_arc_ids)),
        "classifier_only_candidate_label_accuracy": float(classifier_only["label"].mean()),
        "decoded_candidate_label_accuracy": float(decoded["label"].mean()),
        "segments": int(decoded["segment_id"].nunique()),
        "connectivity_resets": max(int(decoded["reset_before_point"].sum()) - 1, 0),
        "connected_transitions": int(connected_transitions),
        "connectivity_violations": int(connectivity_violations),
        "same_segment_transitions": int(same_segment.sum()),
        "model_probabilities_used": True,
        "student_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-id", default="00000005")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/reports/hmm_viterbi_smoke_metrics.json",
    )
    args = parser.parse_args()
    metrics = run_smoke_test(args.trip_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()