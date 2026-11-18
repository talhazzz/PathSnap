"""Validate candidate feature generation on evaluator inputs without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from ml.preprocessing.build_shared_dataset import (
    FEATURE_COLUMNS,
    load_table,
    point_to_segments,
    project_coordinates,
    track_context,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "artifacts/configs/shared_pipeline.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/reports/student_test_preprocessing_validation.json"
EXPECTED_INFERENCE_COLUMNS = [
    "trip_id",
    "point_idx",
    "candidate_arc_id",
    *FEATURE_COLUMNS,
]


def build_inference_features(
    trip_id: str,
    folder: Path,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate candidate rows without opening the reference route file."""
    # Deliberately read only track, nodes, and arcs. Never construct labels here.
    track = load_table(folder / f"{trip_id}.track", 3)
    nodes = load_table(folder / f"{trip_id}.nodes", 2)
    arcs = load_table(folder / f"{trip_id}.arcs", 2, dtype=np.int64)
    if not len(track):
        raise ValueError("Track contains no GPS observations")
    if not len(arcs):
        raise ValueError("Road graph contains no arcs; no candidate can be generated")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Track or node coordinates/time contain non-finite values")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Arc endpoint index is outside the .nodes table")

    origin_lon = float(np.median(track[:, 0]))
    origin_lat = float(np.median(track[:, 1]))
    track_xy = project_coordinates(track[:, :2], origin_lon, origin_lat)
    node_xy = project_coordinates(nodes, origin_lon, origin_lat)
    arc_starts = node_xy[arcs[:, 0]]
    arc_ends = node_xy[arcs[:, 1]]
    tree = STRtree(
        [LineString((start, end)) for start, end in zip(arc_starts, arc_ends)]
    )

    radius = float(settings["candidate_generation"]["search_radius_m"])
    max_candidates = int(settings["candidate_generation"]["max_candidates_per_point"])
    step_distance, speed, time_delta, gps_bearing, gps_bearing_valid = track_context(
        track_xy, track[:, 2]
    )

    records: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    fallback_points = 0
    outside_radius_candidates = 0
    for point_idx, point_xy in enumerate(track_xy):
        point = Point(float(point_xy[0]), float(point_xy[1]))
        nearby_indices = np.asarray(
            tree.query(
                box(
                    point_xy[0] - radius,
                    point_xy[1] - radius,
                    point_xy[0] + radius,
                    point_xy[1] + radius,
                )
            ),
            dtype=np.int64,
        )
        if nearby_indices.size:
            distances, fractions, lateral = point_to_segments(
                point_xy, arc_starts[nearby_indices], arc_ends[nearby_indices]
            )
            in_radius = distances <= radius
            nearby_indices = nearby_indices[in_radius]
            distances = distances[in_radius]
            fractions = fractions[in_radius]
            lateral = lateral[in_radius]

        if nearby_indices.size:
            order = np.lexsort((nearby_indices, distances))
            selected = order[:max_candidates]
            nearby_indices = nearby_indices[selected]
            distances = distances[selected]
            fractions = fractions[selected]
            lateral = lateral[selected]
        else:
            fallback_points += 1
            nearby_indices = np.asarray([int(tree.nearest(point))], dtype=np.int64)
            distances, fractions, lateral = point_to_segments(
                point_xy, arc_starts[nearby_indices], arc_ends[nearby_indices]
            )

        rank_order = np.lexsort((nearby_indices, distances))
        ranks = np.empty(len(nearby_indices), dtype=np.int32)
        ranks[rank_order] = np.arange(len(nearby_indices), dtype=np.int32)
        vectors = arc_ends[nearby_indices] - arc_starts[nearby_indices]
        arc_lengths = np.linalg.norm(vectors, axis=1)
        arc_bearings = np.arctan2(vectors[:, 1], vectors[:, 0])
        heading_delta = gps_bearing[point_idx] - arc_bearings
        if gps_bearing_valid[point_idx]:
            alignment_cos = np.cos(heading_delta)
            alignment_sin = np.sin(heading_delta)
        else:
            alignment_cos = np.zeros(len(vectors), dtype=float)
            alignment_sin = np.zeros(len(vectors), dtype=float)

        candidate_counts.append(len(nearby_indices))
        outside_radius_candidates += int(np.count_nonzero(distances > radius))
        for candidate_idx, distance, fraction, side, length, rank, cos_delta, sin_delta in zip(
            nearby_indices,
            distances,
            fractions,
            lateral,
            arc_lengths,
            ranks,
            alignment_cos,
            alignment_sin,
        ):
            records.append(
                {
                    "trip_id": trip_id,
                    "point_idx": int(point_idx),
                    "candidate_arc_id": int(candidate_idx),
                    "distance_m": float(distance),
                    "signed_lateral_offset_m": float(side),
                    "projection_fraction": float(fraction),
                    "distance_to_arc_start_m": float(fraction * length),
                    "distance_to_arc_end_m": float((1.0 - fraction) * length),
                    "arc_length_m": float(length),
                    "candidate_rank_distance": int(rank),
                    "candidate_count_at_point": int(len(nearby_indices)),
                    "is_nearest_candidate": int(rank == 0),
                    "is_within_search_radius": int(distance <= radius),
                    "gps_step_distance_m": float(step_distance[point_idx]),
                    "gps_speed_m_per_time_unit": float(speed[point_idx]),
                    "time_delta_units": float(time_delta[point_idx]),
                    "gps_bearing_valid": int(gps_bearing_valid[point_idx]),
                    "heading_alignment_cos": float(cos_delta),
                    "heading_alignment_sin": float(sin_delta),
                    "is_self_loop": int(length < 0.01),
                }
            )

    frame = pd.DataFrame.from_records(records, columns=EXPECTED_INFERENCE_COLUMNS)
    finite_mask = np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    nonfinite_cells = int(finite_mask.size - finite_mask.sum())
    handled_points = int(frame["point_idx"].nunique()) if not frame.empty else 0
    schema_matches = list(frame.columns[3:]) == FEATURE_COLUMNS
    warnings: list[str] = []
    if fallback_points:
        warnings.append(
            f"{fallback_points} observations had no candidate within {radius:g} m; "
            "the nearest graph arc was added and marked outside the search radius."
        )
    if nonfinite_cells:
        warnings.append(f"{nonfinite_cells} engineered feature values are non-finite.")
    if not schema_matches:
        warnings.append("Generated feature columns/order do not match FEATURE_COLUMNS.")
    if handled_points != len(track):
        warnings.append("One or more GPS observations produced no candidate rows.")

    stats = {
        "trip_id": trip_id,
        "status": "passed"
        if handled_points == len(track) and nonfinite_cells == 0 and schema_matches
        else "failed",
        "gps_observations": int(len(track)),
        "observations_processed": handled_points,
        "observations_without_candidates": int(len(track) - handled_points),
        "candidate_rows": int(len(frame)),
        "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
        "candidate_count_median": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
        "nearest_arc_fallback_observations": fallback_points,
        "outside_search_radius_candidate_rows": outside_radius_candidates,
        "feature_columns_match_training_schema": schema_matches,
        "feature_columns": list(frame.columns[3:]),
        "nonfinite_feature_values": nonfinite_cells,
        "warnings": warnings,
    }
    return frame, stats


def validate(config_path: Path, output_path: Path) -> dict[str, Any]:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    partition_path = ROOT / settings["data"]["root"] / "student_test"
    if not partition_path.is_dir():
        raise FileNotFoundError(f"Evaluator partition not found: {partition_path}")
    folders = sorted(path for path in partition_path.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"No evaluator trip directories found under {partition_path}")

    trip_reports: list[dict[str, Any]] = []
    for index, folder in enumerate(folders, start=1):
        try:
            _, report = build_inference_features(folder.name, folder, settings)
        except Exception as error:  # Keep checking all trips and report each failure.
            report = {
                "trip_id": folder.name,
                "status": "failed",
                "gps_observations": None,
                "observations_processed": 0,
                "observations_without_candidates": None,
                "candidate_rows": 0,
                "nearest_arc_fallback_observations": None,
                "feature_columns_match_training_schema": False,
                "nonfinite_feature_values": None,
                "warnings": [],
                "error": f"{type(error).__name__}: {error}",
            }
        trip_reports.append(report)
        print(
            f"[{index:02d}/{len(folders):02d}] trip={folder.name} "
            f"status={report['status']} "
            f"observations={report.get('observations_processed', 0)}/"
            f"{report.get('gps_observations')} candidates={report.get('candidate_rows', 0):,} "
            f"fallbacks={report.get('nearest_arc_fallback_observations')}"
        )

    passed = sum(report["status"] == "passed" for report in trip_reports)
    total_observations = sum(
        int(report["gps_observations"] or 0) for report in trip_reports
    )
    total_processed = sum(
        int(report["observations_processed"] or 0) for report in trip_reports
    )
    result = {
        "mode": "inference_only",
        "source_partition": "student_test",
        "trip_count": len(trip_reports),
        "passed_trip_count": passed,
        "failed_trip_count": len(trip_reports) - passed,
        "total_gps_observations": total_observations,
        "total_observations_processed": total_processed,
        "total_candidate_rows": sum(int(report["candidate_rows"]) for report in trip_reports),
        "total_nearest_arc_fallback_observations": sum(
            int(report.get("nearest_arc_fallback_observations") or 0)
            for report in trip_reports
        ),
        "training_feature_columns": FEATURE_COLUMNS,
        "training_feature_column_count": len(FEATURE_COLUMNS),
        "inference_feature_columns_match_training_schema": all(
            report.get("feature_columns_match_training_schema", False)
            for report in trip_reports
        ),
        "student_test_route_files_opened": False,
        "labels_created": False,
        "training_or_tuning_performed": False,
        "existing_training_dataset_modified": False,
        "output_candidate_features_saved": False,
        "config": settings,
        "trips": trip_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    validate(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()