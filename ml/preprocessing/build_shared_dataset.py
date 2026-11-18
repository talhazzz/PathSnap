"""Build the shared, route-labeled candidate dataset from training trips only."""

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
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "artifacts/configs/shared_pipeline.yaml"
IDENTIFIER_COLUMNS = ["trip_id", "point_idx", "candidate_arc_id", "split", "label"]
FEATURE_COLUMNS = [
    "distance_m",
    "signed_lateral_offset_m",
    "projection_fraction",
    "distance_to_arc_start_m",
    "distance_to_arc_end_m",
    "arc_length_m",
    "candidate_rank_distance",
    "candidate_count_at_point",
    "is_nearest_candidate",
    "is_within_search_radius",
    "gps_step_distance_m",
    "gps_speed_m_per_time_unit",
    "time_delta_units",
    "gps_bearing_valid",
    "heading_alignment_cos",
    "heading_alignment_sin",
    "is_self_loop",
]
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS


def load_table(path: Path, columns: int, dtype: Any = float) -> np.ndarray:
    """Load a whitespace-delimited source table and enforce its column count."""
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"{path} must have {columns} columns; got {values.shape}")
    return values


def project_coordinates(
    coordinates: np.ndarray, lon0: float, lat0: float
) -> np.ndarray:
    """Project a compact trip-local lon/lat area into approximate metres."""
    scale_x = 111_320.0 * np.cos(np.deg2rad(lat0))
    scale_y = 110_574.0
    return np.column_stack(
        ((coordinates[:, 0] - lon0) * scale_x, (coordinates[:, 1] - lat0) * scale_y)
    )


def point_to_segments(
    point: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact planar segment distances, projection fractions and offsets."""
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        lengths_sq,
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 0,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * vectors
    offsets = point - projections
    distances = np.linalg.norm(offsets, axis=1)
    signed_offsets = np.divide(
        vectors[:, 0] * offsets[:, 1] - vectors[:, 1] * offsets[:, 0],
        np.sqrt(lengths_sq),
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 0,
    )
    return distances, fractions, signed_offsets


def align_track_to_route(
    track_xy: np.ndarray,
    route_starts: np.ndarray,
    route_ends: np.ndarray,
    advance_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Monotonically align points to route arcs by dynamic programming.

    The route is an ordered sequence of arc-row IDs, not a row-wise pairing with
    GPS points. Each point may stay on or advance along that sequence. Advancing
    multiple route rows between adjacent GPS points has a small configurable
    penalty; emission cost is point-to-segment distance in metres.
    """
    point_count = len(track_xy)
    route_count = len(route_starts)
    if not point_count or not route_count:
        raise ValueError("Cannot align an empty track or route")

    route_indices = np.arange(route_count, dtype=np.float64)
    predecessor_dtype = (
        np.uint16 if route_count <= np.iinfo(np.uint16).max else np.uint32
    )
    predecessors = np.zeros((point_count, route_count), dtype=predecessor_dtype)
    dp, _, _ = point_to_segments(track_xy[0], route_starts, route_ends)

    for point_idx in range(1, point_count):
        adjusted = dp - advance_penalty * route_indices
        prefix_values = np.minimum.accumulate(adjusted)
        new_minima = np.r_[True, adjusted[1:] < prefix_values[:-1]]
        prefix_indices = np.maximum.accumulate(
            np.where(new_minima, np.arange(route_count), 0)
        )
        emission, _, _ = point_to_segments(
            track_xy[point_idx], route_starts, route_ends
        )
        dp = emission + advance_penalty * route_indices + prefix_values
        predecessors[point_idx] = prefix_indices.astype(predecessor_dtype)

    route_positions = np.empty(point_count, dtype=np.int32)
    route_positions[-1] = int(np.argmin(dp))
    for point_idx in range(point_count - 1, 0, -1):
        route_positions[point_idx - 1] = predecessors[
            point_idx, route_positions[point_idx]
        ]

    vectors = route_ends[route_positions] - route_starts[route_positions]
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", track_xy - route_starts[route_positions], vectors),
        lengths_sq,
        out=np.zeros(point_count, dtype=float),
        where=lengths_sq > 0,
    ).clip(0.0, 1.0)
    projections = route_starts[route_positions] + fractions[:, None] * vectors
    matched_distance = np.linalg.norm(track_xy - projections, axis=1)
    return route_positions, matched_distance


def make_split(trip_ids: list[str], fraction: float, seed: int) -> dict[str, str]:
    """Create a deterministic held-out split over trip IDs, never GPS rows."""
    if len(trip_ids) < 2:
        raise ValueError("At least two training trips are required for validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    indices = np.arange(len(trip_ids))
    train_idx, validation_idx = next(splitter.split(indices, groups=trip_ids))
    split = {trip_ids[i]: "train" for i in train_idx}
    split.update({trip_ids[i]: "validation" for i in validation_idx})
    return split


def track_context(track_xy: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Calculate per-observation motion context from adjacent GPS samples."""
    count = len(track_xy)
    step_distance = np.zeros(count, dtype=float)
    time_delta = np.zeros(count, dtype=float)
    speed = np.zeros(count, dtype=float)
    bearing = np.zeros(count, dtype=float)
    bearing_valid = np.zeros(count, dtype=bool)
    if count > 1:
        displacement = np.diff(track_xy, axis=0)
        step_distance[1:] = np.linalg.norm(displacement, axis=1)
        time_delta[1:] = np.maximum(np.diff(times), 0.0)
        valid = (time_delta[1:] > 0.0) & (step_distance[1:] > 0.1)
        speed[1:] = np.divide(
            step_distance[1:],
            time_delta[1:],
            out=np.zeros(count - 1, dtype=float),
            where=time_delta[1:] > 0.0,
        )
        bearing[1:] = np.arctan2(displacement[:, 1], displacement[:, 0])
        bearing_valid[1:] = valid
    return step_distance, speed, time_delta, bearing, bearing_valid


def trip_candidate_frame(
    trip_id: str,
    folder: Path,
    split: str,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate and label candidate arcs for one training trip."""
    track = load_table(folder / f"{trip_id}.track", 3)
    nodes = load_table(folder / f"{trip_id}.nodes", 2)
    arcs = load_table(folder / f"{trip_id}.arcs", 2, dtype=np.int64)
    route = np.loadtxt(folder / f"{trip_id}.route", dtype=np.int64, ndmin=1)
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"{trip_id}: arc endpoint is outside the .nodes table")
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"{trip_id}: route entry is outside the .arcs table")

    # .nodes IDs are zero-based row indices; .route values index .arcs rows.
    origin_lon = float(np.median(track[:, 0]))
    origin_lat = float(np.median(track[:, 1]))
    track_xy = project_coordinates(track[:, :2], origin_lon, origin_lat)
    node_xy = project_coordinates(nodes, origin_lon, origin_lat)
    arc_starts = node_xy[arcs[:, 0]]
    arc_ends = node_xy[arcs[:, 1]]
    route_starts = arc_starts[route]
    route_ends = arc_ends[route]
    route_positions, route_match_distance = align_track_to_route(
        track_xy,
        route_starts,
        route_ends,
        float(settings["labeling"]["route_advance_penalty"]),
    )
    route_arc_ids = route[route_positions]

    lines = [LineString((start, end)) for start, end in zip(arc_starts, arc_ends)]
    tree = STRtree(lines)
    radius = float(settings["candidate_generation"]["search_radius_m"])
    max_candidates = int(settings["candidate_generation"]["max_candidates_per_point"])
    max_label_distance = float(
        settings["candidate_generation"]["label_max_distance_m"]
    )
    step_distance, speed, time_delta, gps_bearing, gps_bearing_valid = track_context(
        track_xy, track[:, 2]
    )

    records: list[dict[str, Any]] = []
    retained_points = 0
    points_without_reference = 0
    candidate_counts: list[int] = []
    for point_idx, point_xy in enumerate(track_xy):
        if route_match_distance[point_idx] > max_label_distance:
            points_without_reference += 1
            continue

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
            nearby_indices = np.asarray([int(tree.nearest(point))], dtype=np.int64)
            distances, fractions, lateral = point_to_segments(
                point_xy, arc_starts[nearby_indices], arc_ends[nearby_indices]
            )

        reference_arc_id = int(route_arc_ids[point_idx])
        reference_locations = np.flatnonzero(nearby_indices == reference_arc_id)
        if reference_locations.size == 0:
            ref_distance, ref_fraction, ref_lateral = point_to_segments(
                point_xy,
                arc_starts[reference_arc_id : reference_arc_id + 1],
                arc_ends[reference_arc_id : reference_arc_id + 1],
            )
            nearby_indices = np.append(nearby_indices, reference_arc_id)
            distances = np.append(distances, ref_distance[0])
            fractions = np.append(fractions, ref_fraction[0])
            lateral = np.append(lateral, ref_lateral[0])

        rank_order = np.lexsort((nearby_indices, distances))
        ranks = np.empty(len(nearby_indices), dtype=np.int32)
        ranks[rank_order] = np.arange(len(nearby_indices), dtype=np.int32)
        vectors = arc_ends[nearby_indices] - arc_starts[nearby_indices]
        arc_lengths = np.linalg.norm(vectors, axis=1)
        arc_bearings = np.arctan2(vectors[:, 1], vectors[:, 0])
        heading_delta = gps_bearing[point_idx] - arc_bearings
        alignment_cos = (
            np.cos(heading_delta) if gps_bearing_valid[point_idx] else np.zeros(len(vectors))
        )
        alignment_sin = (
            np.sin(heading_delta) if gps_bearing_valid[point_idx] else np.zeros(len(vectors))
        )
        candidate_counts.append(len(nearby_indices))
        retained_points += 1

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
                    "point_idx": point_idx,
                    "candidate_arc_id": int(candidate_idx),
                    "split": split,
                    "label": int(candidate_idx == reference_arc_id),
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

    frame = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    stats = {
        "trip_id": trip_id,
        "split": split,
        "track_points": int(len(track)),
        "nodes": int(len(nodes)),
        "directed_arcs": int(len(arcs)),
        "route_arcs": int(len(route)),
        "route_directed_continuity_fraction": float(
            np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]) if len(route) > 1 else 1.0
        ),
        "reference_matched_points": retained_points,
        "unmatched_or_low_confidence_points": points_without_reference,
        "candidate_rows": int(len(frame)),
        "positive_rows": int(frame["label"].sum()) if not frame.empty else 0,
        "reference_distance_m_p50_p90_p99": [
            float(x) for x in np.percentile(route_match_distance, [50, 90, 99])
        ],
        "candidates_per_labeled_point_p50_p90": [
            float(x) for x in np.percentile(candidate_counts, [50, 90])
        ]
        if candidate_counts
        else [],
    }
    return frame, stats


def build(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = ROOT / settings["data"]["root"] / settings["data"]["training_partition"]
    output_dir = ROOT / settings["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trip_folders = sorted(path for path in data_root.iterdir() if path.is_dir())
    trip_ids = [folder.name for folder in trip_folders]
    split_map = make_split(
        trip_ids,
        float(settings["validation"]["fraction"]),
        int(settings["validation"]["random_seed"]),
    )

    split_frame = pd.DataFrame(
        [{"trip_id": trip_id, "split": split_map[trip_id]} for trip_id in trip_ids]
    )
    split_frame.to_csv(output_dir / settings["output"]["split"], index=False)
    candidate_path = output_dir / settings["output"]["candidate_features"]
    sample_path = output_dir / settings["output"]["sample"]
    candidate_path.unlink(missing_ok=True)
    sample_path.unlink(missing_ok=True)

    trip_stats: list[dict[str, Any]] = []
    sample_remaining = int(settings["sample"]["max_rows"])
    for index, folder in enumerate(trip_folders, start=1):
        frame, stats = trip_candidate_frame(
            folder.name, folder, split_map[folder.name], settings
        )
        frame.to_csv(
            candidate_path,
            mode="a",
            header=not candidate_path.exists(),
            index=False,
            float_format="%.6f",
        )
        trips_remaining = len(trip_folders) - index + 1
        sample_quota = int(np.ceil(sample_remaining / trips_remaining))
        if sample_remaining > 0 and not frame.empty:
            sample = frame.sample(
                n=min(sample_quota, len(frame)),
                random_state=int(settings["validation"]["random_seed"]) + index,
            )
            sample.to_csv(
                sample_path,
                mode="a",
                header=not sample_path.exists(),
                index=False,
                float_format="%.6f",
            )
            sample_remaining -= len(sample)
        trip_stats.append(stats)
        print(
            f"[{index:02d}/{len(trip_folders):02d}] trip={folder.name} "
            f"split={split_map[folder.name]} candidates={stats['candidate_rows']:,} "
            f"labeled_points={stats['reference_matched_points']:,}/"
            f"{stats['track_points']:,} positives={stats['positive_rows']:,}"
        )

    total_points = sum(item["track_points"] for item in trip_stats)
    total_candidates = sum(item["candidate_rows"] for item in trip_stats)
    total_positives = sum(item["positive_rows"] for item in trip_stats)
    summary = {
        "source_partition": "train",
        "source_path": str(data_root.relative_to(ROOT)),
        "student_test_read": False,
        "trip_count": len(trip_stats),
        "train_trip_count": sum(value == "train" for value in split_map.values()),
        "validation_trip_count": sum(value == "validation" for value in split_map.values()),
        "raw_total_track_points": total_points,
        "candidate_feature_rows": total_candidates,
        "positive_candidate_rows": total_positives,
        "negative_candidate_rows": total_candidates - total_positives,
        "candidate_rows_by_split": {
            split: sum(
                item["candidate_rows"] for item in trip_stats if item["split"] == split
            )
            for split in ("train", "validation")
        },
        "positive_rows_by_split": {
            split: sum(
                item["positive_rows"] for item in trip_stats if item["split"] == split
            )
            for split in ("train", "validation")
        },
        "config": settings,
        "trips": trip_stats,
    }
    (output_dir / settings["output"]["summary"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {total_candidates:,} candidate rows ({total_positives:,} positive) "
        f"to {candidate_path.relative_to(ROOT)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    build(args.config.resolve())


if __name__ == "__main__":
    main()