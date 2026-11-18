"""End-to-end trip candidate extraction, HMM/Viterbi route reconstruction,
server-side evaluation, and Folium map generation for PathSnap.

Aligns strictly with the team's canonical 17-feature schema defined in
ml/preprocessing/build_shared_dataset.py (origin/ml-dev) so models trained
on the shared dataset work seamlessly during live evaluation and deployment.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_ROOT = BACKEND_DIR / "data" / "raw" / "Student folder"

# Import Medha's canonical 17-feature list directly from ml/preprocessing when available
try:
    from ml.preprocessing.build_shared_dataset import FEATURE_COLUMNS as CANONICAL_ML_FEATURES
    FEATURE_COLUMNS: list[str] = list(CANONICAL_ML_FEATURES)
except Exception:
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

# Exact formula-identical column aliases between ml-dev cf6b948 and 7bab901
FEATURE_COLUMN_ALIASES: dict[str, str] = {
    "gps_speed_m_per_time_unit": "gps_speed_mps",
    "gps_speed_mps": "gps_speed_m_per_time_unit",
    "time_delta_units": "time_delta_s",
    "time_delta_s": "time_delta_units",
}


def align_feature_frame(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Align a candidate feature DataFrame to a model artifact's exact expected_columns.

    Transparently bridges the formula-identical column rename between ml-dev cf6b948
    (gps_speed_mps / time_delta_s) and ml-dev 7bab901 (gps_speed_m_per_time_unit / time_delta_units),
    while strictly rejecting any truly missing or unexpected columns.
    """
    aligned = frame.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            alias = FEATURE_COLUMN_ALIASES.get(col)
            if alias and alias in aligned.columns:
                aligned[col] = aligned[alias]

    # Drop replaced alias columns only if they are not themselves expected
    expected_set = set(expected_columns)
    for col in list(aligned.columns):
        if col not in expected_set:
            alias = FEATURE_COLUMN_ALIASES.get(col)
            if alias and alias in expected_set:
                aligned = aligned.drop(columns=[col])

    missing = expected_set - set(aligned.columns)
    extra = set(aligned.columns) - expected_set
    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; unexpected: {sorted(extra)}"
        )
    return aligned[expected_columns]


def load_raw_table(path: Path, expected_cols: int, dtype: Any = float) -> np.ndarray:
    """Load a whitespace-delimited PathSnap file without headers."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        raise ValueError(
            f"{path.name}: expected {expected_cols} columns, got shape {arr.shape}"
        )
    return arr


def load_trip_arrays(
    trip_id: str,
    partition_dir: Path,
    include_route: bool = True,
) -> dict[str, Any]:
    """Load raw numpy arrays for a single trip; .route is optional."""
    trip_dir = partition_dir / str(trip_id)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    prefix = trip_dir / str(trip_id)
    track = load_raw_table(prefix.with_suffix(".track"), 3, dtype=np.float64)
    nodes = load_raw_table(prefix.with_suffix(".nodes"), 2, dtype=np.float64)
    arcs = load_raw_table(prefix.with_suffix(".arcs"), 2, dtype=np.int64)

    route = None
    route_path = prefix.with_suffix(".route")
    if include_route and route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64, ndmin=1)

    return {
        "trip_id": str(trip_id),
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "has_route": route is not None,
    }


def project_coordinates(
    coordinates: np.ndarray, lon0: float, lat0: float
) -> np.ndarray:
    """Project trip-local lon/lat coordinates into approximate planar metres."""
    scale_x = 111_320.0 * np.cos(np.deg2rad(lat0))
    scale_y = 110_574.0
    return np.column_stack(
        ((coordinates[:, 0] - lon0) * scale_x, (coordinates[:, 1] - lat0) * scale_y)
    )


def point_to_segments(
    point: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return planar point-to-segment distances, projection fractions, signed offsets."""
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


def track_context(track_xy: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute per-observation GPS motion features."""
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


def align_track_to_route(
    track_xy: np.ndarray,
    route_starts: np.ndarray,
    route_ends: np.ndarray,
    advance_penalty: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Monotonically align GPS observations to a reference .route sequence (server-side eval only)."""
    point_count = len(track_xy)
    route_count = len(route_starts)
    if point_count == 0 or route_count == 0:
        raise ValueError("Cannot align an empty track or route")

    route_indices = np.arange(route_count, dtype=np.float64)
    pred_dtype = np.uint16 if route_count <= np.iinfo(np.uint16).max else np.uint32
    predecessors = np.zeros((point_count, route_count), dtype=pred_dtype)
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
        predecessors[point_idx] = prefix_indices.astype(pred_dtype)

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


def extract_trip_candidates(
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    search_radius_m: float = 60.0,
    max_candidates: int = 16,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract canonical 17-feature candidate rows for inference (NEVER reads .route)."""
    finite_mask = np.isfinite(track).all(axis=1)
    if not finite_mask.all():
        track = track[finite_mask]
    if len(track) == 0 or len(arcs) == 0 or len(nodes) == 0:
        raise ValueError(f"Trip {trip_id}: empty or invalid track/nodes/arcs input")

    origin_lon = float(np.median(track[:, 0]))
    origin_lat = float(np.median(track[:, 1]))
    track_xy = project_coordinates(track[:, :2], origin_lon, origin_lat)
    node_xy = project_coordinates(nodes, origin_lon, origin_lat)
    arc_starts = node_xy[arcs[:, 0]]
    arc_ends = node_xy[arcs[:, 1]]

    # Restrict STRtree construction to corridor bounding box around trajectory for speed
    margin = max(search_radius_m + 300.0, 500.0)
    xmin, ymin = track_xy.min(axis=0) - margin
    xmax, ymax = track_xy.max(axis=0) + margin
    arc_xmin = np.minimum(arc_starts[:, 0], arc_ends[:, 0])
    arc_xmax = np.maximum(arc_starts[:, 0], arc_ends[:, 0])
    arc_ymin = np.minimum(arc_starts[:, 1], arc_ends[:, 1])
    arc_ymax = np.maximum(arc_starts[:, 1], arc_ends[:, 1])
    corridor_mask = (
        (arc_xmax >= xmin) & (arc_xmin <= xmax) & (arc_ymax >= ymin) & (arc_ymin <= ymax)
    )
    corridor_arc_ids = np.flatnonzero(corridor_mask)
    if corridor_arc_ids.size == 0:
        corridor_arc_ids = np.arange(len(arcs), dtype=np.int64)

    lines = [
        LineString((arc_starts[aid], arc_ends[aid])) for aid in corridor_arc_ids
    ]
    tree = STRtree(lines)

    step_distance, speed, time_delta, gps_bearing, gps_bearing_valid = track_context(
        track_xy, track[:, 2]
    )

    records: list[dict[str, Any]] = []
    for point_idx, point_xy in enumerate(track_xy):
        point = Point(float(point_xy[0]), float(point_xy[1]))
        local_hits = np.asarray(
            tree.query(
                box(
                    point_xy[0] - search_radius_m,
                    point_xy[1] - search_radius_m,
                    point_xy[0] + search_radius_m,
                    point_xy[1] + search_radius_m,
                )
            ),
            dtype=np.int64,
        )
        nearby_indices = corridor_arc_ids[local_hits] if local_hits.size else np.empty(0, dtype=np.int64)

        if nearby_indices.size:
            distances, fractions, lateral = point_to_segments(
                point_xy, arc_starts[nearby_indices], arc_ends[nearby_indices]
            )
            in_radius = distances <= search_radius_m
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
            nearest_local = int(tree.nearest(point))
            nearby_indices = np.asarray([int(corridor_arc_ids[nearest_local])], dtype=np.int64)
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
        alignment_cos = (
            np.cos(heading_delta) if gps_bearing_valid[point_idx] else np.zeros(len(vectors))
        )
        alignment_sin = (
            np.sin(heading_delta) if gps_bearing_valid[point_idx] else np.zeros(len(vectors))
        )

        for cid, dist, frac, side, length, rank, cos_d, sin_d in zip(
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
                    "trip_id": str(trip_id),
                    "point_idx": int(point_idx),
                    "candidate_arc_id": int(cid),
                    "distance_m": float(dist),
                    "signed_lateral_offset_m": float(side),
                    "projection_fraction": float(frac),
                    "distance_to_arc_start_m": float(frac * length),
                    "distance_to_arc_end_m": float((1.0 - frac) * length),
                    "arc_length_m": float(length),
                    "candidate_rank_distance": int(rank),
                    "candidate_count_at_point": int(len(nearby_indices)),
                    "is_nearest_candidate": int(rank == 0),
                    "is_within_search_radius": int(dist <= search_radius_m),
                    "gps_step_distance_m": float(step_distance[point_idx]),
                    "gps_speed_m_per_time_unit": float(speed[point_idx]),
                    "time_delta_units": float(time_delta[point_idx]),
                    "gps_bearing_valid": int(gps_bearing_valid[point_idx]),
                    "heading_alignment_cos": float(cos_d),
                    "heading_alignment_sin": float(sin_d),
                    "is_self_loop": int(length < 0.01),
                }
            )

    candidates_df = pd.DataFrame.from_records(
        records, columns=["trip_id", "point_idx", "candidate_arc_id"] + FEATURE_COLUMNS
    )
    geo_context = {
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "track_xy": track_xy,
        "node_xy": node_xy,
        "arc_starts": arc_starts,
        "arc_ends": arc_ends,
        "corridor_arc_ids": corridor_arc_ids,
        "step_distance": step_distance,
        "time_delta": time_delta,
    }
    return candidates_df, geo_context


def _build_corridor_graph(
    arcs: np.ndarray,
    arc_starts: np.ndarray,
    arc_ends: np.ndarray,
    corridor_arc_ids: np.ndarray,
) -> tuple[dict[int, list[tuple[int, float, int]]], np.ndarray]:
    """Build directed adjacency list restricted to the trip corridor."""
    all_lengths = np.linalg.norm(arc_ends - arc_starts, axis=1)
    adj: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for aid in corridor_arc_ids:
        u = int(arcs[aid, 0])
        v = int(arcs[aid, 1])
        w = float(all_lengths[aid])
        adj[u].append((v, w, int(aid)))
    return adj, all_lengths


def _bounded_dijkstra(
    adj: dict[int, list[tuple[int, float, int]]],
    start_node: int,
    target_nodes: set[int],
    max_dist: float,
    max_expansions: int = 800,
) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
    """Bounded shortest path from start_node to any node in target_nodes."""
    dist: dict[int, float] = {start_node: 0.0}
    parent: dict[int, tuple[int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, start_node)]
    remaining = set(target_nodes)
    remaining.discard(start_node)
    expansions = 0

    while heap and remaining and expansions < max_expansions:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if d > max_dist:
            break
        expansions += 1
        if u in remaining:
            remaining.remove(u)
            if not remaining:
                break
        for v, w, aid in adj.get(u, ()):
            nd = d + w
            if nd <= max_dist and nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = (u, aid)
                heapq.heappush(heap, (nd, v))
    return dist, parent


def _reconstruct_bridged_arcs(
    adj: dict[int, list[tuple[int, float, int]]],
    arcs: np.ndarray,
    point_arc_ids: list[int],
) -> list[int]:
    """Deduplicate consecutive identical arcs and bridge multi-hop transitions via shortest path."""
    if not point_arc_ids:
        return []
    collapsed: list[int] = [int(point_arc_ids[0])]
    for aid in point_arc_ids[1:]:
        aid_int = int(aid)
        if aid_int != collapsed[-1]:
            collapsed.append(aid_int)

    bridged: list[int] = [collapsed[0]]
    for prev_arc, next_arc in zip(collapsed[:-1], collapsed[1:]):
        u_end = int(arcs[prev_arc, 1])
        v_start = int(arcs[next_arc, 0])
        if u_end == v_start:
            bridged.append(next_arc)
            continue
        dist_map, parent = _bounded_dijkstra(
            adj, u_end, {v_start}, max_dist=1500.0, max_expansions=1500
        )
        if v_start in dist_map:
            path_arcs: list[int] = []
            cur = v_start
            while cur != u_end and cur in parent:
                pnode, paid = parent[cur]
                path_arcs.append(paid)
                cur = pnode
            if cur == u_end:
                path_arcs.reverse()
                for baid in path_arcs:
                    if baid != bridged[-1]:
                        bridged.append(baid)
        if next_arc != bridged[-1]:
            bridged.append(next_arc)
    return bridged


def compute_route_continuity(arcs: np.ndarray, arc_sequence: list[int] | np.ndarray) -> float:
    """Compute fraction of consecutive arc transitions that are directed-connected (to_node == next from_node)."""
    seq = np.asarray(arc_sequence, dtype=np.int64)
    if len(seq) <= 1:
        return 1.0
    # Collapse consecutive identical arcs (vehicle staying on the same road segment)
    keep = np.r_[True, seq[1:] != seq[:-1]]
    collapsed = seq[keep]
    if len(collapsed) <= 1:
        return 1.0
    return float(np.mean(arcs[collapsed[:-1], 1] == arcs[collapsed[1:], 0]))


def classifier_only_decode(
    candidates_df: pd.DataFrame, probabilities: np.ndarray
) -> list[int]:
    """Pick the single highest-probability candidate arc per GPS observation."""
    ranked = candidates_df[["point_idx", "candidate_arc_id"]].copy()
    ranked["prob"] = probabilities
    best = (
        ranked.sort_values(
            ["point_idx", "prob", "candidate_arc_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("point_idx")
        .sort_values("point_idx")
    )
    return [int(x) for x in best["candidate_arc_id"].to_numpy()]


def viterbi_decode_route(
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    arcs: np.ndarray,
    geo_context: dict[str, Any],
    emission_weight: float = 1.2,
    sigma_z: float = 25.0,
    beta_base: float = 12.0,
    disconnect_penalty: float = 22.0,
) -> dict[str, Any]:
    """HMM + Viterbi decoding combining ML classifier emissions with directed road-network transitions."""
    adj, all_lengths = _build_corridor_graph(
        arcs,
        geo_context["arc_starts"],
        geo_context["arc_ends"],
        geo_context["corridor_arc_ids"],
    )
    step_distance = geo_context["step_distance"]
    time_delta = geo_context["time_delta"]

    work = candidates_df[
        ["point_idx", "candidate_arc_id", "distance_m", "projection_fraction"]
    ].copy()
    work["prob"] = np.clip(probabilities, 1e-6, 1.0 - 1e-6)

    grouped = [grp for _, grp in work.groupby("point_idx", sort=True)]
    num_steps = len(grouped)
    if num_steps == 0:
        return {"point_arc_ids": [], "route_arc_ids": [], "connected_transitions": 0, "total_transitions": 0}

    # Build step state arrays
    step_arc_ids: list[np.ndarray] = []
    step_dists: list[np.ndarray] = []
    step_fracs: list[np.ndarray] = []
    step_emissions: list[np.ndarray] = []
    step_point_indices: list[int] = []

    for grp in grouped:
        aids = grp["candidate_arc_id"].to_numpy(dtype=np.int64)
        dists = grp["distance_m"].to_numpy(dtype=np.float64)
        fracs = grp["projection_fraction"].to_numpy(dtype=np.float64)
        probs = grp["prob"].to_numpy(dtype=np.float64)
        # Combined log emission: ML probability + Gaussian spatial anchor
        em = emission_weight * np.log(probs) - 0.5 * (dists / sigma_z) ** 2
        step_arc_ids.append(aids)
        step_dists.append(dists)
        step_fracs.append(fracs)
        step_emissions.append(em)
        step_point_indices.append(int(grp["point_idx"].iloc[0]))

    dp_prev = step_emissions[0].copy()
    backpointers: list[np.ndarray] = []

    for t in range(1, num_steps):
        prev_aids = step_arc_ids[t - 1]
        prev_fracs = step_fracs[t - 1]
        curr_aids = step_arc_ids[t]
        curr_fracs = step_fracs[t]
        curr_em = step_emissions[t]

        p_idx = step_point_indices[t]
        d_gps = float(step_distance[p_idx])
        dt = float(time_delta[p_idx])
        # Adapt beta and search cutoff during GPS gaps
        gap_factor = min(max(dt, 1.0), 30.0)
        beta = beta_base + 0.15 * d_gps + (1.5 * gap_factor if dt > 3.0 else 0.0)
        search_cutoff = min(max(d_gps * 2.5 + 80.0, 150.0), 1800.0)

        curr_from_nodes = arcs[curr_aids, 0]
        target_node_set = {int(n) for n in curr_from_nodes}

        # Precompute bounded shortest paths per unique prev_to_node (only when needed)
        dijkstra_cache: dict[int, dict[int, float]] = {}

        n_prev = len(prev_aids)
        n_curr = len(curr_aids)
        dp_curr = np.full(n_curr, -np.inf, dtype=np.float64)
        bp = np.zeros(n_curr, dtype=np.int32)

        for i in range(n_prev):
            u_arc = int(prev_aids[i])
            u_to = int(arcs[u_arc, 1])
            u_len = float(all_lengths[u_arc])
            u_rem = (1.0 - float(prev_fracs[i])) * u_len
            base_score = dp_prev[i]

            # Check if we need Dijkstra for non-trivial targets
            need_dijkstra = False
            for j in range(n_curr):
                v_arc = int(curr_aids[j])
                if v_arc != u_arc and int(curr_from_nodes[j]) != u_to:
                    need_dijkstra = True
                    break

            if need_dijkstra and u_to not in dijkstra_cache:
                d_map, _ = _bounded_dijkstra(
                    adj, u_to, target_node_set, max_dist=search_cutoff, max_expansions=600
                )
                dijkstra_cache[u_to] = d_map

            d_map = dijkstra_cache.get(u_to, {})

            for j in range(n_curr):
                v_arc = int(curr_aids[j])
                v_from = int(curr_from_nodes[j])
                v_len = float(all_lengths[v_arc])
                v_prog = float(curr_fracs[j]) * v_len

                if v_arc == u_arc:
                    delta_f = float(curr_fracs[j]) - float(prev_fracs[i])
                    if delta_f >= -0.05:
                        d_route = max(0.0, delta_f * u_len)
                        trans = -abs(d_route - d_gps) / beta
                    else:
                        # Slight backtracking penalty along the same directed arc
                        d_route = abs(delta_f) * u_len
                        trans = -(abs(d_route - d_gps) + 12.0) / beta
                elif v_from == u_to:
                    # Direct directed road connection
                    d_route = u_rem + v_prog
                    trans = -abs(d_route - d_gps) / beta
                elif v_from in d_map:
                    # Multi-hop directed road connection
                    d_route = u_rem + d_map[v_from] + v_prog
                    trans = -abs(d_route - d_gps) / beta - 0.15
                else:
                    # Disconnected / GPS jump fallback penalty
                    trans = -disconnect_penalty - 0.02 * d_gps

                score = base_score + trans + curr_em[j]
                if score > dp_curr[j]:
                    dp_curr[j] = score
                    bp[j] = i

        backpointers.append(bp)
        dp_prev = dp_curr

    # Backtrack optimal state sequence
    best_states = np.empty(num_steps, dtype=np.int32)
    best_states[-1] = int(np.argmax(dp_prev))
    for t in range(num_steps - 1, 0, -1):
        best_states[t - 1] = backpointers[t - 1][best_states[t]]

    point_arc_ids = [int(step_arc_ids[t][best_states[t]]) for t in range(num_steps)]
    route_arc_ids = _reconstruct_bridged_arcs(adj, arcs, point_arc_ids)

    return {
        "point_arc_ids": point_arc_ids,
        "route_arc_ids": route_arc_ids,
    }


def arcs_to_polyline(
    nodes: np.ndarray, arcs: np.ndarray, arc_ids: list[int]
) -> list[list[float]]:
    """Convert an ordered sequence of arc IDs to [[lat, lon], ...] polyline coordinates."""
    if not arc_ids:
        return []
    coords: list[list[float]] = []
    for idx, aid in enumerate(arc_ids):
        u = int(arcs[aid, 0])
        v = int(arcs[aid, 1])
        pt_u = [float(nodes[u, 1]), float(nodes[u, 0])]
        pt_v = [float(nodes[v, 1]), float(nodes[v, 0])]
        if idx == 0 or coords[-1] != pt_u:
            coords.append(pt_u)
        coords.append(pt_v)
    return coords


def scores_to_per_point_softmax(
    candidates_df: pd.DataFrame,
    scores: np.ndarray,
    temperature: float = 5.0,
) -> np.ndarray:
    """Convert continuous candidate ranking scores (e.g. Linear Regression) into
    valid per-observation emission probabilities via numerically stable softmax.
    Preserves exact candidate rank order at every GPS point without treating
    regression outputs as binary class labels.
    """
    probs = np.empty(len(scores), dtype=np.float64)
    p_idxs = candidates_df["point_idx"].to_numpy()
    # Group contiguous/grouped point indices cleanly
    work = pd.DataFrame({"point_idx": p_idxs, "pos": np.arange(len(scores)), "score": scores})
    for _, grp in work.groupby("point_idx", sort=False):
        pos = grp["pos"].to_numpy()
        s = grp["score"].to_numpy(dtype=np.float64) * temperature
        s = s - np.max(s)
        ex = np.exp(np.clip(s, -50.0, 0.0))
        probs[pos] = ex / np.sum(ex)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def evaluate_against_reference(
    track_xy: np.ndarray,
    arcs: np.ndarray,
    arc_starts: np.ndarray,
    arc_ends: np.ndarray,
    route_ref: np.ndarray,
    candidates_df: pd.DataFrame,
    probabilities: np.ndarray,
    clf_point_arcs: list[int],
    hmm_point_arcs: list[int],
    hmm_route_arcs: list[int],
    label_max_distance_m: float = 45.0,
    is_regression_scorer: bool = False,
) -> dict[str, Any]:
    """Server-side evaluation ONLY (never exposed to model inputs or UI raw labels)."""
    route_starts = arc_starts[route_ref]
    route_ends = arc_ends[route_ref]
    route_positions, match_dist = align_track_to_route(
        track_xy, route_starts, route_ends, advance_penalty=0.15
    )
    ref_point_arcs = route_ref[route_positions]
    valid_mask = match_dist <= label_max_distance_m

    # Check candidate miss rate on valid points
    cand_by_point = (
        candidates_df.groupby("point_idx")["candidate_arc_id"].apply(set).to_dict()
    )
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) > 0:
        hits = sum(
            int(ref_point_arcs[idx]) in cand_by_point.get(int(idx), set())
            for idx in valid_indices
        )
        candidate_miss_rate = float(1.0 - hits / len(valid_indices))
        clf_top1 = float(
            np.mean([clf_point_arcs[idx] == int(ref_point_arcs[idx]) for idx in valid_indices])
        )
        hmm_top1 = float(
            np.mean([hmm_point_arcs[idx] == int(ref_point_arcs[idx]) for idx in valid_indices])
        )
    else:
        candidate_miss_rate = 0.0
        clf_top1 = 0.0
        hmm_top1 = 0.0

    # Sequence / Arc-set overlap metrics (precision, recall, F1, Jaccard vs reference .route)
    ref_set = set(int(x) for x in route_ref)
    clf_set = set(int(x) for x in clf_point_arcs)
    hmm_set = set(int(x) for x in hmm_route_arcs)

    def _set_metrics(pred_set: set[int]) -> dict[str, float]:
        inter = len(pred_set & ref_set)
        prec = inter / len(pred_set) if pred_set else 0.0
        rec = inter / len(ref_set) if ref_set else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        jacc = inter / len(pred_set | ref_set) if (pred_set | ref_set) else 0.0
        return {
            "route_precision": round(prec, 4),
            "route_recall": round(rec, 4),
            "route_f1": round(f1, 4),
            "route_jaccard": round(jacc, 4),
        }

    # Candidate-row metrics on valid points
    y_true_rows: list[int] = []
    y_prob_rows: list[float] = []
    y_top1_pred_rows: list[int] = []
    p_idxs = candidates_df["point_idx"].to_numpy()
    c_aids = candidates_df["candidate_arc_id"].to_numpy()
    for p_i, c_a, prob in zip(p_idxs, c_aids, probabilities):
        if valid_mask[p_i]:
            y_true_rows.append(1 if int(c_a) == int(ref_point_arcs[p_i]) else 0)
            y_prob_rows.append(float(prob))
            y_top1_pred_rows.append(1 if int(c_a) == int(clf_point_arcs[p_i]) else 0)

    from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

    y_true_arr = np.asarray(y_true_rows, dtype=np.int8)
    y_prob_arr = np.clip(np.asarray(y_prob_rows, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    # For regression scorers, evaluate Top-1 ranked selection (never threshold raw regression outputs as class labels)
    if is_regression_scorer:
        y_pred_arr = np.asarray(y_top1_pred_rows, dtype=np.int8)
        ll_value = None
    else:
        y_pred_arr = (y_prob_arr >= 0.5).astype(np.int8)
        ll_value = (
            round(float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])), 4)
            if len(y_true_arr) > 0
            else 0.0
        )

    macro_f1 = (
        float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        if len(y_true_arr) > 0
        else 0.0
    )

    return {
        "valid_evaluated_points": int(len(valid_indices)),
        "total_gps_points": int(len(track_xy)),
        "scoring_mode": "regression_ranking" if is_regression_scorer else "classification_probability",
        "candidate_miss_rate": round(candidate_miss_rate, 4),
        "candidate_macro_f1": round(macro_f1, 4),
        "candidate_precision_macro": round(
            float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
            if len(y_true_arr)
            else 0.0,
            4,
        ),
        "candidate_recall_macro": round(
            float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
            if len(y_true_arr)
            else 0.0,
            4,
        ),
        "candidate_log_loss": ll_value,
        "classifier_only": {
            "top1_accuracy": round(clf_top1, 4),
            "route_continuity": round(compute_route_continuity(arcs, clf_point_arcs), 4),
            **_set_metrics(clf_set),
        },
        "hmm_viterbi": {
            "top1_accuracy": round(hmm_top1, 4),
            "route_continuity": round(compute_route_continuity(arcs, hmm_route_arcs), 4),
            **_set_metrics(hmm_set),
        },
    }


def generate_folium_map(
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    hmm_route_arcs: list[int],
    clf_point_arcs: list[int] | None = None,
    route_ref: np.ndarray | None = None,
    corridor_arc_ids: np.ndarray | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate an interactive Folium HTML map with distinct layers, legend, and LayerControl."""
    center_lat = float(np.median(track[:, 1]))
    center_lon = float(np.median(track[:, 0]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Relevant road-network geometry (sample up to 2,000 corridor arcs for snappy rendering)
    road_group = folium.FeatureGroup(name="Road Network (Corridor)", show=True)
    if corridor_arc_ids is not None and len(corridor_arc_ids) > 0:
        step = max(1, len(corridor_arc_ids) // 1500)
        for aid in corridor_arc_ids[::step]:
            u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
            folium.PolyLine(
                [[float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]],
                color="#94a3b8",
                weight=1.2,
                opacity=0.45,
            ).add_to(road_group)
    road_group.add_to(fmap)

    # Layer 2: Raw GPS observations
    gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    gps_step = max(1, len(track) // 400)
    for row in track[::gps_step]:
        folium.CircleMarker(
            location=[float(row[1]), float(row[0])],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.8,
            weight=1,
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Layer 3: Reference route (only for training / offline visualization, never for evaluator live UI)
    if route_ref is not None and len(route_ref) > 0:
        ref_group = folium.FeatureGroup(name="Reference Route", show=True)
        ref_coords = arcs_to_polyline(nodes, arcs, [int(x) for x in route_ref])
        if ref_coords:
            folium.PolyLine(
                ref_coords,
                color="#16a34a",
                weight=5,
                opacity=0.7,
                tooltip=f"Reference Route ({len(route_ref)} arcs)",
            ).add_to(ref_group)
        ref_group.add_to(fmap)

    # Layer 4: Classifier-only route (optional comparison layer)
    if clf_point_arcs:
        clf_group = folium.FeatureGroup(name="Classifier-Only Route", show=False)
        dedup_clf = [clf_point_arcs[0]] + [
            a for a, b in zip(clf_point_arcs[1:], clf_point_arcs[:-1]) if a != b
        ]
        clf_coords = arcs_to_polyline(nodes, arcs, dedup_clf)
        if clf_coords:
            folium.PolyLine(
                clf_coords,
                color="#f59e0b",
                weight=3.5,
                opacity=0.75,
                dash_array="6 4",
                tooltip=f"Classifier-Only ({len(dedup_clf)} arcs)",
            ).add_to(clf_group)
        clf_group.add_to(fmap)

    # Layer 5: Predicted HMM + Viterbi Route
    pred_group = folium.FeatureGroup(name="Predicted Route (ML + HMM/Viterbi)", show=True)
    pred_coords = arcs_to_polyline(nodes, arcs, hmm_route_arcs)
    if pred_coords:
        folium.PolyLine(
            pred_coords,
            color="#dc2626",
            weight=4,
            opacity=0.9,
            tooltip=f"Predicted Route ({len(hmm_route_arcs)} arcs)",
        ).add_to(pred_group)
        folium.Marker(
            pred_coords[0],
            popup=f"Trip {trip_id} Start",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(pred_group)
        folium.Marker(
            pred_coords[-1],
            popup=f"Trip {trip_id} End",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(pred_group)
    pred_group.add_to(fmap)

    # Fit bounds to GPS trajectory
    fmap.fit_bounds(
        [[float(track[:, 1].min()), float(track[:, 0].min())],
         [float(track[:, 1].max()), float(track[:, 0].max())]]
    )

    # Custom floating legend
    ref_legend = (
        '<div><span style="display:inline-block;width:14px;height:4px;background:#16a34a;margin-right:6px;"></span>Reference Route</div>'
        if route_ref is not None
        else ""
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,0.94); padding: 10px 14px;
                border-radius: 8px; border: 1px solid #cbd5e1; font-family: Arial, sans-serif;
                font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
        <div style="font-weight: 700; margin-bottom: 6px;">PathSnap — Trip {trip_id}</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3b82f6;margin-right:6px;"></span>Raw GPS Points</div>
        <div><span style="display:inline-block;width:14px;height:2px;background:#94a3b8;margin-right:6px;"></span>Road Network</div>
        {ref_legend}
        <div><span style="display:inline-block;width:14px;height:4px;background:#dc2626;margin-right:6px;"></span>Predicted Route (HMM/Viterbi)</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(output_path))
    return fmap.get_root().render()


def run_trip_inference(
    trip_id: str,
    model_artifact: dict[str, Any],
    partition_dir: Path,
    use_hmm: bool = True,
    evaluate_server_side: bool = True,
) -> dict[str, Any]:
    """Execute the complete inference pipeline for one trip (zero label leakage)."""
    t0 = time.perf_counter()
    trip = load_trip_arrays(trip_id, partition_dir, include_route=evaluate_server_side)
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    t_feat_start = time.perf_counter()
    candidates_df, geo_ctx = extract_trip_candidates(trip_id, track, nodes, arcs)
    feat_time = time.perf_counter() - t_feat_start

    model = model_artifact["model"]
    feature_cols = model_artifact["feature_columns"]
    aligned_features = align_feature_frame(
        candidates_df.drop(columns=["trip_id", "point_idx", "candidate_arc_id"]),
        feature_cols,
    )
    X = aligned_features.to_numpy(dtype=np.float32)

    is_regression_scorer = (
        model_artifact.get("model_type") == "regression_scorer"
        or not hasattr(model, "predict_proba")
    )

    t_clf_start = time.perf_counter()
    if is_regression_scorer:
        # Linear Regression outputs continuous candidate ranking scores (NOT binary class labels)
        raw_scores = model.predict(X).astype(np.float64)
        temp = float(model_artifact.get("hyperparameters", {}).get("hmm_softmax_temperature", 5.0))
        emission_probs = scores_to_per_point_softmax(candidates_df, raw_scores, temperature=temp)
        ranking_scores = raw_scores
    else:
        emission_probs = model.predict_proba(X)[:, 1].astype(np.float64)
        ranking_scores = emission_probs
    clf_time = time.perf_counter() - t_clf_start

    clf_point_arcs = classifier_only_decode(candidates_df, ranking_scores)

    t_hmm_start = time.perf_counter()
    if use_hmm:
        hmm_res = viterbi_decode_route(candidates_df, emission_probs, arcs, geo_ctx)
        hmm_point_arcs = hmm_res["point_arc_ids"]
        hmm_route_arcs = hmm_res["route_arc_ids"]
    else:
        hmm_point_arcs = clf_point_arcs
        hmm_route_arcs = [clf_point_arcs[0]] + [
            a for a, b in zip(clf_point_arcs[1:], clf_point_arcs[:-1]) if a != b
        ] if clf_point_arcs else []
    hmm_time = time.perf_counter() - t_hmm_start

    total_time = time.perf_counter() - t0
    predicted_continuity = compute_route_continuity(arcs, hmm_route_arcs)
    clf_continuity = compute_route_continuity(arcs, clf_point_arcs)

    # Server-side evaluation ONLY (after inference is completely finished!)
    evaluation = None
    if evaluate_server_side and trip["route"] is not None:
        evaluation = evaluate_against_reference(
            track_xy=geo_ctx["track_xy"],
            arcs=arcs,
            arc_starts=geo_ctx["arc_starts"],
            arc_ends=geo_ctx["arc_ends"],
            route_ref=trip["route"],
            candidates_df=candidates_df,
            probabilities=emission_probs,
            clf_point_arcs=clf_point_arcs,
            hmm_point_arcs=hmm_point_arcs,
            hmm_route_arcs=hmm_route_arcs,
            is_regression_scorer=is_regression_scorer,
        )

    # Downsample GPS points slightly only for lightweight JSON rendering if huge
    gps_coords = [[float(r[1]), float(r[0])] for r in track]
    predicted_coords = arcs_to_polyline(nodes, arcs, hmm_route_arcs)
    approx_dist_m = float(np.sum(geo_ctx["step_distance"]))
    duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0

    return {
        "trip_id": str(trip_id),
        "model_name": model_artifact.get("model_name", "unknown"),
        "model_type": "regression_scorer" if is_regression_scorer else "classifier",
        "use_hmm": use_hmm,
        "gps_points_count": int(len(track)),
        "candidate_rows_count": int(len(candidates_df)),
        "predicted_arc_count": int(len(hmm_route_arcs)),
        "predicted_arc_ids": hmm_route_arcs,
        "predicted_route_coordinates": predicted_coords,
        "gps_coordinates": gps_coords,
        "approx_distance_km": round(approx_dist_m / 1000.0, 3),
        "duration_seconds": round(duration_s, 1),
        "route_continuity": round(predicted_continuity, 4),
        "classifier_only_route_continuity": round(clf_continuity, 4),
        "timings_seconds": {
            "feature_extraction": round(feat_time, 4),
            "classifier_inference": round(clf_time, 4),
            "hmm_viterbi_decoding": round(hmm_time, 4),
            "total": round(total_time, 4),
        },
        "server_evaluation": evaluation,
        # Internal objects not for JSON serialization, stripped by API
        "_internal": {
            "track": track,
            "nodes": nodes,
            "arcs": arcs,
            "corridor_arc_ids": geo_ctx["corridor_arc_ids"],
            "clf_point_arcs": clf_point_arcs,
            "hmm_route_arcs": hmm_route_arcs,
        },
    }
