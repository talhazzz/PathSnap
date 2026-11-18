"""Inspect raw trips dynamically, validating train and student_test separately."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/raw/Student folder"
PARTITIONS = ("train", "student_test")
EXTENSIONS = (".track", ".nodes", ".arcs", ".route")


def load_source(path: Path, columns: int, dtype: Any = float) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got {values.shape}")
    return values


def inspect_partition(partition: str) -> dict[str, Any]:
    """Validate every immediate trip directory in one partition independently."""
    partition_path = DATA_ROOT / partition
    trips: list[dict[str, Any]] = []
    for folder in sorted(path for path in partition_path.iterdir() if path.is_dir()):
        trip_id = folder.name
        paths = {extension: folder / f"{trip_id}{extension}" for extension in EXTENSIONS}
        absent = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
        if absent:
            raise FileNotFoundError(f"{partition}/{trip_id}: missing files {absent}")

        track = load_source(paths[".track"], 3)
        nodes = load_source(paths[".nodes"], 2)
        arcs = load_source(paths[".arcs"], 2, dtype=np.int64)
        route = np.loadtxt(paths[".route"], dtype=np.int64, ndmin=1)
        if not np.isfinite(track).all() or not np.isfinite(nodes).all():
            raise ValueError(f"{partition}/{trip_id}: non-finite coordinate or time value")
        if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
            raise ValueError(f"{partition}/{trip_id}: arc endpoint outside .nodes")
        if np.any(route < 0) or np.any(route >= len(arcs)):
            raise ValueError(f"{partition}/{trip_id}: route entry outside .arcs")

        continuity = (
            float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]))
            if len(route) > 1
            else 1.0
        )
        trips.append(
            {
                "trip_id": trip_id,
                "track_points": int(len(track)),
                "nodes": int(len(nodes)),
                "directed_arcs": int(len(arcs)),
                "route_arcs": int(len(route)),
                "route_directed_continuity_fraction": continuity,
            }
        )

    return {
        "partition": partition,
        "trip_count": len(trips),
        "total_track_points": sum(trip["track_points"] for trip in trips),
        "total_nodes": sum(trip["nodes"] for trip in trips),
        "total_directed_arcs": sum(trip["directed_arcs"] for trip in trips),
        "total_route_arcs": sum(trip["route_arcs"] for trip in trips),
        "trips": trips,
    }


def main() -> None:
    reports = {partition: inspect_partition(partition) for partition in PARTITIONS}
    output = ROOT / "artifacts/reports/dataset_inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for partition, report in reports.items():
        print(
            f"{partition}: trips={report['trip_count']} "
            f"track_points={report['total_track_points']:,} "
            f"nodes={report['total_nodes']:,} "
            f"arcs={report['total_directed_arcs']:,} "
            f"route_arcs={report['total_route_arcs']:,}"
        )
    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()