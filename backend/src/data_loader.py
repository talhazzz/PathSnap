from pathlib import Path
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data" / "raw" / "Student folder"


def _read_file(path, expected_columns, dtype=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)

    if df.shape[1] != expected_columns:
        raise ValueError(
            f"{path.name}: expected {expected_columns} columns, "
            f"found {df.shape[1]}"
        )

    if df.empty:
        raise ValueError(f"{path.name} is empty")

    return df


def load_trip(trip_id, partition="train"):
    """Load and validate all four files belonging to one trip."""

    if partition not in {"train", "student_test"}:
        raise ValueError("partition must be 'train' or 'student_test'")

    trip_id = str(trip_id)
    trip_dir = DATA_DIR / partition / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    prefix = trip_dir / trip_id

    track = _read_file(
        prefix.with_suffix(".track"), 3, dtype=float
    )
    nodes = _read_file(
        prefix.with_suffix(".nodes"), 2, dtype=float
    )
    arcs = _read_file(
        prefix.with_suffix(".arcs"), 2, dtype=np.int64
    )
    route = _read_file(
        prefix.with_suffix(".route"), 1, dtype=np.int64
    )

    track.columns = ["longitude", "latitude", "timestamp"]
    nodes.columns = ["longitude", "latitude"]
    arcs.columns = ["from_node_id", "to_node_id"]
    route.columns = ["arc_id"]

    # Check numerical validity.
    for name, frame in [
        ("track", track),
        ("nodes", nodes),
    ]:
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{trip_id}: invalid numeric values in {name}")

    if not np.isfinite(route.to_numpy(dtype=float)).all():
        raise ValueError(f"{trip_id}: invalid values in route")

    # Validate geographic coordinates.
    if not track["longitude"].between(-180, 180).all():
        raise ValueError(f"{trip_id}: invalid GPS longitude")

    if not track["latitude"].between(-90, 90).all():
        raise ValueError(f"{trip_id}: invalid GPS latitude")

    if not nodes["longitude"].between(-180, 180).all():
        raise ValueError(f"{trip_id}: invalid node longitude")

    if not nodes["latitude"].between(-90, 90).all():
        raise ValueError(f"{trip_id}: invalid node latitude")

    # Node IDs are zero-based row positions in the nodes file.
    if arcs[["from_node_id", "to_node_id"]].min().min() < 0:
        raise ValueError(f"{trip_id}: negative node ID in arcs")

    if arcs[["from_node_id", "to_node_id"]].max().max() >= len(nodes):
        raise ValueError(f"{trip_id}: arc references a missing node")

    # Arc IDs are zero-based row positions in the arcs file.
    if route["arc_id"].min() < 0 or route["arc_id"].max() >= len(arcs):
        raise ValueError(f"{trip_id}: route references a missing arc")

    if not track["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{trip_id}: GPS timestamps are not chronological")

    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def load_partition(partition="train"):
    """Load all trips in a dataset partition."""

    partition_dir = DATA_DIR / partition

    if not partition_dir.exists():
        raise FileNotFoundError(partition_dir)

    trip_dirs = sorted(
        path for path in partition_dir.iterdir() if path.is_dir()
    )

    trips = []
    for trip_dir in trip_dirs:
        trips.append(load_trip(trip_dir.name, partition))

    return trips


if __name__ == "__main__":
    for partition in ["train", "student_test"]:
        trips = load_partition(partition)
        print(f"\n{partition.upper()}: {len(trips)} trips")

        for trip in trips[:3]:
            print(
                trip["trip_id"],
                "| GPS:", len(trip["track"]),
                "| Nodes:", len(trip["nodes"]),
                "| Arcs:", len(trip["arcs"]),
                "| Route:", len(trip["route"]),
            )

    print("\nDATA LOADER VALIDATION PASSED")
