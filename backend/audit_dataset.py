from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.data_loader import load_partition

def audit_trip(trip):
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]["arc_id"].to_numpy()

    timestamps = track["timestamp"].to_numpy()
    gaps = np.diff(timestamps)

    arc_array = arcs.to_numpy(dtype=np.int64)
    route_arcs = arc_array[route]

    continuous = (
        route_arcs[:-1, 1] == route_arcs[1:, 0]
        if len(route_arcs) > 1 else np.array([], dtype=bool)
    )

    used_nodes = np.unique(arc_array)
    isolated = len(nodes) - len(used_nodes)

    return {
        "trip": trip["trip_id"],
        "gps": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route": len(route),
        "route_continuity_pct": (
            round(100 * continuous.mean(), 2)
            if len(continuous) else None
        ),
        "isolated_nodes": isolated,
        "timestamp_gaps": int(np.sum(gaps > 1)),
        "max_timestamp_gap": float(gaps.max()) if len(gaps) else 0,
        "gps_lon_min": track["longitude"].min(),
        "gps_lon_max": track["longitude"].max(),
        "gps_lat_min": track["latitude"].min(),
        "gps_lat_max": track["latitude"].max(),
    }


if __name__ == "__main__":
    for partition in ["train", "student_test"]:
        trips = load_partition(partition)
        results = pd.DataFrame([audit_trip(t) for t in trips])

        print("\n" + "=" * 65)
        print(partition.upper(), "DATA QUALITY AUDIT")
        print("=" * 65)
        print(results.to_string(index=False))

        print("\nSUMMARY")
        print(results.describe(include="all").to_string())

        output = Path("backend/experiments")
        output.mkdir(parents=True, exist_ok=True)
        results.to_csv(output / f"{partition}_data_audit.csv", index=False)

    print("\nAUDIT COMPLETE. Reports saved in backend/experiments/")
