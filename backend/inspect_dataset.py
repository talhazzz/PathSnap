from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
root = BACKEND_DIR / "data" / "raw" / "Student folder"

for partition in ["train", "student_test"]:
    folder = root / partition
    if not folder.is_dir():
        print(f"\n{partition.upper()}: NOT PRESENT ({folder})")
        continue
    trips = sorted(p for p in folder.iterdir() if p.is_dir())

    print(f"\n{partition.upper()}")
    print("Trip count:", len(trips))

    for trip in trips[:3]:
        print("Trip:", trip.name)
        for ext in ["track", "nodes", "arcs", "route"]:
            f = trip / f"{trip.name}.{ext}"
            print(f"  {ext}: {'FOUND' if f.exists() else 'MISSING'}")
