from pathlib import Path
import pandas as pd

root = Path("backend/data/raw/Student folder")
trip = sorted((root / "train").iterdir())[0]

for ext in ["track", "nodes", "arcs", "route"]:
    file = trip / f"{trip.name}.{ext}"
    df = pd.read_csv(file, sep=r"\s+", header=None)

    print("\n" + "=" * 45)
    print(f"FILE: {trip.name}.{ext}")
    print("SHAPE:", df.shape)
    print("SAMPLE:")
    print(df.head(5).to_string(index=False, header=False))
