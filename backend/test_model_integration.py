import numpy as np
import pandas as pd

from src.train_models import train_models, load_model, predict

# Temporary test data: software integration only.
rng = np.random.default_rng(42)

X = pd.DataFrame({
    "distance_m": rng.uniform(0, 50, 100),
    "heading_difference": rng.uniform(0, 180, 100),
    "speed_mps": rng.uniform(0, 30, 100),
})

y = (
    (X["distance_m"] < 20)
    & (X["heading_difference"] < 90)
).astype(int)

print("TEST ONLY — NOT HACKATHON TRAINING RESULTS")

results = train_models(X, y)

for name in ["decision_tree", "random_forest"]:
    artifact = load_model(name)
    predictions = predict(name, X.head(5))

    print(f"\n{name}")
    print("Saved model:", artifact["model_name"])
    print("Test predictions:", predictions.tolist())

    assert len(predictions) == 5

print("\nMODEL SAVE / LOAD / PREDICTION TEST PASSED")

