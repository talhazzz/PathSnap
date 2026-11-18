from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
]


def load_model(model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")

    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def get_feature_columns(model_name, artifact):
    if isinstance(artifact, Pipeline):
        return list(artifact.feature_names_in_)

    if isinstance(artifact, dict):
        if "features" in artifact:
            return list(artifact["features"])
        if "feature_columns" in artifact:
            return list(artifact["feature_columns"])

    raise ValueError(f"Cannot determine feature columns for {model_name}")


def predict_model(model_name, features):
    artifact = load_model(model_name)
    columns = get_feature_columns(model_name, artifact)

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame")

    missing = set(columns) - set(features.columns)
    extra = set(features.columns) - set(columns)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    frame = features[columns]

    if isinstance(artifact, Pipeline):
        model = artifact
        predictions = model.predict(frame)
        probability_input = frame

    elif model_name == "knn":
        model = artifact["model"]
        probability_input = artifact["scaler"].transform(frame)
        predictions = model.predict(probability_input)

    else:
        model = artifact["model"]
        predictions = model.predict(frame)
        probability_input = frame

    probabilities = None

    if hasattr(model, "predict_proba"):
        all_probabilities = model.predict_proba(probability_input)
        classes = list(model.classes_)

        if 1 in classes:
            probabilities = all_probabilities[:, classes.index(1)].tolist()

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities,
        "feature_columns": columns,
    }
