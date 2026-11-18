from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "models"
RESULT_DIR = BACKEND_DIR / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def train_models(X, y, random_state=42):
    """
    Train Decision Tree and Random Forest using genuine
    candidate-level features and labels.

    X: pandas DataFrame of engineered features.
    y: corresponding training labels.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths")

    if len(X) == 0:
        raise ValueError("Training dataset is empty")

    if X.isnull().any().any():
        raise ValueError("X contains missing feature values")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("All model features must be numeric")

    if len(pd.Series(y).unique()) < 2:
        raise ValueError("Training requires at least two classes")

    feature_columns = list(X.columns)

    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    results = []

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X, y)
        training_time = time.perf_counter() - start

        predictions = model.predict(X)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y, predictions),
            "precision_macro": precision_score(
                y, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y, predictions, average="macro", zero_division=0
            ),
            "training_time_seconds": training_time,
        }

        artifact = {
            "model": model,
            "feature_columns": feature_columns,
            "model_name": name,
        }

        joblib.dump(artifact, MODEL_DIR / f"{name}.joblib")
        results.append(metrics)

        print(f"\n{name.upper()}")
        print(json.dumps(metrics, indent=2))
        print(classification_report(
            y, predictions, zero_division=0
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULT_DIR / "decision_tree_random_forest_training.csv",
        index=False,
    )

    return results_df


def load_model(model_name):
    """Load a saved model and its expected feature columns."""

    if model_name not in {"decision_tree", "random_forest"}:
        raise ValueError("Unknown model name")

    path = MODEL_DIR / f"{model_name}.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def predict(model_name, features):
    """Generate predictions using a saved model."""

    artifact = load_model(model_name)
    model = artifact["model"]
    columns = artifact["feature_columns"]

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    missing = set(columns) - set(features.columns)
    extra = set(features.columns) - set(columns)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    return model.predict(features[columns])


if __name__ == "__main__":
    print("Decision Tree and Random Forest training module")
    print("Waiting for genuine candidate features and labels from the ML pipeline.")
    print("No synthetic training data will be generated.")
