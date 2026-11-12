"""Training pipeline: load data -> train -> evaluate -> save model + metadata.

Run:  python -m src.train
"""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src import config
from src.logger import get_logger

logger = get_logger("pipeline.train")


def load_data():
    logger.info("Loading Breast Cancer Wisconsin dataset from %s", config.DATA_PATH)
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_PATH}. Place the Kaggle CSV there."
        )
    data = pd.read_csv(config.DATA_PATH)
    # Some Kaggle copies append an empty trailing CSV column.
    data = data.dropna(axis=1, how="all")
    required = {"id", "diagnosis"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data["diagnosis"].isna().any() or data.drop(
        columns=["id", "diagnosis"]
    ).isna().any().any():
        raise ValueError("Dataset contains missing target or feature values")

    X = data.drop(columns=["id", "diagnosis"])
    y = data["diagnosis"].map({"B": 0, "M": 1})
    if y.isna().any():
        raise ValueError("Diagnosis must contain only B (benign) or M (malignant)")
    return X, y.astype(int), ["benign", "malignant"]


def train_model() -> dict:
    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train size=%d, Test size=%d", len(X_train), len(X_test))

    logger.info("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    logger.info("Model accuracy: %.4f", accuracy)
    logger.info(
        "\n%s", classification_report(y_test, predictions, target_names=target_names)
    )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    metadata = {
        "model": "RandomForestClassifier",
        "dataset": "Breast Cancer Wisconsin Diagnostic",
        "target": "diagnosis",
        "target_mapping": {"0": "benign", "1": "malignant"},
        "accuracy": accuracy,
        "n_estimators": config.N_ESTIMATORS,
        "features": list(X.columns),
        "classes": target_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info("Model saved to %s", config.MODEL_PATH)
    return metadata


if __name__ == "__main__":
    train_model()
