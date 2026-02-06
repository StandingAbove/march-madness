from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


@dataclass
class ModelArtifacts:
    model: HistGradientBoostingClassifier
    feature_columns: list[str]


def train_model(dataset: pd.DataFrame, random_state: int = 42) -> ModelArtifacts:
    """Train a gradient boosting model on matchup data."""
    dataset = dataset.copy()
    y = dataset.pop("Result")
    dataset = dataset.drop(columns=["Season"], errors="ignore")
    feature_columns = dataset.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        dataset, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.3f}")

    return ModelArtifacts(model=model, feature_columns=feature_columns)


def save_model(artifacts: ModelArtifacts, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": artifacts.model, "features": artifacts.feature_columns}, output_path
    )


def load_model(model_path: Path) -> ModelArtifacts:
    payload = joblib.load(model_path)
    return ModelArtifacts(model=payload["model"], feature_columns=payload["features"])
