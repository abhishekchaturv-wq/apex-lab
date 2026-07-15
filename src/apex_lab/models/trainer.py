"""Training utilities for baseline predictive reversal models."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apex_lab.models.importance import compute_feature_importance, save_feature_importance_csv
from apex_lab.models.metrics import evaluate_binary_classifier

MODEL_REGISTRY = {
    "logistic_regression": lambda random_state: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1_000, random_state=random_state)),
        ]
    ),
    "random_forest": lambda random_state: RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    ),
    "gradient_boosting": lambda random_state: GradientBoostingClassifier(random_state=random_state),
}


@dataclass(frozen=True)
class ModelOutputPaths:
    """File paths for persisted model outputs."""

    model_artifact: Path
    metrics_json: Path
    feature_importance_csv: Path
    prediction_probabilities_csv: Path


@dataclass(frozen=True)
class TrainingResult:
    """Training outputs for baseline classification models."""

    model_name: str
    model: object
    feature_names: list[str]
    class_mapping: dict[str, int]
    metrics: dict[str, Any]
    feature_importance: pl.DataFrame
    prediction_probabilities: pl.DataFrame
    paths: ModelOutputPaths | None


def train_baseline_model(
    dataset: pl.DataFrame,
    *,
    target_column: str,
    model_name: str = "logistic_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Path | None = None,
    test_df: pl.DataFrame | None = None,
) -> TrainingResult:
    """Train one baseline model and return metrics plus output artifacts.

    Args:
        dataset: Training data (or full dataset when *test_df* is ``None``).
        target_column: Binary target column name.
        model_name: Key in :data:`MODEL_REGISTRY`.
        test_size: Fraction of data for testing when *test_df* is ``None``.
        random_state: Seed for model and (legacy) random split.
        output_dir: Optional directory to persist artifacts.
        test_df: Pre-split test DataFrame.  When provided, *dataset* is used
            only for training and ``train_test_split`` is **not** called,
            preserving chronological integrity.
    """
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    feature_names = _select_feature_columns(dataset, target_column=target_column)
    if not feature_names:
        raise ValueError("No numeric feature columns available for training")

    working_train = dataset.select(feature_names + [target_column]).drop_nulls()
    if len(working_train) < 20:
        raise ValueError("Dataset has too few non-null rows for training")

    x = working_train.select(feature_names).to_numpy()
    y_raw = working_train[target_column].to_numpy()
    y, class_mapping = _encode_binary_labels(y_raw)

    estimator = _build_model(model_name=model_name, random_state=random_state)

    if test_df is not None:
        # Chronological split supplied by caller – skip random shuffle entirely.
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in test_df")
        working_test = test_df.select(feature_names + [target_column]).drop_nulls()
        x_train, y_train = x, y
        x_test = working_test.select(feature_names).to_numpy()
        y_test_raw = working_test[target_column].to_numpy()
        y_test = _apply_class_mapping(y_test_raw, class_mapping)
    else:
        # Legacy behaviour: random stratified split.
        stratify = y if _can_stratify(y) else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(x_test)
    y_prob = estimator.predict_proba(x_test)[:, 1]

    metrics = evaluate_binary_classifier(y_test, y_pred, y_prob)
    feature_importance = compute_feature_importance(estimator, x_test, y_test, feature_names, random_state=random_state)

    prediction_probabilities = pl.DataFrame(
        {
            "y_true": y_test.astype(int),
            "y_pred": np.asarray(y_pred, dtype=int),
            "probability": np.asarray(y_prob, dtype=float),
        }
    )

    paths = None
    if output_dir is not None:
        paths = _persist_outputs(
            model=estimator,
            metrics=metrics,
            feature_importance=feature_importance,
            prediction_probabilities=prediction_probabilities,
            output_dir=output_dir,
            model_name=model_name,
        )

    return TrainingResult(
        model_name=model_name,
        model=estimator,
        feature_names=feature_names,
        class_mapping=class_mapping,
        metrics=metrics,
        feature_importance=feature_importance,
        prediction_probabilities=prediction_probabilities,
        paths=paths,
    )


def _build_model(*, model_name: str, random_state: int) -> object:
    """Build a model from the baseline model registry."""
    if model_name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY)
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name](random_state)


def _select_feature_columns(dataset: pl.DataFrame, *, target_column: str) -> list[str]:
    """Select numeric feature columns excluding target."""
    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }

    return [
        column
        for column, dtype in dataset.schema.items()
        if column != target_column and dtype in numeric_dtypes
    ]


def _encode_binary_labels(y_raw: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Encode binary labels into {0, 1} target values."""
    values = [value.item() if isinstance(value, np.generic) else value for value in y_raw]
    unique_values = sorted({str(value) for value in values})

    if len(unique_values) != 2:
        raise ValueError(f"Binary classification requires exactly two classes. Found: {unique_values}")

    mapping = {label: idx for idx, label in enumerate(unique_values)}
    encoded = np.asarray([mapping[str(value)] for value in values], dtype=int)
    return encoded, mapping


def _apply_class_mapping(y_raw: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """Encode labels using an existing class mapping (no uniqueness check)."""
    values = [value.item() if isinstance(value, np.generic) else value for value in y_raw]
    return np.asarray([mapping[str(value)] for value in values], dtype=int)


def _can_stratify(y: np.ndarray) -> bool:
    """Return True when labels can support stratified splitting."""
    _, counts = np.unique(y, return_counts=True)
    return counts.min() > 1


def _persist_outputs(
    *,
    model: object,
    metrics: dict[str, Any],
    feature_importance: pl.DataFrame,
    prediction_probabilities: pl.DataFrame,
    output_dir: Path,
    model_name: str,
) -> ModelOutputPaths:
    """Persist model artifact and baseline evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.pkl"
    metrics_path = output_dir / "metrics.json"
    importance_path = output_dir / "feature_importance.csv"
    probabilities_path = output_dir / "prediction_probabilities.csv"

    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_feature_importance_csv(feature_importance, importance_path)
    prediction_probabilities.write_csv(probabilities_path)

    return ModelOutputPaths(
        model_artifact=model_path,
        metrics_json=metrics_path,
        feature_importance_csv=importance_path,
        prediction_probabilities_csv=probabilities_path,
    )
