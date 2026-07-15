"""Inference helpers for serialized baseline models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl


def load_model(path: Path) -> object:
    """Load a serialized model artifact from disk."""
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted local artifact


def predict_probabilities(
    model: object,
    features: pl.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Generate positive-class prediction probabilities."""
    x_matrix = _to_feature_matrix(features, feature_names=feature_names)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba for probability inference")

    proba = model.predict_proba(x_matrix)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("predict_proba output must contain at least two class probability columns")

    return np.asarray(proba[:, 1], dtype=float)


def predict_from_artifact(
    model_path: Path,
    features: pl.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
    output_path: Path | None = None,
) -> np.ndarray:
    """Load model artifact and run probability inference."""
    model = load_model(model_path)
    probabilities = predict_probabilities(model, features, feature_names=feature_names)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"probability": probabilities}).write_csv(output_path)

    return probabilities


def _to_feature_matrix(
    features: pl.DataFrame | np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Convert feature input into numpy matrix suitable for sklearn models."""
    if isinstance(features, np.ndarray):
        return np.asarray(features, dtype=float)

    selected = features.select(feature_names) if feature_names is not None else features
    return selected.to_numpy()
