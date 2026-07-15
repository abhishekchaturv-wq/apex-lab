"""Feature importance utilities for baseline models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    random_state: int = 42,
    n_repeats: int = 8,
) -> pl.DataFrame:
    """Compute model-native and permutation feature importance."""
    estimator = _unwrap_estimator(model)

    model_importance = np.full(len(feature_names), np.nan, dtype=float)
    if hasattr(estimator, "feature_importances_"):
        model_importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        model_importance = np.abs(coef).mean(axis=0)

    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        random_state=random_state,
        n_repeats=n_repeats,
        scoring="average_precision",
    )

    importance_df = pl.DataFrame(
        {
            "feature": feature_names,
            "model_importance": model_importance,
            "permutation_importance_mean": permutation.importances_mean.astype(float),
            "permutation_importance_std": permutation.importances_std.astype(float),
        }
    )

    return importance_df.sort("permutation_importance_mean", descending=True)


def save_feature_importance_csv(feature_importance: pl.DataFrame, path: Path) -> Path:
    """Persist feature importance DataFrame as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.write_csv(path)
    return path


def _unwrap_estimator(model: object) -> object:
    """Extract inner estimator when model is a sklearn Pipeline."""
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        if "model" in steps:
            return steps["model"]
        last_name = next(reversed(steps))
        return steps[last_name]
    return model
