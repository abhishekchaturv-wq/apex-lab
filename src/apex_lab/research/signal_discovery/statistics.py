"""Statistical helpers for signal discovery analysis."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

_EPSILON = 1e-12


def normalize_series(values: list[float | None]) -> list[float]:
    """Min-max normalize numeric values; nulls become 0."""
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return [0.0 for _ in values]
    minimum = min(clean)
    maximum = max(clean)
    span = maximum - minimum
    if span <= 0:
        return [0.0 for _ in values]
    return [
        0.0
        if value is None or not math.isfinite(float(value))
        else (float(value) - minimum) / span
        for value in values
    ]


def entropy(values: Iterable[str]) -> float:
    """Compute Shannon entropy for categorical values."""
    items = list(values)
    if not items:
        return 0.0
    counts = Counter(items)
    total = float(len(items))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def discretize_numeric(values: np.ndarray, bins: int = 5) -> np.ndarray:
    """Discretize numeric values into quantile bins."""
    if values.size == 0:
        return np.array([], dtype=np.int32)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size <= 2:
        return np.zeros(values.size, dtype=np.int32)
    return np.digitize(values, edges[1:-1], right=True).astype(np.int32)


def discretize_series(series: pl.Series, bins: int = 4) -> pl.Series:
    """Convert a Polars series to stable string buckets for grouping."""
    if series.dtype.is_numeric():
        array = np.array(series.to_list(), dtype=np.float64)
        mask = np.isfinite(array)
        bucketed = np.full(series.len(), "null", dtype=object)
        if mask.any():
            labels = discretize_numeric(array[mask], bins=bins)
            bucketed[np.where(mask)[0]] = np.array([f"q{label}" for label in labels], dtype=object)
        return pl.Series(series.name, bucketed, dtype=pl.Utf8)
    return series.cast(pl.Utf8, strict=False).fill_null("null")


def information_gain(feature: pl.Series, target: pl.Series, bins: int = 5) -> float | None:
    """Estimate information gain of feature on a categorical target."""
    pair = pl.DataFrame({"feature": feature, "target": target}).drop_nulls()
    if pair.height < 3:
        return None

    target_values = [str(value) for value in pair.get_column("target").to_list()]
    base_entropy = entropy(target_values)
    if base_entropy <= 0:
        return 0.0

    feat = pair.get_column("feature")
    if feat.dtype.is_numeric():
        feat_values = np.array(feat.to_list(), dtype=np.float64)
        discrete = discretize_numeric(feat_values, bins=bins)
        feature_values = [str(value) for value in discrete.tolist()]
    else:
        feature_values = [str(value) for value in feat.cast(pl.Utf8, strict=False).to_list()]

    grouped_targets: dict[str, list[str]] = {}
    for feat_value, target_value in zip(feature_values, target_values, strict=True):
        grouped_targets.setdefault(feat_value, []).append(target_value)

    total = float(len(target_values))
    conditional_entropy = sum((len(values) / total) * entropy(values) for values in grouped_targets.values())
    gain = base_entropy - conditional_entropy
    return max(0.0, float(gain))


def _prepare_feature_target(
    feature: pl.Series,
    target: pl.Series,
) -> tuple[np.ndarray, np.ndarray] | None:
    pair = pl.DataFrame({"feature": feature, "target": target}).drop_nulls()
    if pair.height < 3:
        return None
    return pair.get_column("feature").to_numpy(), pair.get_column("target").to_numpy()


def mutual_information(
    feature: pl.Series,
    target: pl.Series,
    *,
    target_is_classification: bool,
) -> float | None:
    """Compute mutual information between one feature and one target."""
    prepared = _prepare_feature_target(feature, target)
    if prepared is None:
        return None
    feature_array, target_array = prepared

    if feature.dtype.is_numeric():
        x = np.asarray(feature_array, dtype=np.float64).reshape(-1, 1)
        discrete_features = False
    else:
        _, encoded = np.unique(feature_array.astype(str), return_inverse=True)
        x = encoded.reshape(-1, 1)
        discrete_features = True

    if target_is_classification:
        _, encoded_target = np.unique(target_array.astype(str), return_inverse=True)
        mi = mutual_info_classif(
            x,
            encoded_target,
            discrete_features=discrete_features,
            random_state=0,
        )
    else:
        y = np.asarray(target_array, dtype=np.float64)
        mi = mutual_info_regression(
            x,
            y,
            discrete_features=discrete_features,
            random_state=0,
        )

    if mi.size == 0:
        return None
    value = float(mi[0])
    return value if math.isfinite(value) else None


def correlation(feature: pl.Series, target: pl.Series, method: str) -> float | None:
    """Compute Pearson or Spearman correlation."""
    pair = pl.DataFrame({"feature": feature, "target": target}).drop_nulls()
    if pair.height < 3:
        return None

    x = np.asarray(pair.get_column("feature").to_list(), dtype=np.float64)
    y = np.asarray(pair.get_column("target").to_list(), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None

    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) <= _EPSILON or np.std(y_valid) <= _EPSILON:
        return 0.0

    if method == "pearson":
        corr, _ = pearsonr(x_valid, y_valid)
    elif method == "spearman":
        corr, _ = spearmanr(x_valid, y_valid)
    else:
        raise ValueError(f"unsupported correlation method: {method}")

    if not math.isfinite(float(corr)):
        return None
    return float(corr)


def predictive_power(feature: pl.Series, target: pl.Series, *, categorical_feature: bool) -> float | None:
    """Estimate predictive power as variance explained ratio in [0, 1]."""
    pair = pl.DataFrame({"feature": feature, "target": target}).drop_nulls()
    if pair.height < 3:
        return None

    target_values = np.array(pair.get_column("target").to_list(), dtype=np.float64)
    total_var = float(np.var(target_values))
    if total_var <= _EPSILON:
        return 0.0

    if categorical_feature:
        group_labels = pair.get_column("feature").cast(pl.Utf8, strict=False).fill_null("null").to_list()
    else:
        feature_values = np.array(pair.get_column("feature").to_list(), dtype=np.float64)
        labels = discretize_numeric(feature_values, bins=5)
        group_labels = [str(label) for label in labels.tolist()]

    grouped: dict[str, list[float]] = {}
    for label, value in zip(group_labels, target_values.tolist(), strict=True):
        grouped.setdefault(label, []).append(value)

    overall_mean = float(np.mean(target_values))
    between_variance = 0.0
    total = float(target_values.size)

    for values in grouped.values():
        if not values:
            continue
        group_mean = float(np.mean(values))
        between_variance += len(values) * ((group_mean - overall_mean) ** 2)

    explained = between_variance / (total * total_var)
    return float(max(0.0, min(1.0, explained)))


def stability_label(score: float) -> str:
    """Convert numeric stability score to label."""
    if score >= 0.75:
        return "Stable"
    if score >= 0.5:
        return "Moderately Stable"
    return "Unstable"
