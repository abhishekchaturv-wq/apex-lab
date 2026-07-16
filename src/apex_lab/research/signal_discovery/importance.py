"""Feature importance and categorical signal analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from apex_lab.research.signal_discovery.statistics import (
    correlation,
    information_gain,
    mutual_information,
    normalize_series,
    predictive_power,
)

RETURN_TARGETS: tuple[str, ...] = (
    "future_return_5",
    "future_return_10",
    "future_return_20",
    "future_return_40",
)
CLASS_TARGET = "signal_class"
ALL_TARGETS: tuple[str, ...] = (*RETURN_TARGETS, CLASS_TARGET)


_EMPTY_IMPORTANCE_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "feature": pl.Utf8,
    "is_categorical": pl.Boolean,
    "information_gain": pl.Float64,
    "mutual_information": pl.Float64,
    "pearson": pl.Float64,
    "spearman": pl.Float64,
    "predictive_power": pl.Float64,
    "consistency": pl.Float64,
    "robustness": pl.Float64,
    "mi_score": pl.Float64,
    "corr_score": pl.Float64,
    "predictive_score": pl.Float64,
    "consistency_score": pl.Float64,
    "robustness_score": pl.Float64,
    "composite_score": pl.Float64,
}


_CATEGORICAL_HINTS = (
    "bucket",
    "state",
    "regime",
    "hour",
    "day",
    "month",
    "quarter",
    "opening_range",
    "or_",
    "gap",
)
# Composite feature-importance weighting: information and predictive power are
# prioritized, with consistency/robustness as secondary tie-breakers.
_MI_WEIGHT = 0.30
_CORR_WEIGHT = 0.20
_PREDICTIVE_WEIGHT = 0.25
_CONSISTENCY_WEIGHT = 0.15
_ROBUSTNESS_WEIGHT = 0.10


def is_categorical_feature(series: pl.Series, name: str) -> bool:
    """Detect whether a feature should be treated as categorical."""
    if series.dtype in {pl.Utf8, pl.Categorical, pl.Boolean}:
        return True
    lowered = name.lower()
    if any(token in lowered for token in _CATEGORICAL_HINTS):
        return True

    if series.dtype.is_integer() and series.n_unique() <= 16:
        return True
    return False


def analyze_feature_importance(
    df: pl.DataFrame,
    feature_columns: list[str],
) -> pl.DataFrame:
    """Compute feature importance table for all input features."""
    if not feature_columns:
        return pl.DataFrame(schema=_EMPTY_IMPORTANCE_SCHEMA)

    rows: list[dict[str, Any]] = []

    for feature_name in feature_columns:
        feature = df.get_column(feature_name)
        categorical = is_categorical_feature(feature, feature_name)

        mi_values: list[float] = []
        pearson_values: list[float] = []
        spearman_values: list[float] = []
        power_values: list[float] = []
        signed_corr: list[float] = []

        for target_name in ALL_TARGETS:
            if target_name not in df.columns:
                continue
            target = df.get_column(target_name)
            target_is_classification = target_name == CLASS_TARGET

            mi = mutual_information(feature, target, target_is_classification=target_is_classification)
            if mi is not None:
                mi_values.append(abs(mi))

            power = predictive_power(feature, target, categorical_feature=categorical)
            if power is not None:
                power_values.append(power)

            if target_name in RETURN_TARGETS and feature.dtype.is_numeric():
                pearson = correlation(feature, target, method="pearson")
                spearman = correlation(feature, target, method="spearman")
                if pearson is not None:
                    pearson_values.append(abs(pearson))
                    signed_corr.append(pearson)
                if spearman is not None:
                    spearman_values.append(abs(spearman))

        info_gain = (
            information_gain(feature, df.get_column(CLASS_TARGET))
            if CLASS_TARGET in df.columns
            else None
        )
        null_ratio = feature.null_count() / max(1, feature.len())

        if len(signed_corr) >= 2:
            signs = np.sign(np.asarray(signed_corr, dtype=np.float64))
            consistency = float(np.mean(signs == signs[0]))
        elif signed_corr:
            consistency = 1.0
        else:
            consistency = 0.0

        robustness = float(max(0.0, 1.0 - null_ratio))

        rows.append(
            {
                "feature": feature_name,
                "is_categorical": categorical,
                "information_gain": info_gain,
                "mutual_information": float(np.mean(mi_values)) if mi_values else None,
                "pearson": float(np.mean(pearson_values)) if pearson_values else None,
                "spearman": float(np.mean(spearman_values)) if spearman_values else None,
                "predictive_power": float(np.mean(power_values)) if power_values else None,
                "consistency": consistency,
                "robustness": robustness,
            }
        )

    importance = pl.DataFrame(rows)

    mi_norm = normalize_series(importance.get_column("mutual_information").to_list())
    corr_norm = normalize_series(
        importance.select(
            ((pl.col("pearson").fill_null(0.0) + pl.col("spearman").fill_null(0.0)) / 2.0).alias("corr_avg")
        )
        .get_column("corr_avg")
        .to_list()
    )
    predictive_norm = normalize_series(importance.get_column("predictive_power").to_list())
    consistency_norm = normalize_series(importance.get_column("consistency").to_list())
    robustness_norm = normalize_series(importance.get_column("robustness").to_list())

    scored = importance.with_columns(
        [
            pl.Series("mi_score", mi_norm, dtype=pl.Float64),
            pl.Series("corr_score", corr_norm, dtype=pl.Float64),
            pl.Series("predictive_score", predictive_norm, dtype=pl.Float64),
            pl.Series("consistency_score", consistency_norm, dtype=pl.Float64),
            pl.Series("robustness_score", robustness_norm, dtype=pl.Float64),
        ]
    ).with_columns(
        (
            (pl.col("mi_score") * _MI_WEIGHT)
            + (pl.col("corr_score") * _CORR_WEIGHT)
            + (pl.col("predictive_score") * _PREDICTIVE_WEIGHT)
            + (pl.col("consistency_score") * _CONSISTENCY_WEIGHT)
            + (pl.col("robustness_score") * _ROBUSTNESS_WEIGHT)
        ).alias("composite_score")
    )

    return (
        scored.sort(["composite_score", "mutual_information"], descending=[True, True])
        .with_row_index(name="rank", offset=1)
        .with_columns(
            [
                pl.col("information_gain").round(6),
                pl.col("mutual_information").round(6),
                pl.col("pearson").round(6),
                pl.col("spearman").round(6),
                pl.col("predictive_power").round(6),
                pl.col("consistency").round(6),
                pl.col("robustness").round(6),
                pl.col("composite_score").round(6),
            ]
        )
    )


def _default_categorical_columns(df: pl.DataFrame, feature_columns: list[str]) -> list[str]:
    columns: list[str] = []
    for name in feature_columns:
        series = df.get_column(name)
        if is_categorical_feature(series, name):
            columns.append(name)

    for fallback in ("market_regime", "hour", "day", "month", "quarter"):
        if fallback in df.columns and fallback not in columns:
            columns.append(fallback)
    return columns


def analyze_categorical_features(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_column: str = "future_return_20",
) -> pl.DataFrame:
    """Compute category-level win rate and expectancy for categorical features."""
    if target_column not in df.columns:
        return pl.DataFrame(
            {
                "feature": pl.Series([], dtype=pl.Utf8),
                "category": pl.Series([], dtype=pl.Utf8),
                "trade_frequency": pl.Series([], dtype=pl.Int64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
            }
        )

    rows: list[pl.DataFrame] = []
    categorical_columns = _default_categorical_columns(df, feature_columns)

    for column in categorical_columns:
        frame = (
            df.select(
                [
                    pl.col(column).cast(pl.Utf8, strict=False).fill_null("null").alias("category"),
                    pl.col(target_column).alias("target"),
                ]
            )
            .drop_nulls("target")
            .group_by("category")
            .agg(
                [
                    pl.len().alias("trade_frequency"),
                    pl.col("target").mean().alias("average_return"),
                    (pl.col("target") > 0.0).mean().alias("win_rate"),
                ]
            )
            .with_columns(
                [
                    pl.lit(column).alias("feature"),
                    pl.col("average_return").alias("expectancy"),
                ]
            )
            .select(
                [
                    "feature",
                    "category",
                    "trade_frequency",
                    "win_rate",
                    "average_return",
                    "expectancy",
                ]
            )
        )
        rows.append(frame)

    if not rows:
        return pl.DataFrame(
            {
                "feature": pl.Series([], dtype=pl.Utf8),
                "category": pl.Series([], dtype=pl.Utf8),
                "trade_frequency": pl.Series([], dtype=pl.Int64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
            }
        )

    combined = pl.concat(rows, how="vertical_relaxed")
    return (
        combined.sort(["expectancy", "trade_frequency"], descending=[True, True])
        .with_columns(
            [
                pl.col("win_rate").round(6),
                pl.col("average_return").round(6),
                pl.col("expectancy").round(6),
            ]
        )
    )
