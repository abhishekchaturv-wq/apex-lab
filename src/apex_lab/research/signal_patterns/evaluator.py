"""Evaluation of candidate signal rules against historical data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from apex_lab.research.signal_discovery.statistics import discretize_series
from apex_lab.research.signal_patterns.candidate_generator import CandidateRule

# Default forward-return target used when the caller does not specify one.
DEFAULT_TARGET = "future_return_20"
# MFE/MAE look for high/low returns if available.
_MFE_COL = "future_high_return"
_MAE_COL = "future_low_return"

EPSILON = 1e-12


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _profit_factor(returns: np.ndarray) -> float | None:
    gross_win = float(returns[returns > 0].sum())
    gross_loss = float(abs(returns[returns < 0].sum()))
    if gross_loss <= EPSILON:
        return None
    pf = gross_win / gross_loss
    return pf if math.isfinite(pf) else None


def _mask_for_rule(df: pl.DataFrame, rule: CandidateRule, bins: int) -> pl.Series:
    """Return a boolean mask identifying rows that match *rule*."""
    masks: list[pl.Series] = []
    bucket_values = rule.bucket_key.split("|")

    for feature, expected_bucket in zip(rule.features, bucket_values, strict=True):
        if feature not in df.columns:
            return pl.Series([False] * df.height, dtype=pl.Boolean)
        buckets = discretize_series(df.get_column(feature), bins=bins)
        masks.append(buckets == expected_bucket)

    if not masks:
        return pl.Series([False] * df.height, dtype=pl.Boolean)

    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask
    return combined


def evaluate_rule(
    df: pl.DataFrame,
    rule: CandidateRule,
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> dict[str, Any] | None:
    """Evaluate a single candidate rule and return its statistics.

    Returns ``None`` when there are not enough matching rows.
    """
    if target_column not in df.columns:
        return None

    mask = _mask_for_rule(df, rule, bins=bins)
    matched = df.filter(mask)

    if matched.height < 3:
        return None

    returns = np.asarray(matched.get_column(target_column).drop_nulls().to_list(), dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    n = int(returns.size)
    if n < 3:
        return None

    wins = int((returns > 0).sum())
    win_rate = wins / n
    avg_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    pf = _profit_factor(returns)

    avg_win = float(np.mean(returns[returns > 0])) if wins > 0 else 0.0
    avg_loss = float(np.mean(returns[returns < 0])) if (n - wins) > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

    # MFE / MAE (use dedicated columns when present; fall back to target)
    mfe_col = _MFE_COL if _MFE_COL in df.columns else target_column
    mae_col = _MAE_COL if _MAE_COL in df.columns else target_column

    mfe_vals = np.asarray(
        matched.get_column(mfe_col).drop_nulls().to_list(), dtype=np.float64
    )
    mfe_vals = mfe_vals[np.isfinite(mfe_vals)]
    mae_vals = np.asarray(
        matched.get_column(mae_col).drop_nulls().to_list(), dtype=np.float64
    )
    mae_vals = mae_vals[np.isfinite(mae_vals)]

    avg_mfe = _safe_float(float(np.mean(mfe_vals))) if mfe_vals.size > 0 else None
    avg_mae = _safe_float(float(np.mean(mae_vals))) if mae_vals.size > 0 else None

    return {
        "rule_label": rule.label(),
        "features": list(rule.features),
        "conditions": list(rule.conditions),
        "combination_size": len(rule.features),
        "signal_frequency": n,
        "win_rate": round(win_rate, 6),
        "average_return": round(avg_return, 6),
        "median_return": round(median_return, 6),
        "profit_factor": _safe_float(pf),
        "expectancy": round(expectancy, 6),
        "average_mfe": avg_mfe,
        "average_mae": avg_mae,
    }


def precompute_bucket_columns(
    df: pl.DataFrame,
    features: list[str],
    bins: int = 4,
) -> pl.DataFrame:
    """Add ``_b_{feature}`` bucket columns for every feature in *features*.

    Each column is computed exactly once.  Features missing from *df* are
    silently skipped.  Already-present bucket columns are left unchanged.
    """
    new_cols: list[pl.Series] = []
    for feature in features:
        col_name = f"_b_{feature}"
        if feature in df.columns and col_name not in df.columns:
            buckets = discretize_series(df.get_column(feature), bins=bins)
            new_cols.append(buckets.rename(col_name))
    return df.with_columns(new_cols) if new_cols else df


def _group_stats(
    df: pl.DataFrame,
    bucket_cols: list[str],
    target_col: str,
) -> pl.DataFrame:
    """Group by *bucket_cols* and compute all signal statistics in one pass.

    Rows with non-finite (null, NaN, or infinite) target values are excluded
    before aggregation.  Groups with fewer than 3 valid samples are dropped.

    Returns a DataFrame whose column names all start with ``_`` to avoid
    collisions with the caller's schema.
    """
    mfe_col = _MFE_COL if _MFE_COL in df.columns else target_col
    mae_col = _MAE_COL if _MAE_COL in df.columns else target_col

    t = pl.col(target_col)
    mfe = pl.col(mfe_col)
    mae = pl.col(mae_col)

    # Exclude rows where the target is null, NaN, or infinite.
    df_clean = df.filter(t.is_not_null() & t.is_finite())
    if df_clean.is_empty():
        return pl.DataFrame()

    grouped = (
        df_clean.group_by(bucket_cols)
        .agg(
            [
                pl.len().alias("_n"),
                pl.when(t > 0).then(pl.lit(1)).otherwise(pl.lit(0)).sum().alias("_wins"),
                t.mean().alias("_avg_ret"),
                t.median().alias("_med_ret"),
                pl.when(t > 0).then(t).otherwise(pl.lit(0.0)).sum().alias("_gross_win"),
                pl.when(t < 0).then(-t).otherwise(pl.lit(0.0)).sum().alias("_gross_loss"),
                # avg_win / avg_loss: null when there are no wins / no losses.
                pl.when(t > 0).then(t).otherwise(None).mean().alias("_avg_win"),
                pl.when(t < 0).then(t).otherwise(None).mean().alias("_avg_loss"),
                # avg_mfe / avg_mae: filter to finite values before averaging.
                pl.when(mfe.is_not_null() & mfe.is_finite()).then(mfe).otherwise(None).mean().alias("_avg_mfe"),
                pl.when(mae.is_not_null() & mae.is_finite()).then(mae).otherwise(None).mean().alias("_avg_mae"),
            ]
        )
        .filter(pl.col("_n") >= 3)
    )
    if grouped.is_empty():
        return pl.DataFrame()

    return (
        grouped.with_columns(
            [
                (pl.col("_wins") / pl.col("_n")).alias("_win_rate"),
                pl.when(pl.col("_gross_loss") > EPSILON)
                .then(pl.col("_gross_win") / pl.col("_gross_loss"))
                .otherwise(None)
                .alias("_profit_factor"),
            ]
        )
        .with_columns(
            (
                pl.col("_win_rate") * pl.col("_avg_win").fill_null(0.0)
                + (1.0 - pl.col("_win_rate")) * pl.col("_avg_loss").fill_null(0.0)
            ).alias("_expectancy")
        )
    )


def evaluate_all_candidates(
    df: pl.DataFrame,
    candidates: list[CandidateRule],
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> pl.DataFrame:
    """Evaluate every candidate rule and return a statistics DataFrame.

    Uses vectorised ``group_by`` aggregation: bucket columns are precomputed
    once per feature, and each unique feature combination is aggregated in a
    single pass instead of per-candidate row filtering.
    """
    if not candidates or target_column not in df.columns:
        return _empty_stats_frame()

    # 1. Collect all unique features across all candidates.
    all_features: list[str] = sorted(
        {f for rule in candidates for f in rule.features if f in df.columns}
    )

    # 2. Precompute bucket columns exactly once.
    df_b = precompute_bucket_columns(df, all_features, bins=bins)

    # 3. Group candidates by their feature combination for batch evaluation.
    combo_to_candidates: dict[tuple[str, ...], list[CandidateRule]] = {}
    for rule in candidates:
        combo_to_candidates.setdefault(rule.features, []).append(rule)

    rows: list[dict[str, Any]] = []

    # 4. One group_by pass per unique feature combination.
    for features_tuple, combo_rules in sorted(combo_to_candidates.items()):
        bucket_cols = [f"_b_{f}" for f in features_tuple]
        if not all(c in df_b.columns for c in bucket_cols):
            continue

        grouped = _group_stats(df_b, bucket_cols, target_column)
        if grouped.is_empty():
            continue

        # Build a lookup: bucket_key → stats row dict.
        lookup: dict[str, dict[str, Any]] = {}
        for row in grouped.to_dicts():
            key = "|".join(str(row[c]) for c in bucket_cols)
            lookup[key] = row

        # 5. Match each candidate rule to its group result.
        for rule in combo_rules:
            stats = lookup.get(rule.bucket_key)
            if stats is None:
                continue

            rows.append(
                {
                    "rule_label": rule.label(),
                    "features": str(list(rule.features)),
                    "conditions": str(list(rule.conditions)),
                    "combination_size": len(rule.features),
                    "signal_frequency": int(stats["_n"]),
                    "win_rate": round(float(stats["_win_rate"]), 6),
                    "average_return": round(float(stats["_avg_ret"]), 6),
                    "median_return": round(float(stats["_med_ret"]), 6),
                    "profit_factor": _safe_float(stats["_profit_factor"]),
                    "expectancy": round(float(stats["_expectancy"]), 6),
                    "average_mfe": _safe_float(stats["_avg_mfe"]),
                    "average_mae": _safe_float(stats["_avg_mae"]),
                }
            )

    if not rows:
        return _empty_stats_frame()

    return pl.DataFrame(
        {
            "rule_label": [r["rule_label"] for r in rows],
            "features": [r["features"] for r in rows],
            "conditions": [r["conditions"] for r in rows],
            "combination_size": [r["combination_size"] for r in rows],
            "signal_frequency": [r["signal_frequency"] for r in rows],
            "win_rate": [r["win_rate"] for r in rows],
            "average_return": [r["average_return"] for r in rows],
            "median_return": [r["median_return"] for r in rows],
            "profit_factor": [r["profit_factor"] for r in rows],
            "expectancy": [r["expectancy"] for r in rows],
            "average_mfe": [r["average_mfe"] for r in rows],
            "average_mae": [r["average_mae"] for r in rows],
        }
    )


def _empty_stats_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rule_label": pl.Series([], dtype=pl.Utf8),
            "features": pl.Series([], dtype=pl.Utf8),
            "conditions": pl.Series([], dtype=pl.Utf8),
            "combination_size": pl.Series([], dtype=pl.Int64),
            "signal_frequency": pl.Series([], dtype=pl.Int64),
            "win_rate": pl.Series([], dtype=pl.Float64),
            "average_return": pl.Series([], dtype=pl.Float64),
            "median_return": pl.Series([], dtype=pl.Float64),
            "profit_factor": pl.Series([], dtype=pl.Float64),
            "expectancy": pl.Series([], dtype=pl.Float64),
            "average_mfe": pl.Series([], dtype=pl.Float64),
            "average_mae": pl.Series([], dtype=pl.Float64),
        }
    )
