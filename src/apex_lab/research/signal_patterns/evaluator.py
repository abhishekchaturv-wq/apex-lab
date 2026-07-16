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


def evaluate_all_candidates(
    df: pl.DataFrame,
    candidates: list[CandidateRule],
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> pl.DataFrame:
    """Evaluate every candidate rule and return a statistics DataFrame."""
    rows: list[dict[str, Any]] = []
    for rule in candidates:
        result = evaluate_rule(df, rule, target_column=target_column, bins=bins)
        if result is not None:
            rows.append(result)

    if not rows:
        return _empty_stats_frame()

    table = pl.DataFrame(
        {
            "rule_label": [r["rule_label"] for r in rows],
            "features": [str(r["features"]) for r in rows],
            "conditions": [str(r["conditions"]) for r in rows],
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
    return table


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
