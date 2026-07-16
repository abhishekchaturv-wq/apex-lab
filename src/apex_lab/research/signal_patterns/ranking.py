"""Walk-forward validation and composite ranking for signal patterns."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from apex_lab.research.signal_discovery.statistics import normalize_series
from apex_lab.research.signal_patterns.candidate_generator import CandidateRule
from apex_lab.research.signal_patterns.evaluator import DEFAULT_TARGET, evaluate_rule

# Composite ranking weights.
_EXPECTANCY_WEIGHT = 0.30
_WIN_RATE_WEIGHT = 0.25
_PROFIT_FACTOR_WEIGHT = 0.20
_FREQUENCY_WEIGHT = 0.10
_ROBUSTNESS_WEIGHT = 0.15

# Walk-forward split ratios.
_TRAIN_RATIO = 0.60
_VALIDATION_RATIO = 0.20
# Out-of-sample is the remainder (0.20).

# Minimum signal count per split for a rule to be considered stable.
_MIN_SPLIT_SAMPLES = 5

EPSILON = 1e-12


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def walk_forward_validate(
    df: pl.DataFrame,
    candidates: list[CandidateRule],
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> pl.DataFrame:
    """Validate candidates chronologically across train / val / OOS splits.

    The dataset is split in chronological order:

    - **train**  — first 60 % of rows
    - **val**    — next  20 % of rows
    - **oos**    — last  20 % of rows (out-of-sample)

    Returns a DataFrame with per-rule per-split statistics and a
    ``is_robust`` flag that is ``True`` when the rule survives all splits
    with at least ``_MIN_SPLIT_SAMPLES`` matching rows and non-negative
    expectancy on the OOS split.
    """
    n = df.height
    train_end = int(n * _TRAIN_RATIO)
    val_end = int(n * (_TRAIN_RATIO + _VALIDATION_RATIO))

    splits = {
        "train": df.slice(0, train_end),
        "val": df.slice(train_end, val_end - train_end),
        "oos": df.slice(val_end, n - val_end),
    }

    rows: list[dict[str, Any]] = []

    for rule in candidates:
        rule_rows: dict[str, Any] = {
            "rule_label": rule.label(),
            "features": str(list(rule.features)),
            "combination_size": len(rule.features),
        }
        is_robust = True

        for split_name, split_df in splits.items():
            stats = evaluate_rule(split_df, rule, target_column=target_column, bins=bins)
            if stats is None or stats["signal_frequency"] < _MIN_SPLIT_SAMPLES:
                is_robust = False
                rule_rows[f"{split_name}_frequency"] = 0
                rule_rows[f"{split_name}_win_rate"] = None
                rule_rows[f"{split_name}_expectancy"] = None
                rule_rows[f"{split_name}_profit_factor"] = None
            else:
                rule_rows[f"{split_name}_frequency"] = stats["signal_frequency"]
                rule_rows[f"{split_name}_win_rate"] = stats["win_rate"]
                rule_rows[f"{split_name}_expectancy"] = stats["expectancy"]
                rule_rows[f"{split_name}_profit_factor"] = stats["profit_factor"]
                if split_name == "oos" and stats["expectancy"] < 0:
                    is_robust = False

        rule_rows["is_robust"] = is_robust
        rows.append(rule_rows)

    if not rows:
        return _empty_wf_frame()

    return pl.DataFrame(rows)


def rank_signals(
    stats: pl.DataFrame,
    wf: pl.DataFrame,
) -> pl.DataFrame:
    """Compute composite scores and produce the final ranked signal table.

    Merges candidate statistics with walk-forward results, normalises each
    metric, and computes a weighted composite score.  Robust signals are
    promoted; non-robust signals are demoted but kept in the output.
    """
    if stats.is_empty():
        return _empty_ranked_frame()

    # Join robustness flag from walk-forward table.
    if not wf.is_empty() and "rule_label" in wf.columns and "is_robust" in wf.columns:
        wf_slim = wf.select(["rule_label", "is_robust"])
        merged = stats.join(wf_slim, on="rule_label", how="left")
        merged = merged.with_columns(
            pl.col("is_robust").fill_null(False)
        )
    else:
        merged = stats.with_columns(pl.lit(False).alias("is_robust"))

    # Normalize scoring metrics.
    expectancy_norm = normalize_series(merged.get_column("expectancy").to_list())
    win_rate_norm = normalize_series(merged.get_column("win_rate").to_list())

    pf_list = merged.get_column("profit_factor").to_list()
    profit_factor_norm = normalize_series(
        [(_safe_float(v) if v is not None else None) for v in pf_list]
    )

    frequency_norm = normalize_series(
        merged.get_column("signal_frequency").cast(pl.Float64).to_list()
    )

    # Robustness bonus: 1.0 for robust, 0.0 for non-robust.
    robustness_scores = [1.0 if r else 0.0 for r in merged.get_column("is_robust").to_list()]
    robustness_norm = normalize_series(robustness_scores)

    composite = [
        (
            (_EXPECTANCY_WEIGHT * e)
            + (_WIN_RATE_WEIGHT * w)
            + (_PROFIT_FACTOR_WEIGHT * p)
            + (_FREQUENCY_WEIGHT * f)
            + (_ROBUSTNESS_WEIGHT * r)
        )
        for e, w, p, f, r in zip(
            expectancy_norm,
            win_rate_norm,
            profit_factor_norm,
            frequency_norm,
            robustness_norm,
            strict=True,
        )
    ]

    ranked = merged.with_columns(
        [
            pl.Series("composite_score", composite, dtype=pl.Float64),
        ]
    )

    return (
        ranked.sort("composite_score", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(
            [
                pl.col("composite_score").round(6),
                pl.col("win_rate").round(6),
                pl.col("expectancy").round(6),
                pl.col("average_return").round(6),
                pl.col("median_return").round(6),
            ]
        )
    )


def _empty_wf_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rule_label": pl.Series([], dtype=pl.Utf8),
            "features": pl.Series([], dtype=pl.Utf8),
            "combination_size": pl.Series([], dtype=pl.Int64),
            "train_frequency": pl.Series([], dtype=pl.Int64),
            "train_win_rate": pl.Series([], dtype=pl.Float64),
            "train_expectancy": pl.Series([], dtype=pl.Float64),
            "train_profit_factor": pl.Series([], dtype=pl.Float64),
            "val_frequency": pl.Series([], dtype=pl.Int64),
            "val_win_rate": pl.Series([], dtype=pl.Float64),
            "val_expectancy": pl.Series([], dtype=pl.Float64),
            "val_profit_factor": pl.Series([], dtype=pl.Float64),
            "oos_frequency": pl.Series([], dtype=pl.Int64),
            "oos_win_rate": pl.Series([], dtype=pl.Float64),
            "oos_expectancy": pl.Series([], dtype=pl.Float64),
            "oos_profit_factor": pl.Series([], dtype=pl.Float64),
            "is_robust": pl.Series([], dtype=pl.Boolean),
        }
    )


def _empty_ranked_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rank": pl.Series([], dtype=pl.UInt32),
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
            "average_holding_return": pl.Series([], dtype=pl.Float64),
            "is_robust": pl.Series([], dtype=pl.Boolean),
            "composite_score": pl.Series([], dtype=pl.Float64),
        }
    )
