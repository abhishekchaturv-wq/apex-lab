"""Per-bucket performance metrics for the Alpha Discovery Engine.

Given a slice of trades belonging to a specific feature bucket, this module
computes the full set of statistics used in ``summary.csv`` and the
leaderboard scoring formula.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

_LOW_SAMPLE_THRESHOLD = 30


def compute_bucket_metrics(trades: pl.DataFrame) -> dict[str, Any]:
    """Compute all performance statistics for a single feature bucket.

    Args:
        trades: Slice of trades belonging to one bucket.  Must contain a
            ``return_pct`` (Float64) column.

    Returns:
        Dict with keys: ``sample_size``, ``low_sample_size``, ``win_rate``,
        ``average_return``, ``median_return``, ``expectancy``,
        ``profit_factor``, ``sharpe``, ``maximum_drawdown``.
    """
    n = trades.height
    returns = trades["return_pct"]

    wins = returns.filter(returns > 0)
    losses = returns.filter(returns <= 0)

    win_rate = float(wins.len() / n)
    avg_win = float(wins.mean()) if wins.len() > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.len() > 0 else 0.0
    avg_ret = float(returns.mean())
    med_ret = float(returns.median())  # type: ignore[arg-type]

    gross_wins = float(wins.sum()) if wins.len() > 0 else 0.0
    gross_losses = abs(float(losses.sum())) if losses.len() > 0 else 0.0
    profit_factor: float | None = gross_wins / gross_losses if gross_losses > 0 else None

    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

    # Per-trade Sharpe (annualisation not applicable at bucket level)
    std_ret = float(returns.std(ddof=1)) if n > 1 else 0.0
    sharpe = avg_ret / std_ret if std_ret > 0 else 0.0

    # Maximum drawdown from cumulative equity curve
    equity = returns.cum_sum().to_list()
    peak: float = equity[0] if equity else 0.0
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    return {
        "sample_size": n,
        "low_sample_size": n < _LOW_SAMPLE_THRESHOLD,
        "win_rate": _safe_round(win_rate),
        "average_return": _safe_round(avg_ret),
        "median_return": _safe_round(med_ret),
        "expectancy": _safe_round(expectancy),
        "profit_factor": _safe_round(profit_factor) if profit_factor is not None else None,
        "sharpe": _safe_round(sharpe),
        "maximum_drawdown": _safe_round(max_dd),
    }


def compute_all_bucket_metrics(
    trade_contexts: pl.DataFrame,
    feature_names: list[str],
) -> pl.DataFrame:
    """Compute per-bucket metrics for every registered feature.

    Args:
        trade_contexts: DataFrame with one row per trade, containing
            ``return_pct`` and ``ctx_{feature_name}`` columns for each feature.
        feature_names: Ordered list of feature names to process.

    Returns:
        summary DataFrame with columns: ``feature``, ``bucket``,
        ``sample_size``, ``low_sample_size``, ``win_rate``, ``average_return``,
        ``median_return``, ``expectancy``, ``profit_factor``, ``sharpe``,
        ``maximum_drawdown``.
    """
    rows: list[dict[str, Any]] = []

    for fname in feature_names:
        ctx_col = f"ctx_{fname}"
        if ctx_col not in trade_contexts.columns:
            continue

        labels = (
            trade_contexts.get_column(ctx_col)
            .drop_nulls()
            .unique(maintain_order=False)
            .sort()
            .to_list()
        )

        for bucket_label in labels:
            subset = trade_contexts.filter(pl.col(ctx_col) == bucket_label)
            if subset.height == 0:
                continue
            metrics = compute_bucket_metrics(subset)
            rows.append({"feature": fname, "bucket": bucket_label, **metrics})

    if not rows:
        return pl.DataFrame(
            {
                "feature": pl.Series([], dtype=pl.Utf8),
                "bucket": pl.Series([], dtype=pl.Utf8),
                "sample_size": pl.Series([], dtype=pl.Int64),
                "low_sample_size": pl.Series([], dtype=pl.Boolean),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "median_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "profit_factor": pl.Series([], dtype=pl.Float64),
                "sharpe": pl.Series([], dtype=pl.Float64),
                "maximum_drawdown": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_round(value: float, decimals: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, decimals)
