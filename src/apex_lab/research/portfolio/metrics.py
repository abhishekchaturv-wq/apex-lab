"""Portfolio performance metrics.

Computes aggregate and time-series metrics from a portfolio equity DataFrame
produced by :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

#: Rolling window size (in trades) used for rolling Sharpe computation.
ROLLING_SHARPE_WINDOW = 20


def compute_portfolio_metrics(
    equity_df: pl.DataFrame,
    initial_capital: float,
) -> dict[str, Any]:
    """Compute summary performance metrics for a simulated portfolio.

    Args:
        equity_df: DataFrame produced by
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.
        initial_capital: The starting capital used in the simulation.

    Returns:
        A dictionary with the following keys:

        - ``initial_capital`` (float)
        - ``ending_capital`` (float)
        - ``total_return_pct`` (float)
        - ``cagr`` (float | None) — annualised compound growth rate in %
        - ``maximum_drawdown`` (float) — maximum peak-to-trough decline in %
          (reported as a positive value; 0.0 when there is no drawdown)
        - ``sharpe_ratio`` (float | None)
        - ``sortino_ratio`` (float | None)
        - ``calmar_ratio`` (float | None)
        - ``profit_factor`` (float | None)
        - ``expectancy`` (float | None) — mean trade PnL in currency
        - ``average_trade`` (float | None) — mean trade return in %
        - ``largest_win`` (float | None) — largest trade return in %
        - ``largest_loss`` (float | None) — largest negative trade return in %
        - ``number_of_trades`` (int)
    """
    n = equity_df.height

    if n == 0:
        return {
            "initial_capital": initial_capital,
            "ending_capital": initial_capital,
            "total_return_pct": 0.0,
            "cagr": None,
            "maximum_drawdown": 0.0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "calmar_ratio": None,
            "profit_factor": None,
            "expectancy": None,
            "average_trade": None,
            "largest_win": None,
            "largest_loss": None,
            "number_of_trades": 0,
        }

    ending_capital = float(equity_df["ending_equity"][-1])
    total_return_pct = (ending_capital / initial_capital - 1.0) * 100.0

    # CAGR – requires at least two distinct timestamps
    cagr: float | None = None
    entry_times = equity_df["entry_time"].to_list()
    exit_times = equity_df["exit_time"].to_list()
    first_ts = entry_times[0]
    last_ts = exit_times[-1]
    total_seconds = _diff_seconds(first_ts, last_ts)
    total_years = total_seconds / (365.25 * 24 * 3600)
    if total_years > 0.0 and initial_capital > 0.0:
        cagr = ((ending_capital / initial_capital) ** (1.0 / total_years) - 1.0) * 100.0

    # Maximum drawdown (reported as positive value)
    min_drawdown = float(equity_df["drawdown"].min() or 0.0)
    maximum_drawdown = abs(min_drawdown)

    # Trade-level return statistics
    returns = equity_df["return_pct"]
    pnls = equity_df["pnl"]
    mean_return = float(returns.mean() or 0.0)
    std_return = float(returns.std(ddof=1) or 0.0)
    largest_win = float(returns.max() or 0.0)
    largest_loss = float(returns.min() or 0.0)
    average_trade = mean_return
    expectancy = float(pnls.mean() or 0.0)

    # Profit factor
    positive_pnl = pnls.filter(pnls > 0)
    negative_pnl = pnls.filter(pnls <= 0)
    gross_wins = float(positive_pnl.sum()) if positive_pnl.len() > 0 else 0.0
    gross_losses = abs(float(negative_pnl.sum())) if negative_pnl.len() > 0 else 0.0
    profit_factor: float | None = gross_wins / gross_losses if gross_losses > 0.0 else None

    # Annualisation factor (trades per year) – used for Sharpe / Sortino
    trades_per_year = n / total_years if total_years > 0.0 else float(n)
    ann_factor = math.sqrt(trades_per_year)

    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None

    if std_return > 0.0:
        sharpe_ratio = (mean_return / std_return) * ann_factor

    # Downside deviation (uses returns below zero as the target)
    negative_returns = returns.filter(returns < 0)
    if negative_returns.len() > 0:
        downside_sq_mean = float((negative_returns**2).mean() or 0.0)
        downside_std = math.sqrt(downside_sq_mean)
        if downside_std > 0.0:
            sortino_ratio = (mean_return / downside_std) * ann_factor

    if maximum_drawdown > 0.0:
        calmar_ratio = total_return_pct / maximum_drawdown

    return {
        "initial_capital": initial_capital,
        "ending_capital": round(ending_capital, 4),
        "total_return_pct": round(total_return_pct, 4),
        "cagr": round(cagr, 4) if cagr is not None else None,
        "maximum_drawdown": round(maximum_drawdown, 4),
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
        "sortino_ratio": round(sortino_ratio, 4) if sortino_ratio is not None else None,
        "calmar_ratio": round(calmar_ratio, 4) if calmar_ratio is not None else None,
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "expectancy": round(expectancy, 4),
        "average_trade": round(average_trade, 4),
        "largest_win": round(largest_win, 4),
        "largest_loss": round(largest_loss, 4),
        "number_of_trades": n,
    }


def compute_monthly_returns(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate equity-curve data into monthly returns.

    For each calendar month that contains at least one trade the return is
    computed as::

        return_pct = (ending_equity_last_trade / starting_equity_first_trade - 1) * 100

    Args:
        equity_df: DataFrame produced by
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.

    Returns:
        A Polars DataFrame with columns ``year`` (Int32), ``month`` (Int32),
        and ``return_pct`` (Float64), ordered chronologically.
    """
    if equity_df.height == 0:
        return pl.DataFrame(
            {
                "year": pl.Series([], dtype=pl.Int32),
                "month": pl.Series([], dtype=pl.Int32),
                "return_pct": pl.Series([], dtype=pl.Float64),
            }
        )

    # Extract year/month from exit_time and keep only the columns we need.
    df = equity_df.select(
        [
            pl.col("exit_time").cast(pl.Datetime).dt.year().cast(pl.Int32).alias("year"),
            pl.col("exit_time").cast(pl.Datetime).dt.month().cast(pl.Int32).alias("month"),
            pl.col("starting_equity"),
            pl.col("ending_equity"),
        ]
    )

    # For each (year, month): first starting_equity and last ending_equity.
    grouped = (
        df.group_by(["year", "month"])
        .agg(
            [
                pl.col("starting_equity").first().alias("equity_start"),
                pl.col("ending_equity").last().alias("equity_end"),
            ]
        )
        .sort(["year", "month"])
    )

    return grouped.with_columns(
        (
            (pl.col("equity_end") / pl.col("equity_start") - 1.0) * 100.0
        ).alias("return_pct")
    ).select(["year", "month", "return_pct"])


def compute_yearly_returns(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate equity-curve data into annual returns.

    For each calendar year the return is computed as::

        return_pct = (ending_equity_last_trade / starting_equity_first_trade - 1) * 100

    Args:
        equity_df: DataFrame produced by
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.

    Returns:
        A Polars DataFrame with columns ``year`` (Int32) and
        ``return_pct`` (Float64), ordered chronologically.
    """
    if equity_df.height == 0:
        return pl.DataFrame(
            {
                "year": pl.Series([], dtype=pl.Int32),
                "return_pct": pl.Series([], dtype=pl.Float64),
            }
        )

    df = equity_df.select(
        [
            pl.col("exit_time").cast(pl.Datetime).dt.year().cast(pl.Int32).alias("year"),
            pl.col("starting_equity"),
            pl.col("ending_equity"),
        ]
    )

    grouped = (
        df.group_by("year")
        .agg(
            [
                pl.col("starting_equity").first().alias("equity_start"),
                pl.col("ending_equity").last().alias("equity_end"),
            ]
        )
        .sort("year")
    )

    return grouped.with_columns(
        (
            (pl.col("equity_end") / pl.col("equity_start") - 1.0) * 100.0
        ).alias("return_pct")
    ).select(["year", "return_pct"])


def compute_rolling_sharpe(
    equity_df: pl.DataFrame,
    window: int = ROLLING_SHARPE_WINDOW,
) -> pl.DataFrame:
    """Compute rolling Sharpe ratio over a sliding trade window.

    Args:
        equity_df: DataFrame produced by
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.
        window: Number of trades in the rolling window (default 20).

    Returns:
        A Polars DataFrame with columns ``date`` (Date) and
        ``rolling_sharpe`` (Float64).  Rows where the window has fewer than
        2 observations carry ``null``.
    """
    if equity_df.height == 0:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "rolling_sharpe": pl.Series([], dtype=pl.Float64),
            }
        )

    returns = equity_df["return_pct"].to_list()
    exit_times = equity_df["exit_time"].to_list()

    rolling: list[float | None] = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_returns = returns[start : i + 1]
        if len(window_returns) < 2:
            rolling.append(None)
            continue
        n = len(window_returns)
        mean = sum(window_returns) / n
        variance = sum((r - mean) ** 2 for r in window_returns) / (n - 1)
        std = math.sqrt(variance)
        rolling.append((mean / std) if std > 0.0 else None)

    # Convert exit_time to date
    dates = [_to_date(ts) for ts in exit_times]

    return pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "rolling_sharpe": pl.Series(rolling, dtype=pl.Float64),
        }
    )


def compute_rolling_drawdown(equity_df: pl.DataFrame) -> pl.DataFrame:
    """Return the per-trade running drawdown as a time series.

    The drawdown values are taken directly from the ``drawdown`` column of
    *equity_df* (which is always ≤ 0).

    Args:
        equity_df: DataFrame produced by
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.

    Returns:
        A Polars DataFrame with columns ``date`` (Date) and
        ``rolling_drawdown`` (Float64).
    """
    if equity_df.height == 0:
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "rolling_drawdown": pl.Series([], dtype=pl.Float64),
            }
        )

    exit_times = equity_df["exit_time"].to_list()
    dates = [_to_date(ts) for ts in exit_times]

    return pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "rolling_drawdown": equity_df["drawdown"],
        }
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _diff_seconds(ts_start: object, ts_end: object) -> float:
    """Return elapsed seconds between two timestamps (Polars or Python datetime)."""
    import datetime

    def _to_dt(ts: object) -> datetime.datetime:
        if isinstance(ts, datetime.datetime):
            return ts
        if isinstance(ts, datetime.date):
            return datetime.datetime(ts.year, ts.month, ts.day)
        # Polars int-based datetime (microseconds since epoch)
        if isinstance(ts, int):
            return datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=ts)
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    return (_to_dt(ts_end) - _to_dt(ts_start)).total_seconds()


def _to_date(ts: object) -> object:
    """Convert a timestamp value to a Python :class:`datetime.date`."""
    import datetime

    if isinstance(ts, datetime.datetime):
        return ts.date()
    if isinstance(ts, datetime.date):
        return ts
    if isinstance(ts, int):
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=ts)
        return dt.date()
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")
