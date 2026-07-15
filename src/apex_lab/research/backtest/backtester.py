"""Event-driven EMA crossover backtester.

Supports two exit modes:

- ``opposite_crossover``: exit when EMA20 crosses below EMA50.
- ``fixed_bars``: exit after a fixed number of bars.

Trades never overlap; new entry signals are ignored while a trade is open.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import polars as pl

logger = logging.getLogger(__name__)

ExitMode = Literal["opposite_crossover", "fixed_bars"]

DEFAULT_TRADES_OUTPUT = Path("reports/lab/backtest/trades.csv")
DEFAULT_SUMMARY_OUTPUT = Path("reports/lab/backtest/summary.json")


# ---------------------------------------------------------------------------
# Core backtesting logic
# ---------------------------------------------------------------------------


def run_backtest(
    df: pl.DataFrame,
    exit_mode: ExitMode = "opposite_crossover",
    fixed_bars: int = 10,
) -> pl.DataFrame:
    """Run an event-driven EMA20/EMA50 crossover backtest on *df*.

    Args:
        df: DataFrame that must contain at least the columns produced by
            ``compute_ema_signals`` from ``scripts/research_lab.py``:
            ``timestamp``, ``close``, ``bullish_crossover``,
            ``bearish_crossover``, ``ema_20``, ``ema_50``.
        exit_mode: How to exit a trade.
            ``"opposite_crossover"`` exits when EMA20 crosses below EMA50.
            ``"fixed_bars"`` exits after *fixed_bars* bars.
        fixed_bars: Number of bars to hold when *exit_mode* is
            ``"fixed_bars"``.  Ignored for ``"opposite_crossover"``.

    Returns:
        A Polars DataFrame with one row per completed trade and columns:
        ``entry_time``, ``exit_time``, ``entry_price``, ``exit_price``,
        ``bars_held``, ``return_pct``, ``exit_reason``.

    Raises:
        ValueError: If *exit_mode* is not a recognised value.
        ValueError: If *df* is missing required columns.
    """
    required = {"timestamp", "close", "bullish_crossover", "bearish_crossover"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    if exit_mode not in ("opposite_crossover", "fixed_bars"):
        raise ValueError(
            f"Unknown exit_mode '{exit_mode}'. "
            "Expected 'opposite_crossover' or 'fixed_bars'."
        )

    timestamps = df["timestamp"].to_list()
    closes = df["close"].to_list()
    bullish = df["bullish_crossover"].to_list()
    bearish = df["bearish_crossover"].to_list()

    trades: list[dict[str, Any]] = []
    in_trade = False
    entry_idx: int = 0
    entry_price: float = 0.0
    entry_time: Any = None

    for i in range(len(df)):
        if not in_trade:
            if bullish[i]:
                in_trade = True
                entry_idx = i
                entry_price = closes[i]
                entry_time = timestamps[i]
        else:
            # Determine whether to exit at bar *i*
            should_exit = False
            exit_reason = ""

            if exit_mode == "opposite_crossover":
                if bearish[i]:
                    should_exit = True
                    exit_reason = "opposite_crossover"
            else:  # fixed_bars
                bars_elapsed = i - entry_idx
                if bars_elapsed >= fixed_bars:
                    should_exit = True
                    exit_reason = "fixed_bars"

            if should_exit:
                exit_price = closes[i]
                bars_held = i - entry_idx
                return_pct = (exit_price / entry_price - 1.0) * 100.0
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamps[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "return_pct": return_pct,
                        "exit_reason": exit_reason,
                    }
                )
                in_trade = False

    if not trades:
        return pl.DataFrame(
            {
                "entry_time": pl.Series([], dtype=df["timestamp"].dtype),
                "exit_time": pl.Series([], dtype=df["timestamp"].dtype),
                "entry_price": pl.Series([], dtype=pl.Float64),
                "exit_price": pl.Series([], dtype=pl.Float64),
                "bars_held": pl.Series([], dtype=pl.Int64),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "exit_reason": pl.Series([], dtype=pl.Utf8),
            }
        )

    return pl.DataFrame(trades)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def compute_metrics(trades: pl.DataFrame) -> dict[str, Any]:
    """Compute performance metrics from a completed-trade log.

    Args:
        trades: DataFrame produced by :func:`run_backtest`.  May be empty.

    Returns:
        A dictionary containing the following keys:

        - ``number_of_trades`` (int)
        - ``win_rate`` (float | None) — fraction of trades with positive return
        - ``average_return`` (float | None)
        - ``median_return`` (float | None)
        - ``profit_factor`` (float | None) — gross wins / |gross losses|
        - ``expectancy`` (float | None) — win_rate * avg_win + loss_rate * avg_loss
        - ``largest_win`` (float | None)
        - ``largest_loss`` (float | None)
        - ``average_bars_held`` (float | None)
        - ``maximum_drawdown`` (float | None) — max peak-to-trough drawdown on
          the equity curve (cumulative ``return_pct`` series).
    """
    n = trades.height

    if n == 0:
        return {
            "number_of_trades": 0,
            "win_rate": None,
            "average_return": None,
            "median_return": None,
            "profit_factor": None,
            "expectancy": None,
            "largest_win": None,
            "largest_loss": None,
            "average_bars_held": None,
            "maximum_drawdown": None,
        }

    returns = trades["return_pct"]

    wins = returns.filter(returns > 0)
    losses = returns.filter(returns <= 0)

    win_rate = len(wins) / n
    avg_return = float(returns.mean())
    median_return = float(returns.median())
    largest_win = float(returns.max())
    largest_loss = float(returns.min())
    avg_bars = float(trades["bars_held"].cast(pl.Float64).mean())

    gross_wins = float(wins.sum()) if wins.len() > 0 else 0.0
    gross_losses = abs(float(losses.sum())) if losses.len() > 0 else 0.0
    # profit_factor is undefined (None) only when there are no losing trades at all.
    # When there are no winning trades, profit_factor is 0.0 (zero gross profit / positive loss).
    profit_factor: float | None = (
        gross_wins / gross_losses if gross_losses > 0 else None
    )

    avg_win = float(wins.mean()) if wins.len() > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.len() > 0 else 0.0
    loss_rate = 1.0 - win_rate
    expectancy = win_rate * avg_win + loss_rate * avg_loss

    # Maximum drawdown on the equity curve
    cum_returns = returns.cum_sum()
    running_max = cum_returns.cum_max()
    drawdowns = running_max - cum_returns
    max_drawdown = float(drawdowns.max())

    return {
        "number_of_trades": n,
        "win_rate": win_rate,
        "average_return": avg_return,
        "median_return": median_return,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "average_bars_held": avg_bars,
        "maximum_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_backtest_reports(
    trades: pl.DataFrame,
    metrics: dict[str, Any],
    trades_output: Path = DEFAULT_TRADES_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
) -> None:
    """Write the trade log and summary metrics to disk.

    Args:
        trades: Completed-trade DataFrame from :func:`run_backtest`.
        metrics: Metrics dictionary from :func:`compute_metrics`.
        trades_output: Destination path for the trades CSV.
        summary_output: Destination path for the summary JSON.
    """
    trades_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    trades.write_csv(trades_output)
    summary_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Wrote %d trades to %s", trades.height, trades_output)
    logger.info("Summary written to %s", summary_output)
