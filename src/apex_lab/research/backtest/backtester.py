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
DEFAULT_EQUITY_CURVE_FILENAME = "equity_curve.csv"
DEFAULT_EQUITY_CURVE_OUTPUT = Path("reports/lab/backtest/equity_curve.csv")
# Trades entered when ATR percentile rank >= 50 (above median) are labelled "high" volatility.
VOLATILITY_HIGH_THRESHOLD = 50.0
REGIME_SUMMARY_COLUMNS: tuple[str, ...] = ("trend_regime", "volatility_regime")


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
        ``bars_held``, ``return_pct``, ``exit_reason``, ``trend_regime``,
        ``volatility_regime``.

    Raises:
        ValueError: If *exit_mode* is not a recognised value.
        ValueError: If *df* is missing required columns.
    """
    required = {
        "timestamp",
        "close",
        "bullish_crossover",
        "bearish_crossover",
        "ema_200",
        "atr_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    if exit_mode not in ("opposite_crossover", "fixed_bars"):
        raise ValueError(
            f"Unknown exit_mode '{exit_mode}'. Expected 'opposite_crossover' or 'fixed_bars'."
        )

    timestamps = df["timestamp"].to_list()
    closes = df["close"].to_list()
    bullish = df["bullish_crossover"].to_list()
    bearish = df["bearish_crossover"].to_list()
    ema_200 = df["ema_200"].to_list()
    atr_pct = df["atr_pct"].to_list()

    trades: list[dict[str, Any]] = []
    in_trade = False
    entry_idx: int = 0
    entry_price: float = 0.0
    entry_time: Any = None
    trend_regime: str = ""
    volatility_regime: str = ""

    for i in range(len(df)):
        if not in_trade:
            if bullish[i]:
                in_trade = True
                entry_idx = i
                entry_price = closes[i]
                entry_time = timestamps[i]
                trend_regime = "above_ema200" if closes[i] > ema_200[i] else "below_ema200"
                if atr_pct[i] is None:
                    volatility_regime = "unknown"
                else:
                    volatility_regime = "high" if atr_pct[i] >= VOLATILITY_HIGH_THRESHOLD else "low"
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
                        "trend_regime": trend_regime,
                        "volatility_regime": volatility_regime,
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
                "trend_regime": pl.Series([], dtype=pl.Utf8),
                "volatility_regime": pl.Series([], dtype=pl.Utf8),
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
        - ``regime_summaries`` (dict[str, dict[str, Any]])
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
            "regime_summaries": {
                "trend_regime": {},
                "volatility_regime": {},
            },
        }

    equity_curve = compute_equity_curve(trades)
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
    profit_factor: float | None = gross_wins / gross_losses if gross_losses > 0 else None

    avg_win = float(wins.mean()) if wins.len() > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.len() > 0 else 0.0
    loss_rate = 1.0 - win_rate
    expectancy = win_rate * avg_win + loss_rate * avg_loss

    max_drawdown_value = equity_curve["drawdown"].max()
    max_drawdown = float(max_drawdown_value) if max_drawdown_value is not None else None

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
        "regime_summaries": _build_regime_summaries(trades),
    }


def compute_equity_curve(trades: pl.DataFrame) -> pl.DataFrame:
    """Compute equity curve, running peak, and drawdown from completed trades."""
    if trades.height == 0:
        exit_dtype = trades.schema.get("exit_time", pl.Datetime)
        return pl.DataFrame(
            {
                "trade_number": pl.Series([], dtype=pl.UInt32),
                "exit_time": pl.Series([], dtype=exit_dtype),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "equity_curve": pl.Series([], dtype=pl.Float64),
                "running_peak": pl.Series([], dtype=pl.Float64),
                "drawdown": pl.Series([], dtype=pl.Float64),
            }
        )

    return (
        trades.select(["exit_time", "return_pct"])
        .with_row_index("trade_number", offset=1)
        .with_columns(pl.col("return_pct").cum_sum().alias("equity_curve"))
        .with_columns(
            [
                pl.col("equity_curve").cum_max().alias("running_peak"),
                (pl.col("equity_curve").cum_max() - pl.col("equity_curve")).alias("drawdown"),
            ]
        )
    )


def _build_regime_summaries(trades: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Build per-regime trade summaries for supported regime columns."""
    regime_summaries: dict[str, dict[str, Any]] = {}

    for column in REGIME_SUMMARY_COLUMNS:
        if column not in trades.columns:
            regime_summaries[column] = {}
            continue

        regimes = trades.get_column(column).drop_nulls().unique(maintain_order=True).to_list()
        summaries: dict[str, Any] = {}

        for regime in regimes:
            subset = trades.filter(pl.col(column) == regime)
            returns = subset["return_pct"]
            wins = returns.filter(returns > 0)
            losses = returns.filter(returns <= 0)
            trade_count = subset.height
            win_rate = wins.len() / trade_count if trade_count > 0 else None
            gross_wins = float(wins.sum()) if wins.len() > 0 else 0.0
            gross_losses = abs(float(losses.sum())) if losses.len() > 0 else 0.0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else None
            avg_win = float(wins.mean()) if wins.len() > 0 else 0.0
            avg_loss = float(losses.mean()) if losses.len() > 0 else 0.0
            # Regimes with no wins or no losses keep the missing side at 0.0 so
            # expectancy remains well-defined and consistent with the top-level metric.
            expectancy = (
                (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
                if win_rate is not None
                else None
            )

            summaries[str(regime)] = {
                "trades": trade_count,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
            }

        regime_summaries[column] = summaries

    return regime_summaries


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_backtest_reports(
    trades: pl.DataFrame,
    metrics: dict[str, Any],
    trades_output: Path = DEFAULT_TRADES_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    equity_curve_output: Path | None = None,
) -> None:
    """Write the trade log and summary metrics to disk.

    Args:
        trades: Completed-trade DataFrame from :func:`run_backtest`.
        metrics: Metrics dictionary from :func:`compute_metrics`.
        trades_output: Destination path for the trades CSV.
        summary_output: Destination path for the summary JSON.
        equity_curve_output: Destination path for the equity-curve CSV.  If
            omitted, writes ``equity_curve.csv`` alongside *trades_output*.
    """
    equity_curve_output = (
        equity_curve_output or trades_output.parent / DEFAULT_EQUITY_CURVE_FILENAME
    )
    equity_curve = compute_equity_curve(trades)

    trades_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    equity_curve_output.parent.mkdir(parents=True, exist_ok=True)

    trades.write_csv(trades_output)
    equity_curve.write_csv(equity_curve_output)
    summary_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Wrote %d trades to %s", trades.height, trades_output)
    logger.info("Equity curve written to %s", equity_curve_output)
    logger.info("Summary written to %s", summary_output)
