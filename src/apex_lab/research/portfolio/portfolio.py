"""Portfolio simulation engine with pluggable position-sizing models.

Supports three sizing strategies:

- ``fixed``: every trade uses exactly *initial_capital* as position size.
- ``percent_equity``: every trade uses the current account value as position size.
- ``risk_percent``: position size = current_equity * risk_percent / 100.

Trading costs (brokerage, exchange fees, slippage) are accepted as parameters
but default to zero.  Future PRs will wire in realistic NSE cost models.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import polars as pl

logger = logging.getLogger(__name__)

PositionSizing = Literal["fixed", "percent_equity", "risk_percent"]

_EQUITY_SCHEMA: dict[str, type[pl.DataType]] = {
    "trade_id": pl.Int64,
    "starting_equity": pl.Float64,
    "position_size": pl.Float64,
    "return_pct": pl.Float64,
    "pnl": pl.Float64,
    "ending_equity": pl.Float64,
    "drawdown": pl.Float64,
}


def simulate_portfolio(
    trades: pl.DataFrame,
    initial_capital: float = 25_000.0,
    position_sizing: PositionSizing = "percent_equity",
    risk_percent: float = 1.0,
    brokerage: float = 0.0,
    exchange_fees: float = 0.0,
    slippage_bps: float = 0.0,
) -> pl.DataFrame:
    """Simulate portfolio growth over a sequence of completed trades.

    Args:
        trades: DataFrame from :func:`~apex_lab.research.backtest.backtester.run_backtest`
            containing at least ``entry_time``, ``exit_time``, and ``return_pct``.
        initial_capital: Starting portfolio value.
        position_sizing: Sizing strategy.  One of ``"fixed"``,
            ``"percent_equity"``, or ``"risk_percent"``.
        risk_percent: Percentage of equity risked per trade (used only by the
            ``"risk_percent"`` model).
        brokerage: Flat brokerage per trade (reserved; default 0).
        exchange_fees: Exchange fee rate (reserved; default 0).
        slippage_bps: Slippage in basis points (reserved; default 0).

    Returns:
        A Polars DataFrame with one row per trade and columns:
        ``trade_id``, ``entry_time``, ``exit_time``, ``starting_equity``,
        ``position_size``, ``return_pct``, ``pnl``, ``ending_equity``,
        ``drawdown``.

    Notes:
        * Trading costs parameters (brokerage, exchange_fees, slippage_bps) are
          structural hooks for future NSE cost integration.  They have no effect
          on results until a cost model is wired in.
        * ``ending_equity`` is floored at zero so the equity curve never goes
          negative.
        * ``drawdown`` is the percentage decline from the running peak equity;
          it is always ≤ 0.
    """
    # Silence unused-variable warnings until cost models are implemented.
    _ = brokerage, exchange_fees, slippage_bps

    if trades.height == 0:
        entry_dtype = trades.schema.get("entry_time", pl.Datetime)
        exit_dtype = trades.schema.get("exit_time", pl.Datetime)
        return pl.DataFrame(
            {
                "trade_id": pl.Series([], dtype=pl.Int64),
                "entry_time": pl.Series([], dtype=entry_dtype),
                "exit_time": pl.Series([], dtype=exit_dtype),
                "starting_equity": pl.Series([], dtype=pl.Float64),
                "position_size": pl.Series([], dtype=pl.Float64),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "pnl": pl.Series([], dtype=pl.Float64),
                "ending_equity": pl.Series([], dtype=pl.Float64),
                "drawdown": pl.Series([], dtype=pl.Float64),
            }
        )

    if position_sizing not in ("fixed", "percent_equity", "risk_percent"):
        raise ValueError(
            f"Unknown position_sizing '{position_sizing}'. "
            "Expected 'fixed', 'percent_equity', or 'risk_percent'."
        )

    entry_times = trades["entry_time"].to_list()
    exit_times = trades["exit_time"].to_list()
    return_pcts = trades["return_pct"].to_list()

    rows: list[dict[str, Any]] = []
    current_equity = initial_capital
    peak_equity = initial_capital

    for i, (entry_time, exit_time, return_pct) in enumerate(
        zip(entry_times, exit_times, return_pcts, strict=False)
    ):
        starting_equity = current_equity

        if position_sizing == "fixed":
            position_size = initial_capital
        elif position_sizing == "percent_equity":
            position_size = current_equity
        else:  # risk_percent
            position_size = current_equity * risk_percent / 100.0

        pnl = position_size * (return_pct / 100.0)
        ending_equity = max(starting_equity + pnl, 0.0)

        peak_equity = max(peak_equity, ending_equity)
        if peak_equity > 0.0:
            drawdown = (ending_equity - peak_equity) / peak_equity * 100.0
        else:
            drawdown = 0.0

        rows.append(
            {
                "trade_id": i + 1,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "starting_equity": starting_equity,
                "position_size": position_size,
                "return_pct": return_pct,
                "pnl": pnl,
                "ending_equity": ending_equity,
                "drawdown": drawdown,
            }
        )

        current_equity = ending_equity

    return pl.DataFrame(rows)
