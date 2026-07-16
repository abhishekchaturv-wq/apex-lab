"""Strategy Research Engine.

Evaluates every registered strategy against a single dataset, collects a
common scorecard for each, and produces a ranked leaderboard.

The engine reuses the existing event-driven backtester and portfolio simulator.
No trade simulation logic is duplicated here.

Example::

    import polars as pl
    from apex_lab.research.strategies.engine import run_strategy_research

    df = pl.read_parquet("data/raw/30minute/NIFTY BANK.parquet")
    leaderboard, metrics_df = run_strategy_research(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.backtest.backtester import compute_metrics, run_backtest
from apex_lab.research.portfolio.metrics import compute_portfolio_metrics
from apex_lab.research.portfolio.portfolio import simulate_portfolio
from apex_lab.research.strategies.registry import StrategyRegistry, get_default_registry
from apex_lab.research.strategies.report import write_strategy_reports

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR: Path = Path("reports/lab/strategy")
DEFAULT_INITIAL_CAPITAL: float = 25_000.0

# ---------------------------------------------------------------------------
# Composite scoring weights
# ---------------------------------------------------------------------------

#: Weights used for the normalised composite score.  Positive = higher-is-better;
#: drawdown is inverted (lower drawdown → higher normalised score).
_SCORE_WEIGHTS: dict[str, float] = {
    "cagr": 0.25,
    "sharpe_ratio": 0.25,
    "profit_factor": 0.20,
    "maximum_drawdown": 0.15,  # inverted during normalisation
    "expectancy": 0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_strategy_research(
    df: pl.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strategy_name: str | None = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    registry: StrategyRegistry | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate strategies and produce a ranked leaderboard.

    Args:
        df: Raw OHLCV DataFrame sorted by ``timestamp`` with at least the
            columns ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``.
        output_dir: Directory where the four report files will be written.
            Previous runs are not overwritten (a timestamp suffix is appended
            when a file already exists).
        strategy_name: If provided, evaluate only this strategy (case-insensitive
            name match).  Defaults to evaluating all registered strategies.
        initial_capital: Starting capital for portfolio simulation.
        registry: Optional registry override (useful for testing).

    Returns:
        A tuple ``(leaderboard, metrics_df)`` where *leaderboard* has one row
        per strategy ranked by composite score and *metrics_df* contains the
        full scorecard for every strategy.
    """
    active_registry = registry if registry is not None else get_default_registry()

    if strategy_name is not None:
        strategies = [active_registry.get(strategy_name)]
    else:
        strategies = active_registry.all_strategies()

    scorecard_rows: list[dict[str, Any]] = []

    for strategy in strategies:
        logger.info("Evaluating strategy: %s", strategy.name)

        enriched = strategy.prepare(df)
        entry = strategy.entry_condition(enriched).fill_null(False)
        exit_ = strategy.exit_condition(enriched).fill_null(False)

        backtest_df = enriched.with_columns(
            [
                entry.alias("bullish_crossover"),
                exit_.alias("bearish_crossover"),
            ]
        )

        trades = run_backtest(backtest_df, exit_mode="opposite_crossover")
        bt_metrics = compute_metrics(trades)

        equity_df = simulate_portfolio(trades, initial_capital=initial_capital)
        port_metrics = compute_portfolio_metrics(equity_df, initial_capital=initial_capital)

        scorecard = _build_scorecard(strategy.name, bt_metrics, port_metrics)
        scorecard_rows.append(scorecard)

        logger.info(
            "  %s: %d trades, win_rate=%.1f%%, cagr=%s, sharpe=%s",
            strategy.name,
            scorecard["number_of_trades"],
            (scorecard["win_rate"] or 0.0) * 100.0,
            scorecard["cagr"],
            scorecard["sharpe_ratio"],
        )

    metrics_df = _build_metrics_df(scorecard_rows)
    leaderboard = _build_leaderboard(metrics_df)

    write_strategy_reports(leaderboard, metrics_df, output_dir)
    return leaderboard, metrics_df


# ---------------------------------------------------------------------------
# Scorecard construction
# ---------------------------------------------------------------------------


def _build_scorecard(
    strategy_name: str,
    bt_metrics: dict[str, Any],
    port_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Merge backtester and portfolio metrics into a unified scorecard row."""
    max_dd = port_metrics["maximum_drawdown"]
    total_return = port_metrics["total_return_pct"]
    return_over_dd = (total_return / max_dd) if (max_dd and max_dd > 0.0) else None

    return {
        "strategy": strategy_name,
        # Portfolio-level metrics
        "cagr": port_metrics["cagr"],
        "annual_return": port_metrics["total_return_pct"],
        "sharpe_ratio": port_metrics["sharpe_ratio"],
        "sortino_ratio": port_metrics["sortino_ratio"],
        "calmar_ratio": port_metrics["calmar_ratio"],
        "maximum_drawdown": max_dd,
        "profit_factor": port_metrics["profit_factor"],
        "average_trade": port_metrics["average_trade"],
        "expectancy": port_metrics["expectancy"],
        "number_of_trades": port_metrics["number_of_trades"],
        "return_over_drawdown": return_over_dd,
        # Backtester-level metrics (not duplicated in portfolio metrics)
        "win_rate": bt_metrics["win_rate"],
        "avg_holding_time": bt_metrics["average_bars_held"],
    }


# ---------------------------------------------------------------------------
# Leaderboard and ranking
# ---------------------------------------------------------------------------


def _build_metrics_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a Polars DataFrame from a list of scorecard dicts."""
    if not rows:
        return pl.DataFrame(
            {
                "strategy": pl.Series([], dtype=pl.Utf8),
                "cagr": pl.Series([], dtype=pl.Float64),
                "annual_return": pl.Series([], dtype=pl.Float64),
                "sharpe_ratio": pl.Series([], dtype=pl.Float64),
                "sortino_ratio": pl.Series([], dtype=pl.Float64),
                "calmar_ratio": pl.Series([], dtype=pl.Float64),
                "maximum_drawdown": pl.Series([], dtype=pl.Float64),
                "profit_factor": pl.Series([], dtype=pl.Float64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_trade": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "number_of_trades": pl.Series([], dtype=pl.Int64),
                "avg_holding_time": pl.Series([], dtype=pl.Float64),
                "return_over_drawdown": pl.Series([], dtype=pl.Float64),
            }
        )
    return pl.DataFrame(rows).cast({"number_of_trades": pl.Int64})


def _build_leaderboard(metrics_df: pl.DataFrame) -> pl.DataFrame:
    """Rank strategies by composite score and return the leaderboard DataFrame."""
    if metrics_df.height == 0:
        return metrics_df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("composite_score"),
            pl.lit(None).cast(pl.Int32).alias("rank"),
        )

    scores = _compute_composite_scores(metrics_df)
    return (
        metrics_df.with_columns(pl.Series("composite_score", scores, dtype=pl.Float64))
        .sort("composite_score", descending=True)
        .with_row_index("rank", offset=1)
        .cast({"rank": pl.Int32})
    )


def _compute_composite_scores(metrics_df: pl.DataFrame) -> list[float]:
    """Compute normalised composite scores for each row in *metrics_df*.

    Each contributing metric is min-max normalised to [0, 1].  Maximum
    drawdown is inverted so that lower drawdown maps to a higher score.
    Null values are treated as the worst possible score (0.0) for that metric.
    """
    n = metrics_df.height
    composite = [0.0] * n

    for metric, weight in _SCORE_WEIGHTS.items():
        col = metrics_df[metric].cast(pl.Float64)
        values = [v if v is not None else float("nan") for v in col.to_list()]

        # Replace NaN with 0.0 for normalisation
        finite = [v for v in values if v == v]  # filter NaN
        if not finite:
            continue

        col_min = min(finite)
        col_max = max(finite)
        span = col_max - col_min

        for i, v in enumerate(values):
            raw = v if v == v else 0.0  # NaN → 0
            if span > 0.0:
                normalised = (raw - col_min) / span
            else:
                normalised = 1.0  # all values are equal → perfect score

            if metric == "maximum_drawdown":
                normalised = 1.0 - normalised  # lower drawdown is better

            composite[i] += weight * normalised

    return [round(v, 6) for v in composite]
