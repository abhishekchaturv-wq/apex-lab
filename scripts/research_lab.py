"""Research lab script for EMA crossover forward-return analysis and backtesting."""

from __future__ import annotations

import argparse
import bisect
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.backtest.backtester import (
    DEFAULT_EQUITY_CURVE_OUTPUT,
    DEFAULT_TRADES_OUTPUT,
    ExitMode,
    compute_metrics,
    run_backtest,
    write_backtest_reports,
)
from apex_lab.research.backtest.backtester import (
    DEFAULT_SUMMARY_OUTPUT as DEFAULT_BACKTEST_SUMMARY_OUTPUT,
)
from apex_lab.research.context.engine import (
    DEFAULT_FIXED_BARS as CONTEXT_DEFAULT_FIXED_BARS,
)
from apex_lab.research.context.engine import (
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTEXT_OUTPUT_DIR,
)
from apex_lab.research.context.engine import (
    run_context_research,
)
from apex_lab.research.alpha.engine import (
    DEFAULT_OUTPUT_DIR as DEFAULT_ALPHA_OUTPUT_DIR,
)
from apex_lab.research.alpha.engine import (
    run_alpha_scoring,
)
from apex_lab.research.factors.factor_engine import (
    DEFAULT_FIXED_BARS as FACTORS_DEFAULT_FIXED_BARS,
)
from apex_lab.research.factors.factor_engine import (
    DEFAULT_OUTPUT_DIR as FACTORS_DEFAULT_OUTPUT_DIR,
)
from apex_lab.research.factors.factor_engine import (
    run_factor_research,
)
from apex_lab.research.optimization.walkforward_optimizer import optimize as run_optimize
from apex_lab.research.portfolio.metrics import (
    compute_monthly_returns,
    compute_portfolio_metrics,
    compute_rolling_drawdown,
    compute_rolling_sharpe,
    compute_yearly_returns,
)
from apex_lab.research.portfolio.portfolio import PositionSizing, simulate_portfolio
from apex_lab.research.portfolio.report import (
    DEFAULT_PORTFOLIO_OUTPUT_DIR,
    write_portfolio_reports,
)

DEFAULT_DATA_PATH = Path("data/raw/30minute/NIFTY BANK.parquet")
DEFAULT_CSV_OUTPUT = Path("reports/lab/csv/ema_cross_returns.csv")
DEFAULT_JSON_OUTPUT = Path("reports/lab/json/ema_cross_summary.json")
FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
ATR_PERIOD = 14
ATR_PERCENTILE_WINDOW = 100

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run EMA crossover forward-return research.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input OHLCV parquet file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Path to the CSV report for bullish crossover forward returns.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Path to the JSON summary report.",
    )
    # Event-driven backtest arguments
    parser.add_argument(
        "--mode",
        choices=["forward_return", "event", "optimize", "factors", "portfolio", "context", "alpha"],
        default="forward_return",
        help=(
            "Analysis mode: forward_return (default), event (backtest), "
            "optimize (walk-forward), factors (factor combination research), "
            "portfolio (portfolio simulation), context (alpha discovery engine), "
            "or alpha (alpha scoring engine)."
        ),
    )
    parser.add_argument(
        "--exit",
        dest="exit_mode",
        choices=["opposite_crossover", "fixed_bars"],
        default="opposite_crossover",
        help="Exit mode for event backtest (default: opposite_crossover).",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=10,
        help="Number of bars to hold when --exit fixed_bars is used (default: 10).",
    )
    parser.add_argument(
        "--trades-output",
        type=Path,
        default=DEFAULT_TRADES_OUTPUT,
        help="Path for the backtest trades CSV.",
    )
    parser.add_argument(
        "--equity-curve-output",
        type=Path,
        default=DEFAULT_EQUITY_CURVE_OUTPUT,
        help="Path for the backtest equity curve CSV.",
    )
    parser.add_argument(
        "--backtest-summary-output",
        type=Path,
        default=DEFAULT_BACKTEST_SUMMARY_OUTPUT,
        help="Path for the backtest summary JSON.",
    )
    parser.add_argument(
        "--optimize-output-dir",
        type=Path,
        default=Path("reports/lab/walkforward"),
        help="Directory for walk-forward optimization output files (default: reports/lab/walkforward).",
    )
    parser.add_argument(
        "--factors-output-dir",
        type=Path,
        default=FACTORS_DEFAULT_OUTPUT_DIR,
        help="Directory for factor combination research output files (default: reports/lab/factors).",
    )
    parser.add_argument(
        "--factors-bars",
        type=int,
        default=FACTORS_DEFAULT_FIXED_BARS,
        help="Number of bars to hold per trade in factor research (default: 10).",
    )
    # ---------------------------------------------------------------------------
    # Portfolio simulation arguments
    # ---------------------------------------------------------------------------
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=25_000.0,
        help="Starting capital for portfolio simulation (default: 25000).",
    )
    parser.add_argument(
        "--position-sizing",
        choices=["fixed", "percent_equity", "risk_percent"],
        default="percent_equity",
        help="Position-sizing strategy (default: percent_equity).",
    )
    parser.add_argument(
        "--risk-percent",
        type=float,
        default=1.0,
        help="Percentage of equity risked per trade for risk_percent sizing (default: 1.0).",
    )
    parser.add_argument(
        "--portfolio-output-dir",
        type=Path,
        default=DEFAULT_PORTFOLIO_OUTPUT_DIR,
        help="Directory for portfolio simulation output files (default: reports/lab/portfolio).",
    )
    parser.add_argument(
        "--context-output-dir",
        type=Path,
        default=DEFAULT_CONTEXT_OUTPUT_DIR,
        help="Directory for context research output files (default: reports/lab/context).",
    )
    parser.add_argument(
        "--context-bars",
        type=int,
        default=CONTEXT_DEFAULT_FIXED_BARS,
        help="Number of bars to hold per trade in context research (default: 10).",
    )
    parser.add_argument(
        "--alpha-output-dir",
        type=Path,
        default=DEFAULT_ALPHA_OUTPUT_DIR,
        help="Directory for alpha scoring output files (default: reports/lab/alpha).",
    )
    # Brokerage / cost hooks (reserved for future NSE cost models)
    parser.add_argument(
        "--brokerage",
        type=float,
        default=0.0,
        help="Flat brokerage cost per trade (default: 0; future use).",
    )
    parser.add_argument(
        "--exchange-fees",
        type=float,
        default=0.0,
        help="Exchange fee rate (default: 0; future use).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Slippage in basis points (default: 0; future use).",
    )
    return parser.parse_args()


def load_ohlcv(path: Path) -> pl.DataFrame:
    """Load and validate an OHLCV parquet file."""
    df = pl.read_parquet(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"input parquet is missing required columns: {missing}")

    return df.sort("timestamp")


def compute_ema_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Append EMA values, crossover signals, and forward-return columns."""
    epsilon = pl.lit(1e-9)
    prev_close = pl.col("close").shift(1)
    true_range = (
        pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        )
        .cast(pl.Float64)
        .alias("_tr")
    )

    enriched = df.with_columns(
        [
            pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20"),
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200"),
            true_range,
        ]
    ).with_columns(
        [
            pl.col("_tr").rolling_mean(window_size=ATR_PERIOD).alias("atr_14"),
            (
                (pl.col("ema_20") > pl.col("ema_50"))
                & (pl.col("ema_20").shift(1) <= pl.col("ema_50").shift(1))
            )
            .fill_null(False)
            .alias("bullish_crossover"),
            (
                (pl.col("ema_20") < pl.col("ema_50"))
                & (pl.col("ema_20").shift(1) >= pl.col("ema_50").shift(1))
            )
            .fill_null(False)
            .alias("bearish_crossover"),
        ]
    )

    atr_pct = _rolling_percentile_rank(enriched.get_column("atr_14"), ATR_PERCENTILE_WINDOW)
    enriched = enriched.with_columns(
        [
            atr_pct.alias("atr_pct"),
            (pl.col("atr_14") / (pl.col("close") + epsilon) * 100.0).alias("atr_norm"),
        ]
    )

    forward_return_columns = [
        ((pl.col("close").shift(-horizon) / pl.col("close")) - 1.0)
        .mul(100.0)
        .alias(f"forward_return_{horizon}")
        for horizon in FORWARD_RETURN_HORIZONS
    ]
    return enriched.with_columns(forward_return_columns).drop("_tr")


def _rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Compute the rolling percentile rank (0–100) of *series*.

    Null input values remain null in the output. The rolling window tracks the
    most recent non-null values up to *window* entries.
    """
    values = series.to_list()
    out: list[float | None] = [None] * len(values)
    active_window: deque[float] = deque()
    sorted_window: list[float] = []

    for index, current in enumerate(values):
        if current is not None:
            # The configured ATR percentile window is small (100 bars) and bounded, so
            # maintaining a sorted in-memory window keeps the implementation simple.
            bisect.insort(sorted_window, current)
            active_window.append(current)

        if len(active_window) > window:
            # Keep exactly the most recent ``window`` non-null values.
            expired = active_window.popleft()
            expired_index = bisect.bisect_left(sorted_window, expired)
            del sorted_window[expired_index]

        if current is None or not sorted_window:
            continue

        rank_position = bisect.bisect_right(sorted_window, current)
        out[index] = rank_position / len(sorted_window) * 100.0

    return pl.Series(out, dtype=pl.Float64)


def build_bullish_returns_report(df: pl.DataFrame) -> pl.DataFrame:
    """Extract bullish crossover rows for CSV export."""
    return df.filter(pl.col("bullish_crossover")).select(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_20",
            "ema_50",
            pl.lit("bullish_crossover").alias("signal"),
            *[f"forward_return_{horizon}" for horizon in FORWARD_RETURN_HORIZONS],
        ]
    )


def summarize_forward_returns(
    bullish_returns: pl.DataFrame,
    bullish_crossovers: int,
    bearish_crossovers: int,
) -> dict[str, Any]:
    """Build summary metrics for each forward-return horizon."""
    forward_returns: dict[str, Any] = {}

    for horizon in FORWARD_RETURN_HORIZONS:
        column = f"forward_return_{horizon}"
        values = bullish_returns.get_column(column).drop_nulls()
        num_signals = len(values)

        if num_signals == 0:
            forward_returns[str(horizon)] = {
                "num_signals": 0,
                "win_rate": None,
                "mean_return": None,
                "median_return": None,
                "standard_deviation": None,
                "maximum_gain": None,
                "maximum_loss": None,
            }
            continue

        wins = values.gt(0).sum()
        forward_returns[str(horizon)] = {
            "num_signals": num_signals,
            "win_rate": float(wins / num_signals * 100.0),
            "mean_return": float(values.mean()),
            "median_return": float(values.median()),
            "standard_deviation": float(values.std(ddof=0)),
            "maximum_gain": float(values.max()),
            "maximum_loss": float(values.min()),
        }

    return {
        "bullish_crossovers": bullish_crossovers,
        "bearish_crossovers": bearish_crossovers,
        "forward_returns": forward_returns,
    }


def write_reports(
    bullish_returns: pl.DataFrame,
    summary: dict[str, Any],
    csv_output: Path,
    json_output: Path,
) -> None:
    """Persist CSV and JSON research outputs."""
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    bullish_returns.write_csv(csv_output)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_research_lab(
    data_path: Path,
    csv_output: Path = DEFAULT_CSV_OUTPUT,
    json_output: Path = DEFAULT_JSON_OUTPUT,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run EMA crossover research and write reports to disk."""
    df = load_ohlcv(data_path)
    enriched = compute_ema_signals(df)
    bullish_returns = build_bullish_returns_report(enriched)
    summary = summarize_forward_returns(
        bullish_returns=bullish_returns,
        bullish_crossovers=int(enriched.get_column("bullish_crossover").sum()),
        bearish_crossovers=int(enriched.get_column("bearish_crossover").sum()),
    )
    write_reports(bullish_returns, summary, csv_output, json_output)
    return bullish_returns, summary


def run_event_backtest(
    data_path: Path,
    exit_mode: ExitMode = "opposite_crossover",
    fixed_bars: int = 10,
    trades_output: Path = DEFAULT_TRADES_OUTPUT,
    summary_output: Path = DEFAULT_BACKTEST_SUMMARY_OUTPUT,
    equity_curve_output: Path | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the event-driven EMA crossover backtest and write reports to disk.

    Args:
        data_path: Path to the OHLCV parquet file.
        exit_mode: How to exit trades (opposite_crossover or fixed_bars).
        fixed_bars: Bars to hold when exit_mode is fixed_bars.
        trades_output: Destination path for trades CSV.
        summary_output: Destination path for summary JSON.
        equity_curve_output: Destination path for the equity curve CSV.  If
            omitted, writes ``equity_curve.csv`` alongside *trades_output*.

    Returns:
        A tuple of (trades DataFrame, metrics dictionary).
    """
    df = load_ohlcv(data_path)
    enriched = compute_ema_signals(df)
    trades = run_backtest(enriched, exit_mode=exit_mode, fixed_bars=fixed_bars)
    metrics = compute_metrics(trades)
    write_backtest_reports(trades, metrics, trades_output, summary_output, equity_curve_output)
    return trades, metrics


def run_portfolio(
    data_path: Path,
    initial_capital: float = 25_000.0,
    position_sizing: PositionSizing = "percent_equity",
    risk_percent: float = 1.0,
    output_dir: Path = DEFAULT_PORTFOLIO_OUTPUT_DIR,
    brokerage: float = 0.0,
    exchange_fees: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[dict, pl.DataFrame]:
    """Run portfolio simulation and write all reports to *output_dir*.

    Args:
        data_path: Path to the OHLCV parquet file.
        initial_capital: Starting capital.
        position_sizing: Sizing strategy.
        risk_percent: Risk percentage for ``risk_percent`` sizing.
        output_dir: Destination directory for all report files.
        brokerage: Reserved brokerage parameter (default 0).
        exchange_fees: Reserved exchange-fees parameter (default 0).
        slippage_bps: Reserved slippage parameter (default 0).

    Returns:
        A tuple of (summary dict, equity DataFrame).
    """
    df = load_ohlcv(data_path)
    enriched = compute_ema_signals(df)
    trades = run_backtest(enriched, exit_mode="opposite_crossover")

    equity_df = simulate_portfolio(
        trades,
        initial_capital=initial_capital,
        position_sizing=position_sizing,
        risk_percent=risk_percent,
        brokerage=brokerage,
        exchange_fees=exchange_fees,
        slippage_bps=slippage_bps,
    )

    summary = compute_portfolio_metrics(equity_df, initial_capital=initial_capital)
    monthly = compute_monthly_returns(equity_df)
    yearly = compute_yearly_returns(equity_df)
    rolling_sharpe = compute_rolling_sharpe(equity_df)
    rolling_dd = compute_rolling_drawdown(equity_df)

    write_portfolio_reports(equity_df, summary, monthly, yearly, rolling_sharpe, rolling_dd, output_dir)
    return summary, equity_df


def run_context(
    data_path: Path,
    output_dir: Path = DEFAULT_CONTEXT_OUTPUT_DIR,
    fixed_bars: int = CONTEXT_DEFAULT_FIXED_BARS,
) -> tuple:
    """Run the Alpha Discovery Engine and write all context research reports.

    Args:
        data_path: Path to the OHLCV parquet file.
        output_dir: Directory for output files.
        fixed_bars: Number of bars to hold per trade.

    Returns:
        A tuple of (summary, leaderboard, best_features, correlation).
    """
    df = load_ohlcv(data_path)
    return run_context_research(df, output_dir=output_dir, fixed_bars=fixed_bars)


def run_factors(
    data_path: Path,
    output_dir: Path = FACTORS_DEFAULT_OUTPUT_DIR,
    fixed_bars: int = FACTORS_DEFAULT_FIXED_BARS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run factor combination research and write reports to disk.

    Args:
        data_path: Path to the OHLCV parquet file.
        output_dir: Directory for leaderboard.csv and summary.csv outputs.
        fixed_bars: Number of bars to hold per trade.

    Returns:
        A tuple of (leaderboard DataFrame, summary DataFrame).
    """
    df = load_ohlcv(data_path)
    return run_factor_research(df, output_dir=output_dir, fixed_bars=fixed_bars)


def run_alpha(
    data_path: Path,
    output_dir: Path = DEFAULT_ALPHA_OUTPUT_DIR,
    fixed_bars: int = CONTEXT_DEFAULT_FIXED_BARS,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], pl.DataFrame, pl.DataFrame]:
    """Run alpha scoring engine and write reports to disk."""
    df = load_ohlcv(data_path)
    result = run_alpha_scoring(df, output_dir=output_dir, fixed_bars=fixed_bars)
    return (
        result.scores,
        result.score_analysis,
        result.validation,
        result.top_trades,
        result.bottom_trades,
    )


def main() -> None:
    """Execute the research lab CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.mode == "context":
        summary, leaderboard, best_features, correlation = run_context(
            data_path=args.data,
            output_dir=args.context_output_dir,
            fixed_bars=args.context_bars,
        )
        logger.info(
            "Context research complete: %d feature-buckets evaluated, "
            "%d leaderboard entries",
            summary.height,
            leaderboard.height,
        )
    elif args.mode == "alpha":
        scores, score_analysis, score_validation, top_trades, bottom_trades = run_alpha(
            data_path=args.data,
            output_dir=args.alpha_output_dir,
            fixed_bars=args.context_bars,
        )
        logger.info("Alpha Summary: %s", json.dumps(
            {
                "number_of_trades": scores.height,
                "mean_alpha_score": float(scores["alpha_score"].mean()) if scores.height else None,
                "median_alpha_score": float(scores["alpha_score"].median()) if scores.height else None,
                "highest_score": float(scores["alpha_score"].max()) if scores.height else None,
                "lowest_score": float(scores["alpha_score"].min()) if scores.height else None,
            },
            indent=2,
        ))
        logger.info("Score Validation: %s", json.dumps(score_validation, indent=2))
        logger.info("Top 20 Score Buckets:\\n%s", score_analysis.head(20))
        logger.info("Top 20 Highest Alpha Trades:\\n%s", top_trades.head(20))
        logger.info("Top 20 Lowest Alpha Trades:\\n%s", bottom_trades.head(20))
    elif args.mode == "event":
        trades, metrics = run_event_backtest(
            data_path=args.data,
            exit_mode=args.exit_mode,
            fixed_bars=args.bars,
            trades_output=args.trades_output,
            summary_output=args.backtest_summary_output,
            equity_curve_output=args.equity_curve_output,
        )
        logger.info(
            "Backtest complete: %d trades, win_rate=%.1f%%, expectancy=%.4f",
            metrics["number_of_trades"],
            (metrics["win_rate"] or 0.0) * 100.0,
            metrics["expectancy"] or 0.0,
        )
    elif args.mode == "optimize":
        df = load_ohlcv(args.data)
        summary, leaderboard, best_params = run_optimize(df, output_dir=args.optimize_output_dir)
        logger.info(
            "Optimization complete: %d window-pair rows evaluated",
            summary.height,
        )
        if best_params:
            logger.info(
                "Best parameters: fast_ema=%d, slow_ema=%d (mean_profit_factor=%.4f)",
                best_params["fast_ema"],
                best_params["slow_ema"],
                best_params.get("mean_profit_factor") or 0.0,
            )
    elif args.mode == "factors":
        leaderboard, summary = run_factors(
            data_path=args.data,
            output_dir=args.factors_output_dir,
            fixed_bars=args.factors_bars,
        )
        logger.info(
            "Factor research complete: %d combinations evaluated, %d total trades",
            leaderboard.height,
            summary.height,
        )
    elif args.mode == "portfolio":
        summary, equity_df = run_portfolio(
            data_path=args.data,
            initial_capital=args.initial_capital,
            position_sizing=args.position_sizing,
            risk_percent=args.risk_percent,
            output_dir=args.portfolio_output_dir,
            brokerage=args.brokerage,
            exchange_fees=args.exchange_fees,
            slippage_bps=args.slippage_bps,
        )
        logger.info(
            "Portfolio simulation complete: %d trades, ending_capital=%.2f, "
            "total_return=%.2f%%, max_drawdown=%.2f%%",
            summary["number_of_trades"],
            summary["ending_capital"],
            summary["total_return_pct"],
            summary["maximum_drawdown"],
        )
    else:
        bullish_returns, summary = run_research_lab(args.data, args.csv_output, args.json_output)
        logger.info(
            "Wrote %d bullish crossover rows to %s",
            bullish_returns.height,
            args.csv_output,
        )
        logger.info(
            "Summary written to %s (%d bullish, %d bearish crossovers)",
            args.json_output,
            summary["bullish_crossovers"],
            summary["bearish_crossovers"],
        )


if __name__ == "__main__":
    main()
