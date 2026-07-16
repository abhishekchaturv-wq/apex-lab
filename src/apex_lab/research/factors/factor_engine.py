"""Generic factor combination research engine.

Evaluates every supported factor combination using boolean AND logic and
backtests each combined signal with the existing event-driven backtester.
Results are exported to ``reports/lab/factors/``.

Example::

    import polars as pl
    from apex_lab.research.factors.factor_engine import run_factor_research

    df = pl.read_parquet("data/raw/30minute/NIFTY BANK.parquet")
    leaderboard, summary = run_factor_research(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.backtest.backtester import compute_metrics, run_backtest
from apex_lab.research.factors.atr_volatility import AtrVolatilityFactor
from apex_lab.research.factors.base import Factor
from apex_lab.research.factors.ema_trend import EmaTrendFactor
from apex_lab.research.factors.macd import MacdFactor
from apex_lab.research.factors.rsi import RsiFactor
from apex_lab.research.factors.vwap import VwapFactor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry and combination catalogue
# ---------------------------------------------------------------------------

#: All supported factor keys mapped to their default instances.
FACTOR_REGISTRY: dict[str, Factor] = {
    "EMA": EmaTrendFactor(),
    "RSI": RsiFactor(),
    "MACD": MacdFactor(),
    "VWAP": VwapFactor(),
    "ATR": AtrVolatilityFactor(),
}

#: Every combination that will be evaluated.  EMA must appear in all of them
#: so that the backtester-required columns (``ema_200``, ``atr_pct``,
#: ``bearish_crossover``) are always present in the enriched DataFrame.
COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("EMA",),
    ("EMA", "RSI"),
    ("EMA", "MACD"),
    ("EMA", "VWAP"),
    ("EMA", "ATR"),
    ("EMA", "RSI", "MACD"),
    ("EMA", "RSI", "VWAP"),
    ("EMA", "RSI", "ATR"),
    ("EMA", "MACD", "VWAP"),
)

#: Default output directory for factor research reports.
DEFAULT_OUTPUT_DIR: Path = Path("reports/lab/factors")

#: Default exit mode passed to the event-driven backtester.
DEFAULT_EXIT_MODE = "fixed_bars"

#: Default number of bars to hold when using the ``fixed_bars`` exit mode.
DEFAULT_FIXED_BARS = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_factor_research(
    df: pl.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixed_bars: int = DEFAULT_FIXED_BARS,
    combinations: tuple[tuple[str, ...], ...] = COMBINATIONS,
    registry: dict[str, Factor] | None = None,
    expected_zero_trade_combinations: tuple[str, ...] = (),
    fail_on_unexpected_zero_trades: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run all factor-combination backtests and write output reports.

    Args:
        df: Raw OHLCV DataFrame (must have ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume`` columns, sorted by timestamp).
        output_dir: Directory where ``leaderboard.csv`` and ``summary.csv``
            will be written.
        fixed_bars: Number of bars to hold before exiting each trade.
        combinations: Sequence of factor-key tuples to evaluate.
        registry: Optional factor registry override (for testing).
        expected_zero_trade_combinations: Combination labels that are
            explicitly allowed to produce zero trades.
        fail_on_unexpected_zero_trades: Raise ``ValueError`` if a combination
            has zero trades and is not listed in
            ``expected_zero_trade_combinations``.

    Returns:
        A tuple ``(leaderboard, summary)`` where *leaderboard* has one row per
        combination and *summary* has one row per trade across all combinations.
    """
    active_registry = registry if registry is not None else FACTOR_REGISTRY
    allowed_zero_trade_labels = set(expected_zero_trade_combinations)

    leaderboard_rows: list[dict[str, Any]] = []
    summary_rows: list[pl.DataFrame] = []
    logged_vwap_diagnostics = False

    for combo in combinations:
        combo_label = " AND ".join(combo)
        logger.info("Evaluating combination: %s", combo_label)

        enriched = _enrich_for_combination(df, combo, active_registry)
        if not logged_vwap_diagnostics and "EMA" in combo and "VWAP" in combo:
            _log_vwap_diagnostics(enriched, combo_label, active_registry)
            logged_vwap_diagnostics = True
        combined_signal = _compute_combined_signal(enriched, combo, active_registry)

        backtest_df = enriched.with_columns(
            [combined_signal.alias("bullish_crossover")]
        )

        trades = run_backtest(backtest_df, exit_mode="fixed_bars", fixed_bars=fixed_bars)
        metrics = compute_metrics(trades)

        leaderboard_rows.append(
            {
                "factor_combination": combo_label,
                "number_of_trades": metrics["number_of_trades"],
                "trade_reduction_pct": None,
                "win_rate": metrics["win_rate"],
                "expectancy": metrics["expectancy"],
                "profit_factor": metrics["profit_factor"],
                "maximum_drawdown": metrics["maximum_drawdown"],
            }
        )

        if metrics["number_of_trades"] == 0 and combo_label not in allowed_zero_trade_labels:
            message = (
                f"Unexpected zero-trade result for combination '{combo_label}'. "
                "Add it to expected_zero_trade_combinations if this is intentional."
            )
            if fail_on_unexpected_zero_trades:
                raise ValueError(message)
            logger.warning(message)

        if trades.height > 0:
            trades_with_combo = trades.with_columns(
                [pl.lit(combo_label).alias("factor_combination")]
            )
            summary_rows.append(trades_with_combo)

        logger.info(
            "  %s: %d trades, win_rate=%.1f%%, expectancy=%.4f",
            combo_label,
            metrics["number_of_trades"],
            (metrics["win_rate"] or 0.0) * 100.0,
            metrics["expectancy"] or 0.0,
        )

    leaderboard = _build_leaderboard(leaderboard_rows)
    summary = _build_summary(summary_rows)

    _write_reports(leaderboard, summary, output_dir)
    return leaderboard, summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enrich_for_combination(
    df: pl.DataFrame,
    combo: tuple[str, ...],
    registry: dict[str, Factor],
) -> pl.DataFrame:
    """Apply each factor's ``compute`` in sequence, accumulating columns."""
    enriched = df
    for key in combo:
        factor = registry[key]
        enriched = factor.compute(enriched)
    return enriched


def _compute_combined_signal(
    df: pl.DataFrame,
    combo: tuple[str, ...],
    registry: dict[str, Factor],
) -> pl.Series:
    """AND all factor signals together into a single boolean Series."""
    if not combo:
        raise ValueError("combination must contain at least one factor")

    signal: pl.Series | None = None
    for key in combo:
        factor_signal = registry[key].signal(df).fill_null(False)
        signal = factor_signal if signal is None else (signal & factor_signal)
    # signal cannot be None here because combinations always have ≥ 1 factor
    assert signal is not None
    return signal


def _build_leaderboard(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build the leaderboard DataFrame from per-combination metric dicts."""
    if not rows:
        return pl.DataFrame(
            {
                "factor_combination": pl.Series([], dtype=pl.Utf8),
                "number_of_trades": pl.Series([], dtype=pl.Int64),
                "trade_reduction_pct": pl.Series([], dtype=pl.Float64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "profit_factor": pl.Series([], dtype=pl.Float64),
                "maximum_drawdown": pl.Series([], dtype=pl.Float64),
            }
        )
    leaderboard = pl.DataFrame(rows)
    baseline = (
        leaderboard.filter(pl.col("factor_combination") == "EMA")
        .select("number_of_trades")
        .to_series()
        .to_list()
    )
    baseline_trades = baseline[0] if baseline else None
    if baseline_trades is not None and baseline_trades > 0:
        leaderboard = leaderboard.with_columns(
            [
                pl.when(pl.col("factor_combination") == "EMA")
                .then(0.0)
                .otherwise((pl.col("number_of_trades") - baseline_trades) / baseline_trades * 100.0)
                .alias("trade_reduction_pct")
            ]
        )
    else:
        leaderboard = leaderboard.with_columns(
            [
                pl.when(pl.col("factor_combination") == "EMA")
                .then(0.0)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("trade_reduction_pct")
            ]
        )

    return leaderboard.select(
        [
            "factor_combination",
            "number_of_trades",
            "trade_reduction_pct",
            "win_rate",
            "expectancy",
            "profit_factor",
            "maximum_drawdown",
        ]
    )


def _collect_vwap_diagnostics(
    enriched: pl.DataFrame,
    registry: dict[str, Factor],
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Collect detailed VWAP diagnostics for EMA/VWAP investigations."""
    ema_signal = registry["EMA"].signal(enriched).fill_null(False)
    vwap_signal = registry["VWAP"].signal(enriched).fill_null(False)
    filtered_signal = ema_signal & vwap_signal

    close_above = int(
        enriched.select((pl.col("close") > pl.col("vwap")).cast(pl.Int64).sum()).item()
    )
    close_below = int(
        enriched.select((pl.col("close") < pl.col("vwap")).cast(pl.Int64).sum()).item()
    )
    stats = {
        "first_20_vwap_values": enriched.select("vwap").head(20)["vwap"].to_list(),
        "min_vwap": enriched.select(pl.col("vwap").min()).item(),
        "max_vwap": enriched.select(pl.col("vwap").max()).item(),
        "null_vwap_values": int(enriched["vwap"].null_count()),
        "close_above_vwap_bars": close_above,
        "close_below_vwap_bars": close_below,
        "bullish_vwap_signals": int(vwap_signal.sum()),
        "bearish_vwap_signals": int((~vwap_signal & enriched["vwap"].is_not_null()).sum()),
        "ema_crossover_count_before_filtering": int(ema_signal.sum()),
        "signals_before_vwap_filter": int(ema_signal.sum()),
        "signals_after_vwap_filter": int(filtered_signal.sum()),
    }
    ema_samples = (
        enriched.with_columns(
            [
                ema_signal.alias("ema_condition"),
                vwap_signal.alias("vwap_condition"),
            ]
        )
        .filter(pl.col("ema_condition"))
        .select(["timestamp", "close", "vwap", "ema_condition", "vwap_condition"])
        .head(20)
    )
    return stats, ema_samples


def _log_vwap_diagnostics(
    enriched: pl.DataFrame,
    combo_label: str,
    registry: dict[str, Factor],
) -> None:
    """Log detailed VWAP diagnostics for the first EMA/VWAP combination."""
    stats, ema_samples = _collect_vwap_diagnostics(enriched, registry)

    logger.info(
        "VWAP diagnostics (%s): first_20_vwap_values=%s, min_vwap=%s, max_vwap=%s, null_vwap_values=%d",
        combo_label,
        stats["first_20_vwap_values"],
        stats["min_vwap"],
        stats["max_vwap"],
        stats["null_vwap_values"],
    )
    logger.info(
        "VWAP signal stats (%s): close_above_vwap_bars=%d, close_below_vwap_bars=%d, bullish_vwap_signals=%d, bearish_vwap_signals=%d",
        combo_label,
        stats["close_above_vwap_bars"],
        stats["close_below_vwap_bars"],
        stats["bullish_vwap_signals"],
        stats["bearish_vwap_signals"],
    )
    logger.info(
        "EMA/VWAP filter stats (%s): ema_crossover_count_before_filtering=%d, signals_before_vwap_filter=%d, signals_after_vwap_filter=%d",
        combo_label,
        stats["ema_crossover_count_before_filtering"],
        stats["signals_before_vwap_filter"],
        stats["signals_after_vwap_filter"],
    )
    logger.info(
        "First 20 EMA signals with VWAP conditions (%s): %s",
        combo_label,
        ema_samples.to_dicts(),
    )


def _build_summary(trade_frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-combination trade logs into a single summary DataFrame."""
    if not trade_frames:
        return pl.DataFrame(
            {
                "factor_combination": pl.Series([], dtype=pl.Utf8),
                "entry_time": pl.Series([], dtype=pl.Datetime),
                "exit_time": pl.Series([], dtype=pl.Datetime),
                "entry_price": pl.Series([], dtype=pl.Float64),
                "exit_price": pl.Series([], dtype=pl.Float64),
                "bars_held": pl.Series([], dtype=pl.Int64),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "exit_reason": pl.Series([], dtype=pl.Utf8),
                "trend_regime": pl.Series([], dtype=pl.Utf8),
                "volatility_regime": pl.Series([], dtype=pl.Utf8),
            }
        )
    combined = pl.concat(trade_frames, how="diagonal_relaxed")
    # Reorder so factor_combination is the first column
    cols = ["factor_combination"] + [c for c in combined.columns if c != "factor_combination"]
    return combined.select(cols)


def _write_reports(
    leaderboard: pl.DataFrame,
    summary: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Persist leaderboard and summary CSVs to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "leaderboard.csv"
    summary_path = output_dir / "summary.csv"
    leaderboard.write_csv(leaderboard_path)
    summary.write_csv(summary_path)
    logger.info("Leaderboard written to %s (%d rows)", leaderboard_path, leaderboard.height)
    logger.info("Summary written to %s (%d rows)", summary_path, summary.height)
