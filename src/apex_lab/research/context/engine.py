"""Alpha Discovery Engine — main orchestrator.

For every EMA crossover trade, computes all registered context features at the
trade's entry bar, then evaluates per-bucket performance statistics to identify
which market conditions genuinely improve trading outcomes.

Outputs (written to ``reports/lab/context/`` by default):

- ``summary.csv``      — one row per feature × bucket with full statistics.
- ``leaderboard.csv``  — valid buckets (sample ≥ 30) ranked by composite score.
- ``best_features.json`` — best bucket per feature group.
- ``correlation.csv``  — Pearson and Spearman correlation with forward return.

Example::

    import polars as pl
    from apex_lab.research.context.engine import run_context_research

    df = pl.read_parquet("data/raw/30minute/NIFTY BANK.parquet")
    summary, leaderboard, best_features, correlation = run_context_research(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

# Import features module to trigger all register_context_feature() calls
import apex_lab.research.context.features as _features_module  # noqa: F401
from apex_lab.research.backtest.backtester import run_backtest
from apex_lab.research.context.metrics import compute_all_bucket_metrics
from apex_lab.research.context.registry import get_registry
from apex_lab.research.context.report import (
    build_best_features,
    build_correlation,
    build_leaderboard,
    write_context_reports,
)
from apex_lab.research.factors.ema_trend import EmaTrendFactor
from apex_lab.research.factors.macd import MacdFactor
from apex_lab.research.factors.rsi import RsiFactor
from apex_lab.research.factors.vwap import VwapFactor

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR: Path = Path("reports/lab/context")
DEFAULT_FIXED_BARS: int = 10


# ---------------------------------------------------------------------------
# OHLCV enrichment
# ---------------------------------------------------------------------------


def _base_enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the four existing factor compute() methods to *df*.

    Adds: ema_20, ema_50, ema_200, atr_14, atr_pct, bullish_crossover,
    bearish_crossover, rsi_14, macd_line, macd_signal, macd_hist, vwap.
    """
    enriched = EmaTrendFactor().compute(df)
    enriched = RsiFactor().compute(enriched)
    enriched = MacdFactor().compute(enriched)
    enriched = VwapFactor().compute(enriched)
    return enriched


def _enrich_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Enrich *df* with all context indicators.

    Steps:
    1. Apply existing factors (EMA, RSI, MACD, VWAP).
    2. Call each registered context feature's ``compute()`` method to add its
       specific indicator columns idempotently.

    Args:
        df: Raw OHLCV DataFrame sorted by timestamp.

    Returns:
        Enriched DataFrame with all indicator columns appended.
    """
    enriched = _base_enrich(df)
    registry = get_registry()
    for feature in registry.values():
        enriched = feature.compute(enriched)
    return enriched


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def run_context_research(
    df: pl.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixed_bars: int = DEFAULT_FIXED_BARS,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Run the full Alpha Discovery Engine pipeline.

    For every EMA crossover trade this function:

    1. Computes all registered context features at the trade's entry bar.
    2. Buckets each continuous feature into labelled intervals.
    3. Evaluates per-bucket performance statistics.
    4. Builds a leaderboard ranked by composite score.
    5. Selects the best bucket per feature group.
    6. Computes Pearson / Spearman correlation of each feature with return.
    7. Writes all four output artefacts to *output_dir*.

    Args:
        df: Raw OHLCV DataFrame (must have timestamp, open, high, low, close,
            volume columns, sorted by timestamp).
        output_dir: Directory for output CSV / JSON files.
        fixed_bars: Number of bars to hold per trade (``fixed_bars`` exit mode).

    Returns:
        Tuple of (summary, leaderboard, best_features, correlation).
    """
    registry = get_registry()
    logger.info("Running context research with %d registered features", len(registry))

    # Step 1: Enrich
    logger.info("Enriching OHLCV with context indicators…")
    enriched = _enrich_ohlcv(df)

    # Step 2: Run backtest to get trades
    logger.info("Running EMA crossover backtest (fixed_bars=%d)…", fixed_bars)
    trades = run_backtest(enriched, exit_mode="fixed_bars", fixed_bars=fixed_bars)
    logger.info("Backtest produced %d trades", trades.height)

    if trades.height == 0:
        logger.warning("No trades generated — writing empty reports")
        summary = _empty_summary()
        leaderboard = _empty_leaderboard()
        best_features: dict[str, Any] = {}
        correlation = _empty_correlation()
        write_context_reports(summary, leaderboard, best_features, correlation, output_dir)
        return summary, leaderboard, best_features, correlation

    # Step 3: Compute bucket labels and numeric values for all features, one bar per row
    logger.info("Computing feature bucket labels…")
    label_series = [
        feature.label(enriched).alias(f"ctx_{fname}") for fname, feature in registry.items()
    ]
    numeric_series = [
        feature.numeric(enriched).cast(pl.Float64).alias(f"num_{fname}")
        for fname, feature in registry.items()
    ]
    enriched_ctx = enriched.with_columns(label_series).with_columns(numeric_series)

    # Step 4: Join trades with enriched context on entry_time == timestamp
    ctx_cols = [f"ctx_{n}" for n in registry] + [f"num_{n}" for n in registry]
    trade_contexts = trades.join(
        enriched_ctx.select(["timestamp"] + ctx_cols),
        left_on="entry_time",
        right_on="timestamp",
        how="left",
    )

    # Step 5: Per-bucket metrics
    logger.info("Computing per-bucket metrics…")
    summary = compute_all_bucket_metrics(trade_contexts, list(registry.keys()))

    # Step 6: Leaderboard
    leaderboard = build_leaderboard(summary)
    logger.info("Leaderboard has %d valid entries", leaderboard.height)

    # Step 7: Best features
    best_features = build_best_features(leaderboard, summary, registry)

    # Step 8: Correlation
    correlation = build_correlation(trade_contexts, registry)

    # Step 9: Write reports
    write_context_reports(summary, leaderboard, best_features, correlation, output_dir)

    return summary, leaderboard, best_features, correlation


# ---------------------------------------------------------------------------
# Empty-frame helpers (for the zero-trades edge case)
# ---------------------------------------------------------------------------


def _empty_summary() -> pl.DataFrame:
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


def _empty_leaderboard() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rank": pl.Series([], dtype=pl.Int64),
            "feature": pl.Series([], dtype=pl.Utf8),
            "bucket": pl.Series([], dtype=pl.Utf8),
            "sample_size": pl.Series([], dtype=pl.Int64),
            "win_rate": pl.Series([], dtype=pl.Float64),
            "expectancy": pl.Series([], dtype=pl.Float64),
            "profit_factor": pl.Series([], dtype=pl.Float64),
            "sharpe": pl.Series([], dtype=pl.Float64),
            "score": pl.Series([], dtype=pl.Float64),
        }
    )


def _empty_correlation() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": pl.Series([], dtype=pl.Utf8),
            "pearson": pl.Series([], dtype=pl.Float64),
            "spearman": pl.Series([], dtype=pl.Float64),
        }
    )
