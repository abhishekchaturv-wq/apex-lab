"""Alpha Scoring Engine orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.alpha.registry import get_alpha_feature_registry
from apex_lab.research.alpha.report import (
    build_alpha_summary,
    build_score_analysis,
    build_scores_report,
    build_top_bottom_trades,
    write_alpha_reports,
)
from apex_lab.research.alpha.score import score_trades
from apex_lab.research.alpha.validation import build_score_validation
from apex_lab.research.alpha.weights import (
    DEFAULT_CONTEXT_LEADERBOARD_PATH,
    DEFAULT_WEIGHTS_FILENAME,
    ensure_weights_file,
    write_weights_file,
)
from apex_lab.research.backtest.backtester import run_backtest
from apex_lab.research.context.engine import DEFAULT_FIXED_BARS, _enrich_ohlcv

DEFAULT_OUTPUT_DIR = Path("reports/lab/alpha")


@dataclass(frozen=True)
class AlphaScoringResult:
    """Result bundle returned by :func:`run_alpha_scoring`."""

    scores: pl.DataFrame
    score_analysis: pl.DataFrame
    validation: dict[str, Any]
    top_trades: pl.DataFrame
    bottom_trades: pl.DataFrame
    alpha_summary: dict[str, Any]


def run_alpha_scoring(
    df: pl.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fixed_bars: int = DEFAULT_FIXED_BARS,
    weights_path: Path | None = None,
    context_leaderboard_path: Path = DEFAULT_CONTEXT_LEADERBOARD_PATH,
) -> AlphaScoringResult:
    """Run alpha scoring over EMA crossover trades and write all report artifacts."""
    alpha_registry = get_alpha_feature_registry()
    weights_file = weights_path or output_dir / DEFAULT_WEIGHTS_FILENAME
    weights = ensure_weights_file(weights_file, context_leaderboard_path)
    write_weights_file(weights_file, weights)

    enriched = _enrich_ohlcv(df)
    trades = run_backtest(enriched, exit_mode="fixed_bars", fixed_bars=fixed_bars)

    if trades.height == 0:
        empty_scores = pl.DataFrame(
            {
                "entry_time": pl.Series([], dtype=enriched["timestamp"].dtype),
                "exit_time": pl.Series([], dtype=enriched["timestamp"].dtype),
                "alpha_score": pl.Series([], dtype=pl.Float64),
                "trend_score": pl.Series([], dtype=pl.Float64),
                "momentum_score": pl.Series([], dtype=pl.Float64),
                "volatility_score": pl.Series([], dtype=pl.Float64),
                "vwap_score": pl.Series([], dtype=pl.Float64),
                "market_structure_score": pl.Series([], dtype=pl.Float64),
                "opening_range_score": pl.Series([], dtype=pl.Float64),
                "time_score": pl.Series([], dtype=pl.Float64),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "score_bucket": pl.Series([], dtype=pl.Utf8),
            }
        )
        score_analysis = build_score_analysis(empty_scores)
        validation = build_score_validation(empty_scores, score_analysis)
        top_trades, bottom_trades = build_top_bottom_trades(empty_scores)
        alpha_summary = build_alpha_summary(empty_scores)
        write_alpha_reports(
            output_dir=output_dir,
            scores=empty_scores,
            score_analysis=score_analysis,
            validation=validation,
            top_trades=top_trades,
            bottom_trades=bottom_trades,
            alpha_summary=alpha_summary,
        )
        return AlphaScoringResult(
            scores=empty_scores,
            score_analysis=score_analysis,
            validation=validation,
            top_trades=top_trades,
            bottom_trades=bottom_trades,
            alpha_summary=alpha_summary,
        )

    label_columns = [
        spec.feature.label(enriched).alias(f"ctx_{name}")
        for name, spec in alpha_registry.items()
    ]
    context_labels = enriched.select(["timestamp", *label_columns])

    trade_contexts = trades.join(
        context_labels,
        left_on="entry_time",
        right_on="timestamp",
        how="left",
    )

    scored_trades = score_trades(trade_contexts, weights)
    scores = build_scores_report(scored_trades)
    score_analysis = build_score_analysis(scored_trades)
    validation = build_score_validation(scored_trades, score_analysis)
    top_trades, bottom_trades = build_top_bottom_trades(scored_trades)
    alpha_summary = build_alpha_summary(scored_trades)

    write_alpha_reports(
        output_dir=output_dir,
        scores=scores,
        score_analysis=score_analysis,
        validation=validation,
        top_trades=top_trades,
        bottom_trades=bottom_trades,
        alpha_summary=alpha_summary,
    )

    return AlphaScoringResult(
        scores=scores,
        score_analysis=score_analysis,
        validation=validation,
        top_trades=top_trades,
        bottom_trades=bottom_trades,
        alpha_summary=alpha_summary,
    )
