"""Report builders and writers for Alpha Scoring Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.alpha.registry import CATEGORY_ORDER

_BUCKET_ORDER = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def build_scores_report(scored_trades: pl.DataFrame) -> pl.DataFrame:
    """Select and order required columns for scores.csv."""
    ordered = scored_trades.select(
        [
            "entry_time",
            "exit_time",
            "alpha_score",
            "trend_score",
            "momentum_score",
            "volatility_score",
            "vwap_score",
            "market_structure_score",
            "opening_range_score",
            "time_score",
            "return_pct",
            "score_bucket",
        ]
    )
    return ordered.sort("entry_time")


def build_score_analysis(scored_trades: pl.DataFrame) -> pl.DataFrame:
    """Build per-alpha-bucket performance analysis."""
    rows: list[dict[str, Any]] = []

    for bucket in _BUCKET_ORDER:
        subset = scored_trades.filter(pl.col("score_bucket") == bucket)
        returns = subset.get_column("return_pct") if "return_pct" in subset.columns else pl.Series([], dtype=pl.Float64)
        if subset.height == 0:
            rows.append(
                {
                    "score_bucket": bucket,
                    "number_of_trades": 0,
                    "win_rate": None,
                    "average_return": None,
                    "median_return": None,
                    "expectancy": None,
                    "profit_factor": None,
                    "sharpe": None,
                    "maximum_drawdown": None,
                }
            )
            continue

        wins = returns.filter(returns > 0)
        losses = returns.filter(returns <= 0)
        n = subset.height
        win_rate = wins.len() / n
        avg_win = float(wins.mean()) if wins.len() > 0 else 0.0
        avg_loss = float(losses.mean()) if losses.len() > 0 else 0.0
        gross_wins = float(wins.sum()) if wins.len() > 0 else 0.0
        gross_losses = abs(float(losses.sum())) if losses.len() > 0 else 0.0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else None
        expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

        std_return = float(returns.std(ddof=1)) if n > 1 else 0.0
        average_return = float(returns.mean())
        sharpe = average_return / std_return if std_return > 0 else 0.0

        equity = returns.cum_sum().to_list()
        peak = equity[0] if equity else 0.0
        max_drawdown = 0.0
        for value in equity:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        rows.append(
            {
                "score_bucket": bucket,
                "number_of_trades": n,
                "win_rate": round(win_rate, 6),
                "average_return": round(average_return, 6),
                "median_return": round(float(returns.median()), 6),
                "expectancy": round(expectancy, 6),
                "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
                "sharpe": round(sharpe, 6),
                "maximum_drawdown": round(max_drawdown, 6),
            }
        )

    return pl.DataFrame(rows)


def build_alpha_summary(scored_trades: pl.DataFrame) -> dict[str, Any]:
    """Build aggregate alpha-score summary statistics."""
    if scored_trades.height == 0:
        return {
            "number_of_trades": 0,
            "mean_alpha_score": None,
            "median_alpha_score": None,
            "highest_score": None,
            "lowest_score": None,
        }

    alpha_scores = scored_trades.get_column("alpha_score")
    return {
        "number_of_trades": scored_trades.height,
        "mean_alpha_score": round(float(alpha_scores.mean()), 6),
        "median_alpha_score": round(float(alpha_scores.median()), 6),
        "highest_score": round(float(alpha_scores.max()), 6),
        "lowest_score": round(float(alpha_scores.min()), 6),
    }


def build_top_bottom_trades(scored_trades: pl.DataFrame, top_n: int = 20) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build top and bottom alpha-score trade extracts."""
    trade_columns = [
        "entry_time",
        "exit_time",
        "alpha_score",
        *[f"{category}_score" for category in CATEGORY_ORDER],
        "return_pct",
    ]
    selected = scored_trades.select(trade_columns)
    top_trades = selected.sort("alpha_score", descending=True).head(top_n)
    bottom_trades = selected.sort("alpha_score", descending=False).head(top_n)
    return top_trades, bottom_trades


def write_alpha_reports(
    output_dir: Path,
    scores: pl.DataFrame,
    score_analysis: pl.DataFrame,
    validation: dict[str, Any],
    top_trades: pl.DataFrame,
    bottom_trades: pl.DataFrame,
    alpha_summary: dict[str, Any],
) -> None:
    """Write all Alpha Scoring Engine output artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scores.select(
        [
            "entry_time",
            "exit_time",
            "alpha_score",
            "trend_score",
            "momentum_score",
            "volatility_score",
            "vwap_score",
            "market_structure_score",
            "opening_range_score",
            "time_score",
            "return_pct",
        ]
    ).write_csv(output_dir / "scores.csv")
    score_analysis.write_csv(output_dir / "score_analysis.csv")
    top_trades.write_csv(output_dir / "top_trades.csv")
    bottom_trades.write_csv(output_dir / "bottom_trades.csv")
    (output_dir / "score_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )
    (output_dir / "alpha_summary.json").write_text(
        json.dumps(alpha_summary, indent=2),
        encoding="utf-8",
    )
