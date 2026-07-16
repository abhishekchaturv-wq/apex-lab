"""Report generation for the Alpha Discovery Engine.

Produces four output artefacts:

- ``summary.csv`` — one row per feature-bucket with all performance metrics.
- ``leaderboard.csv`` — valid buckets (sample_size ≥ 30) ranked by composite score.
- ``best_features.json`` — best-performing bucket per feature group.
- ``correlation.csv`` — Pearson and Spearman correlation of each feature with return.

Scoring formula (documented)
-----------------------------
All three metrics are min-max normalised across the set of valid (sample_size ≥ 30) rows::

    score = 0.40 × normalised_profit_factor
          + 0.30 × normalised_expectancy
          + 0.30 × normalised_sharpe

Normalisation: ``(value − min) / (max − min)`` per column; collapses to 0.0 when
all values are identical.

Correlation
-----------
For each feature the numeric series (raw indicator values at trade entry) is
correlated against ``return_pct`` using:

- Pearson correlation (linear relationship).
- Spearman correlation (monotone relationship; computed as Pearson on ranks).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

_SCORE_WEIGHTS = {"profit_factor": 0.40, "expectancy": 0.30, "sharpe": 0.30}


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def build_leaderboard(summary: pl.DataFrame) -> pl.DataFrame:
    """Rank all statistically valid feature buckets.

    Only rows where ``low_sample_size = False`` participate in ranking.

    Args:
        summary: DataFrame produced by :func:`~apex_lab.research.context.metrics.compute_all_bucket_metrics`.

    Returns:
        DataFrame with columns: ``rank``, ``feature``, ``bucket``,
        ``sample_size``, ``win_rate``, ``expectancy``, ``profit_factor``,
        ``sharpe``, ``score``.
    """
    if summary.is_empty():
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

    valid = summary.filter(pl.col("low_sample_size").is_not_null() & ~pl.col("low_sample_size"))

    if valid.is_empty():
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

    # Replace null profit_factor with 0.0 before scoring
    scored = valid.with_columns(
        pl.col("profit_factor").fill_null(0.0)
    )

    # Min-max normalise the three scoring metrics
    for metric in ("profit_factor", "expectancy", "sharpe"):
        col_min = float(scored[metric].min())  # type: ignore[arg-type]
        col_max = float(scored[metric].max())  # type: ignore[arg-type]
        rng = col_max - col_min
        if rng > 0:
            scored = scored.with_columns(
                ((pl.col(metric) - col_min) / rng).alias(f"_norm_{metric}")
            )
        else:
            scored = scored.with_columns(pl.lit(0.0).alias(f"_norm_{metric}"))

    scored = scored.with_columns(
        (
            pl.lit(_SCORE_WEIGHTS["profit_factor"]) * pl.col("_norm_profit_factor")
            + pl.lit(_SCORE_WEIGHTS["expectancy"]) * pl.col("_norm_expectancy")
            + pl.lit(_SCORE_WEIGHTS["sharpe"]) * pl.col("_norm_sharpe")
        ).alias("score")
    )

    return (
        scored.sort("score", descending=True)
        .with_row_index("rank", offset=1)
        .select(
            [
                "rank",
                "feature",
                "bucket",
                "sample_size",
                "win_rate",
                "expectancy",
                "profit_factor",
                "sharpe",
                "score",
            ]
        )
        .with_columns(pl.col("score").round(6))
    )


# ---------------------------------------------------------------------------
# Best features
# ---------------------------------------------------------------------------


def build_best_features(
    leaderboard: pl.DataFrame,
    summary: pl.DataFrame,
    registry: dict[str, Any],
) -> dict[str, str]:
    """Select the best-performing bucket for each feature group.

    Valid (sample_size ≥ 30) buckets from *leaderboard* are preferred.
    Falls back to the highest ``win_rate`` bucket in *summary* when the
    leaderboard has no valid entry for a group.

    Args:
        leaderboard: Output of :func:`build_leaderboard`.
        summary: Full summary including low-sample rows.
        registry: Feature registry dict mapping name → ContextFeature.

    Returns:
        Dict mapping group name (or feature name) → best bucket label.
    """
    best: dict[str, str] = {}

    for fname, _feature in registry.items():
        # 1) Try leaderboard (already filtered to sample_size >= 30)
        lb_rows = leaderboard.filter(pl.col("feature") == fname)
        if lb_rows.height > 0:
            best_row = lb_rows.sort("score", descending=True).row(0, named=True)
            best[fname] = str(best_row["bucket"])
            continue

        # 2) Fallback: highest win_rate from summary (any sample size)
        sm_rows = summary.filter(pl.col("feature") == fname)
        if sm_rows.height > 0:
            best_row = sm_rows.sort("win_rate", descending=True).row(0, named=True)
            best[fname] = str(best_row["bucket"])

    return best


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def build_correlation(
    trade_contexts: pl.DataFrame,
    registry: dict[str, Any],
) -> pl.DataFrame:
    """Compute Pearson and Spearman correlation of each feature with return.

    Args:
        trade_contexts: Per-trade DataFrame with ``return_pct`` and
            ``num_{feature_name}`` columns for each feature.
        registry: Feature registry dict mapping name → ContextFeature.

    Returns:
        DataFrame with columns: ``feature``, ``pearson``, ``spearman``.
    """
    rows: list[dict[str, Any]] = []
    returns = trade_contexts["return_pct"]

    for fname in registry:
        num_col = f"num_{fname}"
        if num_col not in trade_contexts.columns:
            continue

        feature_vals = trade_contexts[num_col]

        # Drop rows where either column is null
        pair = pl.DataFrame({"feat": feature_vals, "ret": returns}).drop_nulls()
        if pair.height < 2:
            rows.append({"feature": fname, "pearson": None, "spearman": None})
            continue

        pearson = float(pair.select(pl.corr("feat", "ret", method="pearson")).item())

        # Spearman = Pearson of ranks
        spearman = float(pair.select(pl.corr("feat", "ret", method="spearman")).item())

        rows.append(
            {
                "feature": fname,
                "pearson": round(pearson, 6),
                "spearman": round(spearman, 6),
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "feature": pl.Series([], dtype=pl.Utf8),
                "pearson": pl.Series([], dtype=pl.Float64),
                "spearman": pl.Series([], dtype=pl.Float64),
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_context_reports(
    summary: pl.DataFrame,
    leaderboard: pl.DataFrame,
    best_features: dict[str, str],
    correlation: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Write all four context research artefacts to *output_dir*.

    Args:
        summary: Per-bucket summary DataFrame.
        leaderboard: Ranked valid-bucket leaderboard DataFrame.
        best_features: Best-bucket dict per feature.
        correlation: Correlation DataFrame.
        output_dir: Directory for all output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    leaderboard_path = output_dir / "leaderboard.csv"
    best_path = output_dir / "best_features.json"
    corr_path = output_dir / "correlation.csv"

    summary.write_csv(summary_path)
    leaderboard.write_csv(leaderboard_path)
    best_path.write_text(json.dumps(best_features, indent=2), encoding="utf-8")
    correlation.write_csv(corr_path)

    logger.info("summary.csv written to %s (%d rows)", summary_path, summary.height)
    logger.info("leaderboard.csv written to %s (%d rows)", leaderboard_path, leaderboard.height)
    logger.info("best_features.json written to %s", best_path)
    logger.info("correlation.csv written to %s (%d rows)", corr_path, correlation.height)
