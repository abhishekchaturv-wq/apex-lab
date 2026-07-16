"""Validation helpers for Alpha Scoring Engine."""

from __future__ import annotations

from typing import Any

import polars as pl

_BUCKET_ORDER = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def build_score_validation(scored_trades: pl.DataFrame, score_analysis: pl.DataFrame) -> dict[str, Any]:
    """Build score-return correlation and monotonicity validation summary."""
    pair = scored_trades.select(["alpha_score", "return_pct"]).drop_nulls()
    pearson = None
    spearman = None
    if pair.height >= 2:
        pearson = float(pair.select(pl.corr("alpha_score", "return_pct", method="pearson")).item())
        spearman = float(pair.select(pl.corr("alpha_score", "return_pct", method="spearman")).item())

    expectancy_by_bucket = {
        row["score_bucket"]: row["expectancy"] for row in score_analysis.iter_rows(named=True)
    }
    ordered_expectancy = [expectancy_by_bucket.get(bucket) for bucket in _BUCKET_ORDER]
    non_null_expectancy = [value for value in ordered_expectancy if value is not None]

    monotonicity = True
    if len(non_null_expectancy) >= 2:
        monotonicity = all(
            right >= left
            for left, right in zip(non_null_expectancy, non_null_expectancy[1:], strict=False)
        )

    highest_bucket_expectancy = expectancy_by_bucket.get("80-100")
    lowest_bucket_expectancy = expectancy_by_bucket.get("0-20")

    return {
        "pearson": round(pearson, 6) if pearson is not None else None,
        "spearman": round(spearman, 6) if spearman is not None else None,
        "highest_bucket_expectancy": (
            round(float(highest_bucket_expectancy), 6)
            if highest_bucket_expectancy is not None
            else None
        ),
        "lowest_bucket_expectancy": (
            round(float(lowest_bucket_expectancy), 6)
            if lowest_bucket_expectancy is not None
            else None
        ),
        "monotonicity": monotonicity,
    }
