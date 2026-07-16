"""Score computation for Alpha Scoring Engine."""

from __future__ import annotations

import polars as pl

from apex_lab.research.alpha.registry import CATEGORY_ORDER
from apex_lab.research.alpha.weights import FeatureWeight


def score_trades(trade_contexts: pl.DataFrame, weights: list[FeatureWeight]) -> pl.DataFrame:
    """Assign category scores and final alpha score for each trade."""
    category_exprs: dict[str, list[pl.Expr]] = {category: [] for category in CATEGORY_ORDER}

    for weight in weights:
        ctx_col = f"ctx_{weight.feature}"
        if ctx_col not in trade_contexts.columns:
            continue
        category_exprs.setdefault(weight.category, []).append(
            pl.when(pl.col(ctx_col) == pl.lit(weight.bucket))
            .then(pl.lit(weight.weight))
            .otherwise(pl.lit(0.0))
        )

    category_columns: list[pl.Expr] = []
    for category in CATEGORY_ORDER:
        exprs = category_exprs.get(category, [])
        if exprs:
            score_expr = pl.sum_horizontal(exprs)
        else:
            score_expr = pl.lit(0.0)
        category_columns.append(score_expr.cast(pl.Float64).alias(f"{category}_score"))

    scored = trade_contexts.with_columns(category_columns).with_columns(
        pl.sum_horizontal([pl.col(f"{category}_score") for category in CATEGORY_ORDER])
        .clip(0.0, 100.0)
        .round(6)
        .alias("alpha_score")
    )

    score_bucket = (
        pl.when(pl.col("alpha_score") < 20.0)
        .then(pl.lit("0-20"))
        .when(pl.col("alpha_score") < 40.0)
        .then(pl.lit("20-40"))
        .when(pl.col("alpha_score") < 60.0)
        .then(pl.lit("40-60"))
        .when(pl.col("alpha_score") < 80.0)
        .then(pl.lit("60-80"))
        .otherwise(pl.lit("80-100"))
        .alias("score_bucket")
    )
    return scored.with_columns(score_bucket)
