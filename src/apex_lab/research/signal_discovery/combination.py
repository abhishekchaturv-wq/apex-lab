"""Feature-combination ranking for signal discovery."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import polars as pl

from apex_lab.research.signal_discovery.statistics import (
    discretize_series,
    normalize_series,
    predictive_power,
)


@dataclass(frozen=True)
class CombinationConfig:
    """Configuration for top combination search."""

    top_feature_candidates: int = 12
    top_combinations: int = 20


def rank_feature_combinations(
    df: pl.DataFrame,
    importance: pl.DataFrame,
    config: CombinationConfig = CombinationConfig(),
    target_column: str = "future_return_20",
) -> pl.DataFrame:
    """Evaluate top 2- and 3-feature combinations using strongest features only."""
    if importance.is_empty() or target_column not in df.columns:
        return pl.DataFrame(
            {
                "rank": pl.Series([], dtype=pl.Int64),
                "combination": pl.Series([], dtype=pl.Utf8),
                "combination_size": pl.Series([], dtype=pl.Int64),
                "best_bucket": pl.Series([], dtype=pl.Utf8),
                "trade_frequency": pl.Series([], dtype=pl.Int64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "predictive_power": pl.Series([], dtype=pl.Float64),
                "combination_score": pl.Series([], dtype=pl.Float64),
            }
        )

    candidates = (
        importance.select("feature")
        .head(config.top_feature_candidates)
        .get_column("feature")
        .to_list()
    )
    available = [feature for feature in candidates if feature in df.columns]
    if len(available) < 2:
        return pl.DataFrame(
            {
                "rank": pl.Series([], dtype=pl.Int64),
                "combination": pl.Series([], dtype=pl.Utf8),
                "combination_size": pl.Series([], dtype=pl.Int64),
                "best_bucket": pl.Series([], dtype=pl.Utf8),
                "trade_frequency": pl.Series([], dtype=pl.Int64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "predictive_power": pl.Series([], dtype=pl.Float64),
                "combination_score": pl.Series([], dtype=pl.Float64),
            }
        )

    rows: list[dict[str, Any]] = []

    for size in (2, 3):
        if len(available) < size:
            continue
        for combo in combinations(available, size):
            bucket_columns: list[str] = []
            select_columns = [pl.col(target_column).alias("target")]

            for feature in combo:
                bucket_name = f"bucket_{feature}"
                select_columns.append(discretize_series(df.get_column(feature), bins=4).alias(bucket_name))
                bucket_columns.append(bucket_name)

            scoped = df.select(select_columns).drop_nulls(subset=["target"])
            if scoped.height < 10:
                continue

            grouped = (
                scoped.group_by(bucket_columns)
                .agg(
                    [
                        pl.len().alias("trade_frequency"),
                        pl.col("target").mean().alias("average_return"),
                        (pl.col("target") > 0.0).mean().alias("win_rate"),
                    ]
                )
                .with_columns(pl.col("average_return").alias("expectancy"))
                .sort(["expectancy", "trade_frequency"], descending=[True, True])
            )

            if grouped.is_empty():
                continue

            best = grouped.row(0, named=True)
            best_bucket = " | ".join(str(best[column]) for column in bucket_columns)
            scoped_with_key = scoped.with_columns(
                pl.concat_str([pl.col(column) for column in bucket_columns], separator="|", ignore_nulls=False)
                .fill_null("null")
                .alias("combo_key")
            )
            power = predictive_power(
                feature=scoped_with_key.get_column("combo_key"),
                target=scoped_with_key.get_column("target"),
                categorical_feature=True,
            )

            rows.append(
                {
                    "combination": " + ".join(combo),
                    "combination_size": size,
                    "best_bucket": best_bucket,
                    "trade_frequency": int(best["trade_frequency"]),
                    "win_rate": float(best["win_rate"]),
                    "average_return": float(best["average_return"]),
                    "expectancy": float(best["expectancy"]),
                    "predictive_power": power,
                }
            )

    if not rows:
        return pl.DataFrame(
            {
                "rank": pl.Series([], dtype=pl.Int64),
                "combination": pl.Series([], dtype=pl.Utf8),
                "combination_size": pl.Series([], dtype=pl.Int64),
                "best_bucket": pl.Series([], dtype=pl.Utf8),
                "trade_frequency": pl.Series([], dtype=pl.Int64),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "predictive_power": pl.Series([], dtype=pl.Float64),
                "combination_score": pl.Series([], dtype=pl.Float64),
            }
        )

    table = pl.DataFrame(rows)
    predictive_norm = normalize_series(table.get_column("predictive_power").to_list())
    expectancy_norm = normalize_series(table.get_column("expectancy").to_list())
    win_rate_norm = normalize_series(table.get_column("win_rate").to_list())
    frequency_norm = normalize_series(table.get_column("trade_frequency").cast(pl.Float64).to_list())

    ranked = table.with_columns(
        [
            pl.Series("_predictive_norm", predictive_norm, dtype=pl.Float64),
            pl.Series("_expectancy_norm", expectancy_norm, dtype=pl.Float64),
            pl.Series("_win_rate_norm", win_rate_norm, dtype=pl.Float64),
            pl.Series("_frequency_norm", frequency_norm, dtype=pl.Float64),
        ]
    ).with_columns(
        (
            (pl.col("_predictive_norm") * 0.40)
            + (pl.col("_expectancy_norm") * 0.30)
            + (pl.col("_win_rate_norm") * 0.20)
            + (pl.col("_frequency_norm") * 0.10)
        ).alias("combination_score")
    )

    return (
        ranked.sort("combination_score", descending=True)
        .head(config.top_combinations)
        .with_row_index("rank", offset=1)
        .select(
            [
                "rank",
                "combination",
                "combination_size",
                "best_bucket",
                "trade_frequency",
                "win_rate",
                "average_return",
                "expectancy",
                "predictive_power",
                "combination_score",
            ]
        )
        .with_columns(
            [
                pl.col("win_rate").round(6),
                pl.col("average_return").round(6),
                pl.col("expectancy").round(6),
                pl.col("predictive_power").round(6),
                pl.col("combination_score").round(6),
            ]
        )
    )
