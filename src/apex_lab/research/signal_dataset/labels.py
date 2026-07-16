"""Label generation for signal discovery datasets."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SignalLabelConfig:
    """Configurable thresholds used to derive direction and signal labels."""

    horizons: tuple[int, ...] = (5, 10, 20, 40)
    excursion_horizon: int = 40
    direction_horizon: int = 20
    direction_flat_threshold: float = 0.1
    strong_bull_threshold: float = 1.5
    bull_threshold: float = 0.4
    bear_threshold: float = -0.4
    strong_bear_threshold: float = -1.5

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons must be non-empty")
        if self.excursion_horizon <= 0:
            raise ValueError("excursion_horizon must be > 0")
        if self.direction_horizon <= 0:
            raise ValueError("direction_horizon must be > 0")
        if self.strong_bear_threshold >= self.bear_threshold:
            raise ValueError("strong_bear_threshold must be < bear_threshold")
        if self.bear_threshold >= self.bull_threshold:
            raise ValueError("bear_threshold must be < bull_threshold")
        if self.bull_threshold >= self.strong_bull_threshold:
            raise ValueError("bull_threshold must be < strong_bull_threshold")


def append_signal_classes(df: pl.DataFrame, config: SignalLabelConfig) -> pl.DataFrame:
    """Append future-looking labels for supervised signal discovery."""
    labeled = df.with_columns(_future_return_expr(h).alias(f"future_return_{h}") for h in config.horizons)

    horizon = config.excursion_horizon
    future_high = _future_extreme_return_expr("high", horizon).alias("future_high_return")
    future_low = _future_extreme_return_expr("low", horizon).alias("future_low_return")

    direction_col = f"future_return_{config.direction_horizon}"
    if direction_col not in labeled.columns:
        labeled = labeled.with_columns(_future_return_expr(config.direction_horizon).alias(direction_col))

    return (
        labeled.with_columns([future_high, future_low])
        .with_columns(
            [
                pl.col("future_high_return").alias("maximum_favorable_excursion"),
                pl.col("future_low_return").alias("maximum_adverse_excursion"),
                _direction_expr(direction_col, config).alias("direction"),
                _signal_class_expr(direction_col, config).alias("signal_class"),
            ]
        )
        .with_columns(
            [
                (pl.col("signal_class") == "Strong Bull Move")
                .cast(pl.Int8)
                .alias("label_strong_bull_move"),
                (pl.col("signal_class") == "Bull Move").cast(pl.Int8).alias("label_bull_move"),
                (pl.col("signal_class") == "Neutral").cast(pl.Int8).alias("label_neutral"),
                (pl.col("signal_class") == "Bear Move").cast(pl.Int8).alias("label_bear_move"),
                (pl.col("signal_class") == "Strong Bear Move")
                .cast(pl.Int8)
                .alias("label_strong_bear_move"),
            ]
        )
    )


def label_columns(config: SignalLabelConfig) -> list[str]:
    """Return all label column names for the configured horizons."""
    future_return_columns = [f"future_return_{h}" for h in config.horizons]
    return [
        *future_return_columns,
        "future_high_return",
        "future_low_return",
        "maximum_favorable_excursion",
        "maximum_adverse_excursion",
        "direction",
        "signal_class",
        "label_strong_bull_move",
        "label_bull_move",
        "label_neutral",
        "label_bear_move",
        "label_strong_bear_move",
    ]


def _future_return_expr(horizon: int) -> pl.Expr:
    return ((pl.col("close").shift(-horizon) / (pl.col("close") + 1e-9)) - 1.0).mul(100.0)


def _future_extreme_return_expr(column: str, horizon: int) -> pl.Expr:
    future_values = pl.concat_list([pl.col(column).shift(-offset) for offset in range(1, horizon + 1)])
    extreme = future_values.list.max() if column == "high" else future_values.list.min()
    return ((extreme - pl.col("close")) / (pl.col("close") + 1e-9)).mul(100.0)


def _direction_expr(return_column: str, config: SignalLabelConfig) -> pl.Expr:
    return (
        pl.when(pl.col(return_column).is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col(return_column) >= config.direction_flat_threshold)
        .then(pl.lit("Bull"))
        .when(pl.col(return_column) <= -config.direction_flat_threshold)
        .then(pl.lit("Bear"))
        .otherwise(pl.lit("Flat"))
    )


def _signal_class_expr(return_column: str, config: SignalLabelConfig) -> pl.Expr:
    return (
        pl.when(pl.col(return_column).is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col(return_column) >= config.strong_bull_threshold)
        .then(pl.lit("Strong Bull Move"))
        .when(pl.col(return_column) >= config.bull_threshold)
        .then(pl.lit("Bull Move"))
        .when(pl.col(return_column) <= config.strong_bear_threshold)
        .then(pl.lit("Strong Bear Move"))
        .when(pl.col(return_column) <= config.bear_threshold)
        .then(pl.lit("Bear Move"))
        .otherwise(pl.lit("Neutral"))
    )
