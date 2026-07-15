"""Deterministic objective-reversal labeling engine."""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.labels.rules import LabelingRules
from apex_lab.labels.targets import LabelType

logger = logging.getLogger(__name__)


class LabelEngine:
    """Generate supervised learning labels from objective future-price rules.

    A candle is labeled ``BOTTOM`` when upward reward threshold is reached before
    downward risk threshold. ``TOP`` is the mirror condition.
    """

    def __init__(self, rules: LabelingRules | None = None) -> None:
        """Initialize the labeling engine.

        Args:
            rules: Rule configuration. Defaults to :class:`LabelingRules`.
        """
        self.rules = rules or LabelingRules()

    def label(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append labels and target metadata to a price DataFrame.

        Args:
            df: Input DataFrame with ``high``, ``low``, ``close`` and ATR column.

        Returns:
            Input DataFrame with target columns appended.

        Raises:
            ValueError: If required columns are missing.
        """
        self._require_columns(df, ["high", "low", "close", self.rules.atr_column])

        atr = pl.col(self.rules.atr_column)
        scaled_atr = atr * self.rules.atr_multiplier

        reward_threshold = scaled_atr * self.rules.reward_multiplier
        risk_threshold = scaled_atr * self.rules.risk_multiplier

        bottom_target_level = pl.col("low") + reward_threshold
        bottom_failure_level = pl.col("low") - risk_threshold
        top_target_level = pl.col("high") - reward_threshold
        top_failure_level = pl.col("high") + risk_threshold

        lookahead = self.rules.lookahead_window

        bottom_target_candidates = [
            pl.when(pl.col("high").shift(-bar) >= bottom_target_level)
            .then(pl.lit(bar))
            .otherwise(None)
            for bar in range(1, lookahead + 1)
        ]
        bottom_failure_candidates = [
            pl.when(pl.col("low").shift(-bar) <= bottom_failure_level)
            .then(pl.lit(bar))
            .otherwise(None)
            for bar in range(1, lookahead + 1)
        ]
        top_target_candidates = [
            pl.when(pl.col("low").shift(-bar) <= top_target_level)
            .then(pl.lit(bar))
            .otherwise(None)
            for bar in range(1, lookahead + 1)
        ]
        top_failure_candidates = [
            pl.when(pl.col("high").shift(-bar) >= top_failure_level)
            .then(pl.lit(bar))
            .otherwise(None)
            for bar in range(1, lookahead + 1)
        ]

        bottom_target = pl.min_horizontal(bottom_target_candidates).alias("_bottom_target")
        bottom_failure = pl.min_horizontal(bottom_failure_candidates).alias("_bottom_failure")
        top_target = pl.min_horizontal(top_target_candidates).alias("_top_target")
        top_failure = pl.min_horizontal(top_failure_candidates).alias("_top_failure")

        future_close = pl.col("close").shift(-lookahead)
        future_return = ((future_close - pl.col("close")) / pl.col("close")).alias("future_return")

        labeled = df.with_columns([bottom_target, bottom_failure, top_target, top_failure]).with_columns(
            [
                (
                    pl.col("_bottom_target").is_not_null()
                    & (pl.col("_bottom_failure").is_null() | (pl.col("_bottom_target") < pl.col("_bottom_failure")))
                ).alias("_is_bottom"),
                (
                    pl.col("_top_target").is_not_null()
                    & (pl.col("_top_failure").is_null() | (pl.col("_top_target") < pl.col("_top_failure")))
                ).alias("_is_top"),
            ]
        ).with_columns(
            [
                (
                    pl.col("_is_bottom")
                    & (~pl.col("_is_top") | (pl.col("_bottom_target") < pl.col("_top_target")))
                ).alias("_label_bottom"),
                (
                    pl.col("_is_top")
                    & (~pl.col("_is_bottom") | (pl.col("_top_target") < pl.col("_bottom_target")))
                ).alias("_label_top"),
            ]
        ).with_columns(
            [
                pl.when(pl.col("_label_bottom"))
                .then(pl.lit(LabelType.BOTTOM.value))
                .when(pl.col("_label_top"))
                .then(pl.lit(LabelType.TOP.value))
                .otherwise(pl.lit(LabelType.NONE.value))
                .alias("label"),
                pl.when(pl.col("_label_bottom"))
                .then(pl.col("_bottom_target"))
                .when(pl.col("_label_top"))
                .then(pl.col("_top_target"))
                .otherwise(None)
                .cast(pl.Int64)
                .alias("bars_to_target"),
                pl.when(pl.col("_label_bottom"))
                .then(pl.col("_bottom_failure"))
                .when(pl.col("_label_top"))
                .then(pl.col("_top_failure"))
                .otherwise(None)
                .cast(pl.Int64)
                .alias("bars_to_failure"),
                future_return,
            ]
        ).with_columns(
            pl.when(pl.col("label") == LabelType.NONE.value)
            .then(pl.lit(0.0))
            .when(pl.col("bars_to_failure").is_null())
            .then(pl.lit(1.0))
            .otherwise(
                ((pl.col("bars_to_failure") - pl.col("bars_to_target")).cast(pl.Float64) / lookahead).clip(0.0, 1.0)
            )
            .alias("confidence")
        )

        logger.info("Label generation complete for %d rows", len(df))

        return labeled.drop(
            [
                "_bottom_target",
                "_bottom_failure",
                "_top_target",
                "_top_failure",
                "_is_bottom",
                "_is_top",
                "_label_bottom",
                "_label_top",
            ]
        )

    @staticmethod
    def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
        """Validate that required columns exist in the DataFrame."""
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
