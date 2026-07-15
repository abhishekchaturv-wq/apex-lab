"""Market structure feature group.

Identifies swing highs and lows using a rolling window approach, then derives
higher-high / lower-high / higher-low / lower-low regime flags.

Required columns: ``high``, ``low``

Computed features
-----------------
- ``swing_high``   — Rolling maximum of ``high`` over *n* bars
- ``swing_low``    — Rolling minimum of ``low`` over *n* bars
- ``higher_high``  — 1 if current swing_high > previous swing_high, else 0
- ``lower_high``   — 1 if current swing_high < previous swing_high, else 0
- ``higher_low``   — 1 if current swing_low  > previous swing_low,  else 0
- ``lower_low``    — 1 if current swing_low  < previous swing_low,  else 0
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_SWING_WINDOW = 10  # look-back bars for swing detection


class StructureFeatures(FeatureGroup):
    """Swing high / low and HH / LH / HL / LL regime flags.

    Attributes:
        name: ``"structure"``
        warm_up_periods: ``10`` (swing detection window)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "structure"

    @property
    def warm_up_periods(self) -> int:
        """Return swing detection window."""
        return _SWING_WINDOW

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute market structure features and append them to *df*.

        Args:
            df: DataFrame that must contain ``high`` and ``low``.

        Returns:
            Input DataFrame with 6 additional feature columns.

        Raises:
            ValueError: If any required column is absent.
        """
        self._require_columns(df, ["high", "low"])
        logger.debug("[structure] Computing structure features for %d rows", len(df))

        # ------------------------------------------------------------------ #
        # Swing high / low via rolling window                                  #
        # ------------------------------------------------------------------ #
        swing_high = pl.col("high").rolling_max(window_size=_SWING_WINDOW).alias("swing_high")
        swing_low = pl.col("low").rolling_min(window_size=_SWING_WINDOW).alias("swing_low")

        df = df.with_columns([swing_high, swing_low])

        # ------------------------------------------------------------------ #
        # Regime comparison flags                                              #
        # ------------------------------------------------------------------ #
        prev_sh = pl.col("swing_high").shift(1)
        prev_sl = pl.col("swing_low").shift(1)

        higher_high = (
            pl.when(pl.col("swing_high") > prev_sh).then(pl.lit(1)).otherwise(pl.lit(0))
        ).alias("higher_high")

        lower_high = (
            pl.when(pl.col("swing_high") < prev_sh).then(pl.lit(1)).otherwise(pl.lit(0))
        ).alias("lower_high")

        higher_low = (
            pl.when(pl.col("swing_low") > prev_sl).then(pl.lit(1)).otherwise(pl.lit(0))
        ).alias("higher_low")

        lower_low = (
            pl.when(pl.col("swing_low") < prev_sl).then(pl.lit(1)).otherwise(pl.lit(0))
        ).alias("lower_low")

        df = df.with_columns([higher_high, lower_high, higher_low, lower_low])

        logger.debug(
            "[structure] Done – new columns: swing_high, swing_low, "
            "higher_high, lower_high, higher_low, lower_low"
        )
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(StructureFeatures())
