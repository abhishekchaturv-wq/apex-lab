"""Volume feature group.

Computes On-Balance Volume and relative volume indicators.

Required columns: ``close``, ``volume``

Computed features
-----------------
- ``obv``           — On-Balance Volume (cumulative)
- ``rel_volume``    — Volume relative to 20-bar rolling average
- ``avg_volume``    — 20-bar rolling average volume
- ``volume_spike``  — Binary flag: volume > 2× rolling average (1.0 / 0.0)
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_AVG_VOL_WINDOW = 20
_SPIKE_MULTIPLIER = 2.0
_EPSILON = 1e-10


class VolumeFeatures(FeatureGroup):
    """On-Balance Volume and relative volume indicators.

    Attributes:
        name: ``"volume"``
        warm_up_periods: ``20`` (rolling average window)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "volume"

    @property
    def warm_up_periods(self) -> int:
        """Return rolling average volume window."""
        return _AVG_VOL_WINDOW

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute volume features and append them to *df*.

        Args:
            df: DataFrame that must contain ``close`` and ``volume``.

        Returns:
            Input DataFrame with 4 additional feature columns.

        Raises:
            ValueError: If any required column is absent.
        """
        self._require_columns(df, ["close", "volume"])
        logger.debug("[volume] Computing volume features for %d rows", len(df))

        float_volume = pl.col("volume").cast(pl.Float64)
        epsilon = pl.lit(_EPSILON)

        # ------------------------------------------------------------------ #
        # OBV                                                                  #
        # ------------------------------------------------------------------ #
        close_diff = pl.col("close").diff()
        signed_volume = (
            pl.when(close_diff > 0)
            .then(float_volume)
            .when(close_diff < 0)
            .then(-float_volume)
            .otherwise(pl.lit(0.0))
        )
        obv = signed_volume.cum_sum().alias("obv")

        # ------------------------------------------------------------------ #
        # Rolling average volume and relatives                                 #
        # ------------------------------------------------------------------ #
        avg_volume = float_volume.rolling_mean(window_size=_AVG_VOL_WINDOW).alias("avg_volume")

        df = df.with_columns([obv, avg_volume])

        rel_volume = (float_volume / (pl.col("avg_volume") + epsilon)).alias("rel_volume")
        volume_spike = (
            pl.when(float_volume > _SPIKE_MULTIPLIER * pl.col("avg_volume"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
        ).alias("volume_spike")

        df = df.with_columns([rel_volume, volume_spike])

        logger.debug("[volume] Done – new columns: obv, rel_volume, avg_volume, volume_spike")
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(VolumeFeatures())
