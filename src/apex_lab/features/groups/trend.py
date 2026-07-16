"""Trend feature group.

Computes moving-average and VWAP based trend indicators.

Required columns: ``timestamp``, ``high``, ``low``, ``close``, ``volume``

Computed features
-----------------
- ``ema_9``          — 9-period EMA of close
- ``ema_20``         — 20-period EMA of close
- ``ema_50``         — 50-period EMA of close
- ``ema_200``        — 200-period EMA of close
- ``ema_slope``      — Rate of change of EMA-9 (% per bar)
- ``vwap``           — Session VWAP resetting each trading day
- ``vwap_dist``      — Distance of close from VWAP (%)
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_EMA_EPSILON = 1e-10


class TrendFeatures(FeatureGroup):
    """Moving-average and VWAP trend indicators.

    Attributes:
        name: ``"trend"``
        warm_up_periods: ``200`` (longest EMA period)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "trend"

    @property
    def warm_up_periods(self) -> int:
        """Return EMA-200 warm-up window."""
        return 200

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute trend features and append them to *df*.

        Args:
            df: OHLCV DataFrame. Must contain ``timestamp``, ``high``, ``low``,
                ``close``, ``volume``.

        Returns:
            Input DataFrame with 7 additional feature columns.

        Raises:
            ValueError: If any required column is absent.
        """
        self._require_columns(df, ["timestamp", "high", "low", "close", "volume"])
        logger.debug("[trend] Computing trend features for %d rows", len(df))

        # ------------------------------------------------------------------ #
        # EMA variants (adjust=False matches TradingView / industry standard) #
        # ------------------------------------------------------------------ #
        ema_9 = pl.col("close").ewm_mean(span=9, adjust=False).alias("ema_9")
        ema_20 = pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20")
        ema_50 = pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50")
        ema_200 = pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200")

        df = df.with_columns([ema_9, ema_20, ema_50, ema_200])

        # ------------------------------------------------------------------ #
        # EMA-9 slope (% change per bar)                                       #
        # ------------------------------------------------------------------ #
        epsilon = pl.lit(_EMA_EPSILON)
        ema_slope = (
            (pl.col("ema_9") - pl.col("ema_9").shift(1))
            / (pl.col("ema_9").shift(1) + epsilon)
            * 100.0
        ).alias("ema_slope")

        # ------------------------------------------------------------------ #
        # VWAP (session reset by trading date)                                 #
        # ------------------------------------------------------------------ #
        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        session_col = "_trend_session"
        volume = pl.col("volume").cast(pl.Float64)
        cumulative_tpv = (typical_price * volume).cum_sum().over(session_col)
        cumulative_vol = volume.cum_sum().over(session_col)
        vwap = (
            pl.when(cumulative_vol > 0.0)
            .then(cumulative_tpv / cumulative_vol)
            .otherwise(typical_price)
            .alias("vwap")
        )

        df = (
            df.with_columns([ema_slope, pl.col("timestamp").dt.date().alias(session_col)])
            .with_columns([vwap])
            .drop(session_col)
        )

        # ------------------------------------------------------------------ #
        # VWAP distance                                                        #
        # ------------------------------------------------------------------ #
        vwap_dist = ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + epsilon) * 100.0).alias(
            "vwap_dist"
        )

        df = df.with_columns(vwap_dist)

        logger.debug(
            "[trend] Done – new columns: ema_9, ema_20, ema_50, ema_200, "
            "ema_slope, vwap, vwap_dist"
        )
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(TrendFeatures())
