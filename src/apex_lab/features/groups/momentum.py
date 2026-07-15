"""Momentum feature group.

Computes RSI, MACD, and Rate-of-Change oscillators.

Required columns: ``close``

Computed features
-----------------
- ``rsi``        — Relative Strength Index (14)
- ``rsi_slope``  — Bar-to-bar change in RSI
- ``macd``       — MACD line (EMA-12 minus EMA-26)
- ``macd_signal``— 9-period EMA of MACD line
- ``macd_hist``  — MACD histogram (MACD minus signal)
- ``roc``        — Rate of Change over 10 bars (%)
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_ROC_PERIOD = 10
_EPSILON = 1e-10


class MomentumFeatures(FeatureGroup):
    """RSI, MACD, and Rate-of-Change oscillators.

    Attributes:
        name: ``"momentum"``
        warm_up_periods: ``35`` (MACD slow + signal lag)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "momentum"

    @property
    def warm_up_periods(self) -> int:
        """Return MACD + signal warm-up window."""
        return _MACD_SLOW + _MACD_SIGNAL

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute momentum features and append them to *df*.

        Args:
            df: DataFrame that must contain a ``close`` column.

        Returns:
            Input DataFrame with 6 additional feature columns.

        Raises:
            ValueError: If ``close`` column is absent.
        """
        self._require_columns(df, ["close"])
        logger.debug("[momentum] Computing momentum features for %d rows", len(df))

        # ------------------------------------------------------------------ #
        # RSI via exponential smoothing (Wilder / EWMA approach)               #
        # ------------------------------------------------------------------ #
        delta = pl.col("close").diff()

        # Wilder smoothing == EWM with alpha = 1/period
        gain = (pl.when(delta > 0).then(delta).otherwise(pl.lit(0.0))).ewm_mean(
            alpha=1.0 / _RSI_PERIOD, adjust=False
        )

        loss = (pl.when(delta < 0).then(-delta).otherwise(pl.lit(0.0))).ewm_mean(
            alpha=1.0 / _RSI_PERIOD, adjust=False
        )

        rs = gain / (loss + _EPSILON)
        rsi = (pl.lit(100.0) - pl.lit(100.0) / (pl.lit(1.0) + rs)).alias("rsi")

        df = df.with_columns(rsi)

        rsi_slope = (pl.col("rsi") - pl.col("rsi").shift(1)).alias("rsi_slope")

        # ------------------------------------------------------------------ #
        # MACD                                                                 #
        # ------------------------------------------------------------------ #
        ema_fast = pl.col("close").ewm_mean(span=_MACD_FAST, adjust=False)
        ema_slow = pl.col("close").ewm_mean(span=_MACD_SLOW, adjust=False)
        macd = (ema_fast - ema_slow).alias("macd")

        df = df.with_columns([rsi_slope, macd])

        macd_signal = pl.col("macd").ewm_mean(span=_MACD_SIGNAL, adjust=False).alias("macd_signal")
        df = df.with_columns(macd_signal)

        macd_hist = (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")

        # ------------------------------------------------------------------ #
        # Rate of Change                                                       #
        # ------------------------------------------------------------------ #
        prev_close = pl.col("close").shift(_ROC_PERIOD)
        roc = ((pl.col("close") - prev_close) / (prev_close + _EPSILON) * 100.0).alias("roc")

        df = df.with_columns([macd_hist, roc])

        logger.debug(
            "[momentum] Done – new columns: rsi, rsi_slope, macd, " "macd_signal, macd_hist, roc"
        )
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(MomentumFeatures())
