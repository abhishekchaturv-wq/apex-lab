"""Volatility feature group.

Computes statistical and ATR-based volatility indicators.

Required columns: ``high``, ``low``, ``close``

Computed features
-----------------
- ``rolling_std``      — 20-bar rolling std of log returns (annualised %)
- ``bb_width``         — Bollinger Band width (upper−lower) / middle (%)
- ``atr_expansion``    — ATR relative to its 14-bar lag (ratio)
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_STD_WINDOW = 20
_BB_WINDOW = 20
_BB_STD_FACTOR = 2.0
_ATR_PERIOD = 14
_EPSILON = 1e-10


class VolatilityFeatures(FeatureGroup):
    """Rolling standard deviation, Bollinger Band width, and ATR expansion.

    Attributes:
        name: ``"volatility"``
        warm_up_periods: ``20`` (Bollinger Band / std window)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "volatility"

    @property
    def warm_up_periods(self) -> int:
        """Return Bollinger Band window."""
        return _BB_WINDOW

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute volatility features and append them to *df*.

        Args:
            df: DataFrame that must contain ``high``, ``low``, ``close``.

        Returns:
            Input DataFrame with 3 additional feature columns.

        Raises:
            ValueError: If any required column is absent.
        """
        self._require_columns(df, ["high", "low", "close"])
        logger.debug("[volatility] Computing volatility features for %d rows", len(df))

        epsilon = pl.lit(_EPSILON)

        # ------------------------------------------------------------------ #
        # Rolling standard deviation of log returns                            #
        # ------------------------------------------------------------------ #
        log_return = (pl.col("close") / (pl.col("close").shift(1) + epsilon)).log(base=2.718281828)
        rolling_std = (log_return.rolling_std(window_size=_STD_WINDOW) * 100.0).alias("rolling_std")

        # ------------------------------------------------------------------ #
        # Bollinger Band width                                                 #
        # ------------------------------------------------------------------ #
        bb_mid = pl.col("close").rolling_mean(window_size=_BB_WINDOW)
        bb_std = pl.col("close").rolling_std(window_size=_BB_WINDOW)
        bb_upper = bb_mid + _BB_STD_FACTOR * bb_std
        bb_lower = bb_mid - _BB_STD_FACTOR * bb_std
        bb_width = ((bb_upper - bb_lower) / (bb_mid + epsilon) * 100.0).alias("bb_width")

        # ------------------------------------------------------------------ #
        # ATR (internal; dropped after ratio calculation)                      #
        # ------------------------------------------------------------------ #
        tr_hl = pl.col("high") - pl.col("low")
        tr_hc = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr_lc = (pl.col("low") - pl.col("close").shift(1)).abs()
        true_range = pl.max_horizontal(tr_hl, tr_hc, tr_lc)
        atr = true_range.rolling_mean(window_size=_ATR_PERIOD).alias("_vol_atr")

        df = df.with_columns([rolling_std, bb_width, atr])

        # ATR expansion: current ATR vs lagged ATR
        atr_expansion = (
            pl.col("_vol_atr") / (pl.col("_vol_atr").shift(_ATR_PERIOD) + epsilon)
        ).alias("atr_expansion")

        df = df.with_columns(atr_expansion).drop("_vol_atr")

        logger.debug("[volatility] Done – new columns: rolling_std, bb_width, atr_expansion")
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(VolatilityFeatures())
