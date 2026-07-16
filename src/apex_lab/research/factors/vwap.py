"""VWAP factor: Volume-Weighted Average Price price-level confirmation."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor


class VwapFactor(Factor):
    """Cumulative VWAP price-level confirmation filter.

    Computes a cumulative VWAP from the start of the series as::

        vwap = cumulative_sum(typical_price * volume) / cumulative_sum(volume)

    where ``typical_price = (high + low + close) / 3``.

    The cumulative form avoids window-edge churn and aligns with the project
    trend-feature VWAP. It also guards against zero-volume bars by falling back
    to ``typical_price`` until cumulative volume becomes positive. The entry
    signal is
    ``True`` when the close price is above the cumulative VWAP, indicating the
    market is trading at a premium to its recent volume-weighted average.
    """

    @property
    def name(self) -> str:
        return "VWAP"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append the cumulative ``vwap`` column if not already present."""
        col = "vwap"
        if col in df.columns:
            return df

        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        volume = pl.col("volume").cast(pl.Float64).fill_null(0.0)
        cumulative_tpv = (typical_price * volume).cum_sum()
        cumulative_vol = volume.cum_sum()

        return df.with_columns(
            [
                pl.when(cumulative_vol > 0.0)
                .then(cumulative_tpv / cumulative_vol)
                .otherwise(typical_price)
                .alias(col)
            ]
        )

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where close is above cumulative VWAP."""
        return (df["close"] > df["vwap"]).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "VWAP",
            "mode": "cumulative",
            "price": "typical_price",
            "signal": "close > cumulative VWAP",
        }
