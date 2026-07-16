"""VWAP factor: Volume-Weighted Average Price price-level confirmation."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor


class VwapFactor(Factor):
    """Session VWAP price-level confirmation filter.

    Computes an intraday VWAP that resets each trading session as::

        vwap = cumulative_sum(typical_price * volume) / cumulative_sum(volume)

    where ``typical_price = (high + low + close) / 3``.

    The cumulative sums are partitioned by trading date so multi-year history
    cannot pin VWAP to stale price regimes. Zero-volume bars fall back to
    ``typical_price`` until session volume becomes positive. The entry signal is
    ``True`` when the close price is above the session VWAP.
    """

    @property
    def name(self) -> str:
        return "VWAP"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append the session-reset ``vwap`` column if not already present."""
        col = "vwap"
        if col in df.columns:
            return df

        typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        volume = pl.col("volume").cast(pl.Float64).fill_null(0.0)
        session_col = "_vwap_session"
        cumulative_tpv = (typical_price * volume).cum_sum().over(session_col)
        cumulative_vol = volume.cum_sum().over(session_col)

        return (
            df.with_columns(pl.col("timestamp").dt.date().alias(session_col))
            .with_columns(
                [
                    pl.when(cumulative_vol > 0.0)
                    .then(cumulative_tpv / cumulative_vol)
                    .otherwise(typical_price)
                    .alias(col)
                ]
            )
            .drop(session_col)
        )

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where close is above cumulative VWAP."""
        return (df["close"] > df["vwap"]).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "VWAP",
            "mode": "session",
            "price": "typical_price",
            "signal": "close > session VWAP",
        }
