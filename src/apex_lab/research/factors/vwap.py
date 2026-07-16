"""VWAP factor: Volume-Weighted Average Price price-level confirmation."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor

_VWAP_WINDOW = 50


class VwapFactor(Factor):
    """Rolling VWAP price-level confirmation filter.

    Computes a rolling VWAP over the last *window* bars as::

        vwap = rolling_sum(close * volume, window) / rolling_sum(volume, window)

    A rolling (anchored) VWAP is used rather than a session-reset VWAP so the
    factor works correctly on multi-session datasets.  The entry signal is
    ``True`` when the close price is above the rolling VWAP, indicating the
    market is trading at a premium to its recent volume-weighted average.
    """

    def __init__(self, window: int = _VWAP_WINDOW) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "VWAP"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append ``vwap_{window}`` column if not already present."""
        col = f"vwap_{self._window}"
        if col in df.columns:
            return df

        return df.with_columns(
            [
                (
                    (pl.col("close") * pl.col("volume"))
                    .rolling_sum(window_size=self._window)
                    / pl.col("volume").rolling_sum(window_size=self._window)
                ).alias(col)
            ]
        )

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where close is above the rolling VWAP."""
        col = f"vwap_{self._window}"
        return (df["close"] > df[col]).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "VWAP",
            "window": self._window,
            "signal": f"close > rolling VWAP({self._window})",
        }
