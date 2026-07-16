"""MACD factor: Moving Average Convergence Divergence trend confirmation."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor

_FAST_PERIOD = 12
_SLOW_PERIOD = 26
_SIGNAL_PERIOD = 9


class MacdFactor(Factor):
    """MACD trend-confirmation filter.

    Computes the classic MACD indicator (EMA12 − EMA26) with a 9-period EMA
    signal line.  The entry signal is ``True`` when the MACD line is above its
    signal line, confirming positive momentum.
    """

    def __init__(
        self,
        fast: int = _FAST_PERIOD,
        slow: int = _SLOW_PERIOD,
        signal: int = _SIGNAL_PERIOD,
    ) -> None:
        self._fast = fast
        self._slow = slow
        self._signal = signal

    @property
    def name(self) -> str:
        return "MACD"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append ``macd_line``, ``macd_signal``, and ``macd_hist`` columns."""
        if "macd_line" in df.columns:
            return df

        return (
            df.with_columns(
                [
                    pl.col("close").ewm_mean(span=self._fast, adjust=False).alias("_macd_fast"),
                    pl.col("close").ewm_mean(span=self._slow, adjust=False).alias("_macd_slow"),
                ]
            )
            .with_columns([(pl.col("_macd_fast") - pl.col("_macd_slow")).alias("macd_line")])
            .with_columns(
                [pl.col("macd_line").ewm_mean(span=self._signal, adjust=False).alias("macd_signal")]
            )
            .with_columns([(pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")])
            .drop(["_macd_fast", "_macd_slow"])
        )

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where MACD line is above the signal line."""
        return (df["macd_line"] > df["macd_signal"]).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "MACD",
            "fast_period": self._fast,
            "slow_period": self._slow,
            "signal_period": self._signal,
            "signal": "MACD line > signal line",
        }
