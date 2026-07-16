"""RSI factor: Relative Strength Index momentum confirmation."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor

_RSI_PERIOD = 14
_SIGNAL_THRESHOLD = 50.0


class RsiFactor(Factor):
    """RSI momentum confirmation filter.

    Computes the 14-period Wilder RSI using an EWM with ``alpha=1/period``.
    The entry signal is ``True`` when RSI is above the *signal_threshold*
    (default 50), indicating positive momentum at the time the combined entry
    condition is evaluated.
    """

    def __init__(
        self,
        period: int = _RSI_PERIOD,
        signal_threshold: float = _SIGNAL_THRESHOLD,
    ) -> None:
        self._period = period
        self._signal_threshold = signal_threshold

    @property
    def name(self) -> str:
        return "RSI"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append ``rsi_{period}`` column if not already present."""
        col = f"rsi_{self._period}"
        if col in df.columns:
            return df

        alpha = 1.0 / self._period
        delta = pl.col("close").diff(1)
        gain = delta.clip(lower_bound=0.0)
        loss = (-delta).clip(lower_bound=0.0)

        return df.with_columns(
            [
                gain.ewm_mean(alpha=alpha, adjust=False).alias("_rsi_avg_gain"),
                loss.ewm_mean(alpha=alpha, adjust=False).alias("_rsi_avg_loss"),
            ]
        ).with_columns(
            [
                (
                    100.0
                    - 100.0
                    / (1.0 + pl.col("_rsi_avg_gain") / (pl.col("_rsi_avg_loss") + 1e-9))
                ).alias(col)
            ]
        ).drop(["_rsi_avg_gain", "_rsi_avg_loss"])

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where RSI is above the signal threshold."""
        col = f"rsi_{self._period}"
        return (df[col] > self._signal_threshold).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "RSI",
            "period": self._period,
            "signal_threshold": self._signal_threshold,
            "signal": f"RSI > {self._signal_threshold}",
        }
