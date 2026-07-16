"""ATR Volatility factor: Average True Range regime filter."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors._utils import rolling_percentile_rank
from apex_lab.research.factors.base import Factor

_ATR_PERIOD = 14
_ATR_PERCENTILE_WINDOW = 100
_MIN_PERCENTILE = 20.0
_MAX_PERCENTILE = 80.0


class AtrVolatilityFactor(Factor):
    """ATR volatility regime filter.

    Reuses the ``atr_pct`` column (rolling ATR percentile rank 0–100) that the
    :class:`~apex_lab.research.factors.ema_trend.EmaTrendFactor` adds to the
    DataFrame.  If ``atr_pct`` is not yet present this factor computes it
    directly.

    The entry signal is ``True`` when ATR percentile rank is within the
    [*min_pct*, *max_pct*] range, meaning volatility is neither dangerously
    low (which can indicate a false breakout) nor excessively high (which
    inflates risk-per-trade).
    """

    def __init__(
        self,
        period: int = _ATR_PERIOD,
        percentile_window: int = _ATR_PERCENTILE_WINDOW,
        min_pct: float = _MIN_PERCENTILE,
        max_pct: float = _MAX_PERCENTILE,
    ) -> None:
        self._period = period
        self._percentile_window = percentile_window
        self._min_pct = min_pct
        self._max_pct = max_pct

    @property
    def name(self) -> str:
        return "ATR"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append ATR and ``atr_pct`` columns if not already present."""
        if "atr_pct" in df.columns:
            return df

        prev_close = pl.col("close").shift(1)
        true_range = pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        ).cast(pl.Float64)

        enriched = df.with_columns([true_range.alias("_atr_tr")]).with_columns(
            [pl.col("_atr_tr").rolling_mean(window_size=self._period).alias("atr_14")]
        )

        atr_pct = rolling_percentile_rank(
            enriched.get_column("atr_14"), self._percentile_window
        )
        return enriched.with_columns([atr_pct.alias("atr_pct")]).drop("_atr_tr")

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` where ATR percentile is within the target range."""
        atr_pct = df["atr_pct"]
        return ((atr_pct >= self._min_pct) & (atr_pct <= self._max_pct)).fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "ATR Volatility",
            "period": self._period,
            "percentile_window": self._percentile_window,
            "min_pct": self._min_pct,
            "max_pct": self._max_pct,
            "signal": f"ATR percentile in [{self._min_pct}, {self._max_pct}]",
        }
