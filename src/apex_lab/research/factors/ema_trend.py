"""EMA Trend factor: EMA20/EMA50 bullish crossover."""

from __future__ import annotations

from typing import Any

import polars as pl

from apex_lab.research.factors.base import Factor

_FAST_PERIOD = 20
_SLOW_PERIOD = 50
_TREND_PERIOD = 200
_ATR_PERIOD = 14
_ATR_PERCENTILE_WINDOW = 100


class EmaTrendFactor(Factor):
    """Detects bullish EMA crossovers (EMA20 crosses above EMA50).

    This factor also computes the slow EMA-200 (trend filter) and the ATR
    percentile rank required by the existing event-driven backtester.  When the
    engine evaluates an EMA-AND-X combination, calling ``compute`` on this
    factor first ensures all columns that the backtester expects are present.
    """

    @property
    def name(self) -> str:
        return "EMA"

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append EMA, ATR and crossover columns if not already present."""
        if "bullish_crossover" in df.columns:
            return df

        prev_close = pl.col("close").shift(1)
        true_range = pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        ).cast(pl.Float64)

        enriched = df.with_columns(
            [
                pl.col("close").ewm_mean(span=_FAST_PERIOD, adjust=False).alias("ema_20"),
                pl.col("close").ewm_mean(span=_SLOW_PERIOD, adjust=False).alias("ema_50"),
                pl.col("close").ewm_mean(span=_TREND_PERIOD, adjust=False).alias("ema_200"),
                true_range.alias("_ema_tr"),
            ]
        ).with_columns(
            [
                pl.col("_ema_tr").rolling_mean(window_size=_ATR_PERIOD).alias("atr_14"),
                (
                    (pl.col("ema_20") > pl.col("ema_50"))
                    & (pl.col("ema_20").shift(1) <= pl.col("ema_50").shift(1))
                )
                .fill_null(False)
                .alias("bullish_crossover"),
                (
                    (pl.col("ema_20") < pl.col("ema_50"))
                    & (pl.col("ema_20").shift(1) >= pl.col("ema_50").shift(1))
                )
                .fill_null(False)
                .alias("bearish_crossover"),
            ]
        )

        atr_series = enriched.get_column("atr_14")
        atr_pct = _rolling_percentile_rank(atr_series, _ATR_PERCENTILE_WINDOW)
        epsilon = 1e-9
        return enriched.with_columns(
            [
                atr_pct.alias("atr_pct"),
                (pl.col("atr_14") / (pl.col("close") + epsilon) * 100.0).alias("atr_norm"),
            ]
        ).drop("_ema_tr")

    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return ``True`` at bars where EMA20 crosses above EMA50."""
        return df["bullish_crossover"].fill_null(False)

    def metadata(self) -> dict[str, Any]:
        return {
            "factor": "EMA Trend",
            "fast_period": _FAST_PERIOD,
            "slow_period": _SLOW_PERIOD,
            "trend_period": _TREND_PERIOD,
            "signal": "EMA20 crosses above EMA50",
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Compute rolling percentile rank (0–100) without pandas."""
    import bisect
    from collections import deque

    values = series.to_list()
    out: list[float | None] = [None] * len(values)
    active_window: deque[float] = deque()
    sorted_window: list[float] = []

    for index, current in enumerate(values):
        if current is not None:
            bisect.insort(sorted_window, current)
            active_window.append(current)

        if len(active_window) > window:
            expired = active_window.popleft()
            expired_index = bisect.bisect_left(sorted_window, expired)
            del sorted_window[expired_index]

        if current is None or not sorted_window:
            continue

        rank_position = bisect.bisect_right(sorted_window, current)
        out[index] = rank_position / len(sorted_window) * 100.0

    return pl.Series(out, dtype=pl.Float64)
