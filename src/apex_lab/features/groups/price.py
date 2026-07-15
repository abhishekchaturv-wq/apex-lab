"""Price-derived feature group.

Computes candlestick anatomy and price-level indicators that are derivable
solely from OHLC data.  All computations are fully vectorised with Polars.

Required columns: ``open``, ``high``, ``low``, ``close``

Computed features
-----------------
- ``atr_14``          — Average True Range (period 14)
- ``atr_pct``         — Rolling percentile rank of ATR (0–100)
- ``atr_norm``        — ATR normalised by close price (%)
- ``body_pct``        — Candle body as % of total range
- ``upper_wick_pct``  — Upper wick as % of total range
- ``lower_wick_pct``  — Lower wick as % of total range
- ``range``           — High – Low (absolute range)
- ``gap_pct``         — Gap from previous close to current open (%)
- ``typical_price``   — (High + Low + Close) / 3
- ``median_price``    — (High + Low) / 2
- ``weighted_price``  — (High + Low + 2 × Close) / 4
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

_ATR_PERIOD = 14
_ATR_PCT_WINDOW = 100  # rolling window for ATR percentile rank
_EPSILON = 1e-10  # guard against zero-division on doji candles


class PriceFeatures(FeatureGroup):
    """Candlestick anatomy and price-level indicators.

    Attributes:
        name: ``"price"``
        warm_up_periods: ``14`` (ATR rolling window)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "price"

    @property
    def warm_up_periods(self) -> int:
        """Return ATR warm-up window."""
        return _ATR_PERIOD

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute price-derived features and append them to *df*.

        Args:
            df: OHLCV DataFrame.  Must contain ``open``, ``high``, ``low``,
                ``close``.

        Returns:
            Input DataFrame with 11 additional feature columns.

        Raises:
            ValueError: If any required column is absent.
        """
        self._require_columns(df, ["open", "high", "low", "close"])
        logger.debug("[price] Computing %d price features for %d rows", 11, len(df))

        # ------------------------------------------------------------------ #
        # True Range components                                                #
        # ------------------------------------------------------------------ #
        tr_hl = pl.col("high") - pl.col("low")
        tr_hc = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr_lc = (pl.col("low") - pl.col("close").shift(1)).abs()
        true_range = pl.max_horizontal(tr_hl, tr_hc, tr_lc).alias("_tr")

        # ------------------------------------------------------------------ #
        # Candle anatomy helpers                                               #
        # ------------------------------------------------------------------ #
        range_expr = (pl.col("high") - pl.col("low")).alias("range")
        # Use a stable float constant so the expression compiles correctly.
        epsilon = pl.lit(_EPSILON)
        safe_range = pl.col("high") - pl.col("low") + epsilon

        body_hi = pl.max_horizontal("open", "close")
        body_lo = pl.min_horizontal("open", "close")

        body_pct = ((body_hi - body_lo) / safe_range * 100.0).alias("body_pct")
        upper_wick_pct = ((pl.col("high") - body_hi) / safe_range * 100.0).alias("upper_wick_pct")
        lower_wick_pct = ((body_lo - pl.col("low")) / safe_range * 100.0).alias("lower_wick_pct")

        # ------------------------------------------------------------------ #
        # Price levels                                                         #
        # ------------------------------------------------------------------ #
        typical_price = ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias(
            "typical_price"
        )
        median_price = ((pl.col("high") + pl.col("low")) / 2.0).alias("median_price")
        weighted_price = ((pl.col("high") + pl.col("low") + pl.col("close") * 2.0) / 4.0).alias(
            "weighted_price"
        )

        # ------------------------------------------------------------------ #
        # Gap %                                                                #
        # ------------------------------------------------------------------ #
        prev_close = pl.col("close").shift(1)
        gap_pct = ((pl.col("open") - prev_close) / (prev_close + epsilon) * 100.0).alias("gap_pct")

        # First pass: materialise true-range column so we can roll over it.
        df = df.with_columns(true_range)

        # ------------------------------------------------------------------ #
        # ATR (SMA of true range)                                              #
        # ------------------------------------------------------------------ #
        atr_expr = pl.col("_tr").rolling_mean(window_size=_ATR_PERIOD).alias("atr_14")

        df = df.with_columns(atr_expr)

        # ------------------------------------------------------------------ #
        # ATR percentile rank (rolling)                                        #
        # ------------------------------------------------------------------ #
        atr_pct = (
            pl.col("atr_14")
            .map_batches(
                lambda s: _rolling_percentile_rank(s, _ATR_PCT_WINDOW),
                return_dtype=pl.Float64,
            )
            .alias("atr_pct")
        )

        # ATR normalised
        safe_close = pl.col("close") + epsilon
        atr_norm = (pl.col("atr_14") / safe_close * 100.0).alias("atr_norm")

        df = df.with_columns(
            [
                atr_pct,
                atr_norm,
                body_pct,
                upper_wick_pct,
                lower_wick_pct,
                range_expr,
                gap_pct,
                typical_price,
                median_price,
                weighted_price,
            ]
        )

        # Drop the intermediate true-range column
        df = df.drop("_tr")

        logger.debug("[price] Done – columns added: %s", df.columns[-11:])
        return df


# ---------------------------------------------------------------------------
# Helper: rolling percentile rank
# ---------------------------------------------------------------------------


def _rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Compute the rolling percentile rank (0–100) of *series*.

    For each position *i*, this returns the rank of ``series[i]`` among the
    ``min(window, i+1)`` most recent values (inclusive), expressed as a
    percentile.  An expanding window is used for the first ``window - 1``
    positions so that values are produced from the very first non-null row.

    Args:
        series: Input numeric series.
        window: Maximum rolling window size.

    Returns:
        Float series of the same length with values in ``[0, 100]``.
        Returns ``null`` only where the input is ``null``.
    """
    values = series.to_list()
    n = len(values)
    out: list[float | None] = [None] * n

    for i in range(n):
        current = values[i]
        if current is None:
            continue
        start = max(0, i - window + 1)
        window_vals = [v for v in values[start : i + 1] if v is not None]
        if not window_vals:
            continue
        rank = sum(1 for v in window_vals if v <= current)
        out[i] = rank / len(window_vals) * 100.0

    return pl.Series(values=out, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(PriceFeatures())
