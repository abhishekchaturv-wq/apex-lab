"""Time feature group.

Extracts calendar and trading-session attributes from a datetime column.

Required columns: ``timestamp`` (``pl.Datetime``)

Computed features
-----------------
- ``hour``             — Hour of day (0–23)
- ``minute``           — Minute of hour (0–59)
- ``weekday``          — Day of week (1=Monday … 7=Sunday, ISO-8601)
- ``month``            — Month of year (1–12)
- ``quarter``          — Calendar quarter (1–4)
- ``trading_session``  — Encoded session: 0=pre-market, 1=regular, 2=post-market
"""

from __future__ import annotations

import logging

import polars as pl

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import default_registry

logger = logging.getLogger(__name__)

# Indian equity market hours (IST, UTC+5:30)
# Pre-market:   09:00 – 09:14
# Regular:      09:15 – 15:29
# Post-market:  15:30 – 16:00
_PRE_MARKET_START = 9 * 60 + 0  # 09:00
_REGULAR_START = 9 * 60 + 15  # 09:15
_REGULAR_END = 15 * 60 + 30  # 15:30
_POST_MARKET_END = 16 * 60 + 0  # 16:00

_SESSION_PRE = 0
_SESSION_REGULAR = 1
_SESSION_POST = 2


class TimeFeatures(FeatureGroup):
    """Calendar and trading-session features derived from a ``timestamp`` column.

    Attributes:
        name: ``"time"``
        warm_up_periods: ``0`` (no rolling window required)
    """

    @property
    def name(self) -> str:
        """Return group identifier."""
        return "time"

    @property
    def warm_up_periods(self) -> int:
        """Return warm-up window (none required)."""
        return 0

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract time-based features from the ``timestamp`` column.

        Args:
            df: DataFrame with a ``timestamp`` column of dtype ``pl.Datetime``.

        Returns:
            Input DataFrame with 6 additional feature columns.

        Raises:
            ValueError: If the ``timestamp`` column is absent.
        """
        self._require_columns(df, ["timestamp"])
        logger.debug("[time] Computing time features for %d rows", len(df))

        hour = pl.col("timestamp").dt.hour().alias("hour")
        minute = pl.col("timestamp").dt.minute().alias("minute")
        weekday = pl.col("timestamp").dt.weekday().alias("weekday")
        month = pl.col("timestamp").dt.month().alias("month")
        quarter = pl.col("timestamp").dt.quarter().alias("quarter")

        # Minutes-since-midnight for session boundary logic
        minutes_since_midnight = pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col(
            "timestamp"
        ).dt.minute().cast(pl.Int32)

        trading_session = (
            pl.when(
                (minutes_since_midnight >= _PRE_MARKET_START)
                & (minutes_since_midnight < _REGULAR_START)
            )
            .then(pl.lit(_SESSION_PRE))
            .when(
                (minutes_since_midnight >= _REGULAR_START) & (minutes_since_midnight < _REGULAR_END)
            )
            .then(pl.lit(_SESSION_REGULAR))
            .when(
                (minutes_since_midnight >= _REGULAR_END)
                & (minutes_since_midnight < _POST_MARKET_END)
            )
            .then(pl.lit(_SESSION_POST))
            .otherwise(pl.lit(-1))
        ).alias("trading_session")

        df = df.with_columns([hour, minute, weekday, month, quarter, trading_session])

        logger.debug(
            "[time] Done – new columns: hour, minute, weekday, month, quarter, trading_session"
        )
        return df


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

default_registry.register(TimeFeatures())
