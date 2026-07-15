"""Shared fixtures for feature engineering tests."""

from __future__ import annotations

import datetime

import polars as pl
import pytest


@pytest.fixture()
def small_ohlcv() -> pl.DataFrame:
    """Return a 300-row OHLCV DataFrame suitable for unit tests.

    300 rows ensures all built-in feature groups (including EMA-200 with a
    200-bar warm-up) produce at least some non-null values.  The DataFrame
    contains synthetic but realistic price action with a timestamp column
    (minute bars starting 2024-01-15 09:15 IST).
    """
    n = 300
    base_ts = datetime.datetime(2024, 1, 15, 9, 15, 0)
    timestamps = [base_ts + datetime.timedelta(minutes=i) for i in range(n)]

    # Deterministic price walk
    closes = [100.0 + i * 0.5 + (i % 5) * 0.1 for i in range(n)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 0.8 for c in closes]
    lows = [c - 0.6 for c in closes]
    volumes = [10_000 + i * 100 for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture()
def large_ohlcv() -> pl.DataFrame:
    """Return a 100,000-row OHLCV DataFrame for performance tests."""
    n = 100_000
    base_ts = datetime.datetime(2024, 1, 1, 9, 15, 0)
    timestamps = [base_ts + datetime.timedelta(minutes=i) for i in range(n)]

    closes = [100.0 + (i % 500) * 0.1 for i in range(n)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 0.8 for c in closes]
    lows = [c - 0.6 for c in closes]
    volumes = [50_000 + (i % 1000) * 10 for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
