"""Shared utility functions for the factors package."""

from __future__ import annotations

import bisect
from collections import deque

import polars as pl


def rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Compute the rolling percentile rank (0–100) of *series*.

    Null input values remain null in the output.  The rolling window tracks
    the most recent non-null values up to *window* entries.

    Args:
        series: Input series; may contain nulls.
        window: Maximum number of non-null observations to keep in the window.

    Returns:
        ``pl.Float64`` Series of the same length, with nulls where the input
        was null or the window was empty.
    """
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
