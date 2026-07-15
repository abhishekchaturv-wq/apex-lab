"""OHLCV data validation for downloaded market data.

Validation is non-destructive: functions return error lists rather than
raising immediately, allowing callers to decide how to handle failures.
"""

from __future__ import annotations

import polars as pl


def validate_ohlcv(df: pl.DataFrame, *, timestamp_column: str = "timestamp") -> list[str]:
    """Validate an OHLCV DataFrame and return a list of error messages.

    Checks performed:

    * All required columns are present.
    * No duplicate timestamps.
    * Timestamps are strictly sorted in ascending order.
    * OHLC sanity: ``high >= open``, ``high >= close``, ``low <= open``,
      ``low <= close``.
    * Non-negative volume.

    Args:
        df: OHLCV DataFrame to validate.
        timestamp_column: Name of the timestamp column.

    Returns:
        List of human-readable error strings. An empty list means no errors.
    """
    errors: list[str] = []

    # --- required columns ---------------------------------------------------
    required = (timestamp_column, "open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        # Cannot proceed with further checks
        return errors

    if len(df) == 0:
        return errors

    # --- duplicate timestamps -----------------------------------------------
    n_unique = df.select(pl.col(timestamp_column).n_unique()).item()
    n_total = len(df)
    if n_unique < n_total:
        errors.append(f"duplicate timestamps: {n_total - n_unique} duplicate(s) found")

    # --- sorted ascending ---------------------------------------------------
    is_sorted = df.select(
        (pl.col(timestamp_column).diff().drop_nulls() >= pl.duration(seconds=0)).all()
    ).item()
    if not is_sorted:
        errors.append("timestamps are not sorted in ascending order")

    # --- OHLC sanity --------------------------------------------------------
    high_lt_open = df.filter(pl.col("high") < pl.col("open"))
    if len(high_lt_open) > 0:
        errors.append(f"high < open: {len(high_lt_open)} row(s) violate high >= open")

    high_lt_close = df.filter(pl.col("high") < pl.col("close"))
    if len(high_lt_close) > 0:
        errors.append(f"high < close: {len(high_lt_close)} row(s) violate high >= close")

    low_gt_open = df.filter(pl.col("low") > pl.col("open"))
    if len(low_gt_open) > 0:
        errors.append(f"low > open: {len(low_gt_open)} row(s) violate low <= open")

    low_gt_close = df.filter(pl.col("low") > pl.col("close"))
    if len(low_gt_close) > 0:
        errors.append(f"low > close: {len(low_gt_close)} row(s) violate low <= close")

    # --- non-negative volume ------------------------------------------------
    neg_volume = df.filter(pl.col("volume") < 0)
    if len(neg_volume) > 0:
        errors.append(f"negative volume: {len(neg_volume)} row(s) have volume < 0")

    return errors


def assert_valid_ohlcv(df: pl.DataFrame, *, timestamp_column: str = "timestamp") -> None:
    """Assert that an OHLCV DataFrame passes all validation checks.

    Args:
        df: OHLCV DataFrame to validate.
        timestamp_column: Name of the timestamp column.

    Raises:
        ValueError: If any validation errors are found.
    """
    errors = validate_ohlcv(df, timestamp_column=timestamp_column)
    if errors:
        raise ValueError("OHLCV validation failed: " + " | ".join(errors))
