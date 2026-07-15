"""Tests for OHLCV data validator."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from apex_lab.data.validator import assert_valid_ohlcv, validate_ohlcv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 5, *, with_oi: bool = False) -> pl.DataFrame:
    """Return a clean synthetic OHLCV DataFrame."""
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    timestamps = [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)]
    closes = [100.0 + i for i in range(n)]
    data = {
        "timestamp": timestamps,
        "open": [c - 0.5 for c in closes],
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1000 + i * 10 for i in range(n)],
    }
    if with_oi:
        data["oi"] = [500 + i for i in range(n)]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_validate_clean_dataframe_returns_no_errors() -> None:
    """A well-formed OHLCV DataFrame should produce no validation errors."""
    df = _make_ohlcv(10)
    assert validate_ohlcv(df) == []


def test_validate_with_oi_column_returns_no_errors() -> None:
    """Optional OI column should not cause validation failures."""
    df = _make_ohlcv(5, with_oi=True)
    assert validate_ohlcv(df) == []


def test_validate_empty_dataframe_returns_no_errors() -> None:
    """Empty DataFrames should pass validation (nothing to validate)."""
    df = pl.DataFrame(
        {
            "timestamp": pl.Series([], dtype=pl.Datetime),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Int64),
        }
    )
    assert validate_ohlcv(df) == []


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------


def test_validate_missing_required_column() -> None:
    """A DataFrame missing a required column should return an error."""
    df = _make_ohlcv(3).drop("volume")
    errors = validate_ohlcv(df)
    assert any("missing required columns" in e for e in errors)


# ---------------------------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------------------------


def test_validate_duplicate_timestamps() -> None:
    """Duplicate timestamps should be reported as an error."""
    df = _make_ohlcv(5)
    # Duplicate the first row
    df_dup = pl.concat([df, df.head(1)])
    errors = validate_ohlcv(df_dup)
    assert any("duplicate" in e for e in errors)


# ---------------------------------------------------------------------------
# Unsorted timestamps
# ---------------------------------------------------------------------------


def test_validate_unsorted_timestamps() -> None:
    """Timestamps that are not sorted ascending should produce an error."""
    df = _make_ohlcv(5).sort("timestamp", descending=True)
    errors = validate_ohlcv(df)
    assert any("sorted" in e for e in errors)


# ---------------------------------------------------------------------------
# OHLC sanity
# ---------------------------------------------------------------------------


def test_validate_high_less_than_open() -> None:
    """high < open should produce a validation error."""
    df = _make_ohlcv(3)
    # Force high < open on the first row
    df = df.with_columns(
        pl.when(pl.col("timestamp") == df["timestamp"][0])
        .then(pl.col("open") - 2.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    errors = validate_ohlcv(df)
    assert any("high < open" in e for e in errors)


def test_validate_high_less_than_close() -> None:
    """high < close should produce a validation error."""
    df = _make_ohlcv(3)
    df = df.with_columns(
        pl.when(pl.col("timestamp") == df["timestamp"][0])
        .then(pl.col("close") - 2.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    errors = validate_ohlcv(df)
    assert any("high < close" in e for e in errors)


def test_validate_low_greater_than_open() -> None:
    """low > open should produce a validation error."""
    df = _make_ohlcv(3)
    df = df.with_columns(
        pl.when(pl.col("timestamp") == df["timestamp"][0])
        .then(pl.col("open") + 2.0)
        .otherwise(pl.col("low"))
        .alias("low")
    )
    errors = validate_ohlcv(df)
    assert any("low > open" in e for e in errors)


def test_validate_low_greater_than_close() -> None:
    """low > close should produce a validation error."""
    df = _make_ohlcv(3)
    df = df.with_columns(
        pl.when(pl.col("timestamp") == df["timestamp"][0])
        .then(pl.col("close") + 2.0)
        .otherwise(pl.col("low"))
        .alias("low")
    )
    errors = validate_ohlcv(df)
    assert any("low > close" in e for e in errors)


# ---------------------------------------------------------------------------
# Negative volume
# ---------------------------------------------------------------------------


def test_validate_negative_volume() -> None:
    """Negative volume should produce a validation error."""
    df = _make_ohlcv(3)
    df = df.with_columns(
        pl.when(pl.col("timestamp") == df["timestamp"][0])
        .then(pl.lit(-1))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    errors = validate_ohlcv(df)
    assert any("negative volume" in e for e in errors)


# ---------------------------------------------------------------------------
# assert_valid_ohlcv
# ---------------------------------------------------------------------------


def test_assert_valid_raises_on_invalid() -> None:
    """assert_valid_ohlcv should raise ValueError for an invalid DataFrame."""
    df = _make_ohlcv(5)
    df_dup = pl.concat([df, df.head(1)])
    with pytest.raises(ValueError, match="validation failed"):
        assert_valid_ohlcv(df_dup)


def test_assert_valid_passes_on_clean() -> None:
    """assert_valid_ohlcv should not raise for a clean DataFrame."""
    df = _make_ohlcv(5)
    assert_valid_ohlcv(df)  # must not raise
