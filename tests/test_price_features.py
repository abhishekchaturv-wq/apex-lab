"""Tests for PriceFeatures group."""

from __future__ import annotations

import polars as pl
import pytest

from apex_lab.features.groups.price import PriceFeatures


@pytest.fixture()
def group() -> PriceFeatures:
    """Return a fresh PriceFeatures instance."""
    return PriceFeatures()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_price_group_name(group: PriceFeatures):
    """Group name is 'price'."""
    assert group.name == "price"


def test_price_warm_up_periods(group: PriceFeatures):
    """Warm-up is 14 (ATR period)."""
    assert group.warm_up_periods == 14


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------


_EXPECTED_COLUMNS = [
    "atr_14",
    "atr_pct",
    "atr_norm",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range",
    "gap_pct",
    "typical_price",
    "median_price",
    "weighted_price",
]


def test_price_output_columns_present(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """All 11 expected price columns are present in the result."""
    result = group.compute(small_ohlcv)
    for col in _EXPECTED_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_price_no_extra_internal_columns(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """The intermediate _tr column is removed from the output."""
    result = group.compute(small_ohlcv)
    assert "_tr" not in result.columns


def test_price_row_count_preserved(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """Row count is unchanged after compute."""
    result = group.compute(small_ohlcv)
    assert len(result) == len(small_ohlcv)


# ---------------------------------------------------------------------------
# Missing columns guard
# ---------------------------------------------------------------------------


def test_price_missing_column_raises(group: PriceFeatures):
    """ValueError raised when a required column is absent."""
    bad_df = pl.DataFrame({"open": [1.0], "high": [2.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        group.compute(bad_df)


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_price_atr_non_negative(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """ATR must be non-negative for all non-null values."""
    result = group.compute(small_ohlcv)
    atr = result["atr_14"].drop_nulls()
    assert (atr >= 0).all()


def test_price_atr_pct_range(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """ATR percentile rank must be between 0 and 100."""
    result = group.compute(small_ohlcv)
    pct = result["atr_pct"].drop_nulls()
    assert (pct >= 0).all() and (pct <= 100).all()


def test_price_body_pct_range(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """body_pct should be in [0, 100]."""
    result = group.compute(small_ohlcv)
    bp = result["body_pct"].drop_nulls()
    assert (bp >= 0).all() and (bp <= 100).all()


def test_price_wick_pcts_range(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """Upper and lower wick percentages must be in [0, 100]."""
    result = group.compute(small_ohlcv)
    for col in ("upper_wick_pct", "lower_wick_pct"):
        values = result[col].drop_nulls()
        assert (values >= 0).all() and (values <= 100).all(), f"{col} out of range"


def test_price_range_equals_high_minus_low(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """range == high - low for all rows."""
    result = group.compute(small_ohlcv)
    expected = (result["high"] - result["low"]).round(8)
    actual = result["range"].round(8)
    assert (expected - actual).abs().max() < 1e-6


def test_price_typical_price_formula(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """typical_price == (high + low + close) / 3."""
    result = group.compute(small_ohlcv)
    expected = ((result["high"] + result["low"] + result["close"]) / 3.0).round(8)
    actual = result["typical_price"].round(8)
    assert (expected - actual).abs().max() < 1e-6


def test_price_median_price_formula(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """median_price == (high + low) / 2."""
    result = group.compute(small_ohlcv)
    expected = ((result["high"] + result["low"]) / 2.0).round(8)
    actual = result["median_price"].round(8)
    assert (expected - actual).abs().max() < 1e-6


def test_price_weighted_price_formula(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """weighted_price == (high + low + 2*close) / 4."""
    result = group.compute(small_ohlcv)
    expected = ((result["high"] + result["low"] + result["close"] * 2.0) / 4.0).round(8)
    actual = result["weighted_price"].round(8)
    assert (expected - actual).abs().max() < 1e-6


def test_price_gap_pct_first_row_null(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """The first row of gap_pct must be null (no previous close)."""
    result = group.compute(small_ohlcv)
    assert result["gap_pct"][0] is None


def test_price_atr_first_rows_null(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """The first 13 ATR values must be null (rolling window not yet full)."""
    result = group.compute(small_ohlcv)
    # rows 0..12 should be null; row 13 should not be null
    assert result["atr_14"][:13].is_null().all()
    assert result["atr_14"][13] is not None


# ---------------------------------------------------------------------------
# Warm-up NaN check
# ---------------------------------------------------------------------------


def test_price_post_warmup_no_all_null(group: PriceFeatures, small_ohlcv: pl.DataFrame):
    """After warm-up, no price feature column should be entirely null."""
    warmup = group.warm_up_periods
    result = group.compute(small_ohlcv).slice(warmup)
    for col in _EXPECTED_COLUMNS:
        assert not result[col].is_null().all(), f"'{col}' is entirely null after warm-up"
