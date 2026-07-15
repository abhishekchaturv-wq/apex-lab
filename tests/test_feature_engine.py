"""Tests for FeatureEngine."""

from __future__ import annotations

import time

import polars as pl
import pytest

from apex_lab.features.base import FeatureGroup
from apex_lab.features.engine import FeatureEngine
from apex_lab.features.registry import FeatureRegistry

# ---------------------------------------------------------------------------
# Stub groups
# ---------------------------------------------------------------------------


class _ColAdder(FeatureGroup):
    """Appends a constant-value column to the DataFrame."""

    def __init__(self, name: str, col_name: str, warm_up: int = 0) -> None:
        self._name = name
        self._col_name = col_name
        self._warm_up = warm_up

    @property
    def name(self) -> str:
        return self._name

    @property
    def warm_up_periods(self) -> int:
        return self._warm_up

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(1.0).alias(self._col_name))


def _make_engine(*groups: FeatureGroup) -> FeatureEngine:
    registry = FeatureRegistry()
    for g in groups:
        registry.register(g)
    return FeatureEngine(registry)


# ---------------------------------------------------------------------------
# Basic engine tests
# ---------------------------------------------------------------------------


def test_engine_compute_returns_dataframe(small_ohlcv: pl.DataFrame):
    """compute() returns a Polars DataFrame."""
    engine = _make_engine(_ColAdder("g1", "col_a"))
    result = engine.compute(small_ohlcv)
    assert isinstance(result, pl.DataFrame)


def test_engine_compute_preserves_row_count(small_ohlcv: pl.DataFrame):
    """compute() must not drop or duplicate rows."""
    engine = _make_engine(_ColAdder("g1", "col_a"))
    result = engine.compute(small_ohlcv)
    assert len(result) == len(small_ohlcv)


def test_engine_compute_appends_columns(small_ohlcv: pl.DataFrame):
    """compute() appends expected new columns."""
    engine = _make_engine(_ColAdder("g1", "col_a"), _ColAdder("g2", "col_b"))
    result = engine.compute(small_ohlcv)
    assert "col_a" in result.columns
    assert "col_b" in result.columns


def test_engine_compute_preserves_original_columns(small_ohlcv: pl.DataFrame):
    """Original OHLCV columns must still be present after compute()."""
    engine = _make_engine(_ColAdder("g1", "extra"))
    result = engine.compute(small_ohlcv)
    for col in small_ohlcv.columns:
        assert col in result.columns


def test_engine_compute_no_groups_returns_input(small_ohlcv: pl.DataFrame):
    """compute() with an empty registry returns the input unchanged."""
    engine = FeatureEngine(FeatureRegistry())
    result = engine.compute(small_ohlcv)
    assert result.columns == small_ohlcv.columns


def test_engine_compute_subset_of_groups(small_ohlcv: pl.DataFrame):
    """Requesting specific groups only applies those groups."""
    engine = _make_engine(
        _ColAdder("g1", "col_a"),
        _ColAdder("g2", "col_b"),
    )
    result = engine.compute(small_ohlcv, groups=["g1"])
    assert "col_a" in result.columns
    assert "col_b" not in result.columns


def test_engine_compute_unknown_group_raises(small_ohlcv: pl.DataFrame):
    """Requesting an unregistered group raises KeyError."""
    engine = _make_engine(_ColAdder("real", "col_a"))
    with pytest.raises(KeyError):
        engine.compute(small_ohlcv, groups=["nonexistent"])


def test_engine_compute_groups_applied_in_order(small_ohlcv: pl.DataFrame):
    """Groups are applied in registration (and request) order."""
    call_order: list[str] = []

    class _OrderTracker(FeatureGroup):
        def __init__(self, n: str) -> None:
            self._n = n

        @property
        def name(self) -> str:
            return self._n

        def compute(self, df: pl.DataFrame) -> pl.DataFrame:
            call_order.append(self._n)
            return df

    engine = _make_engine(_OrderTracker("first"), _OrderTracker("second"))
    engine.compute(small_ohlcv)
    assert call_order == ["first", "second"]


# ---------------------------------------------------------------------------
# Warm-up helper
# ---------------------------------------------------------------------------


def test_engine_warm_up_periods_returns_max(small_ohlcv: pl.DataFrame):
    """warm_up_periods() returns the maximum across requested groups."""
    engine = _make_engine(
        _ColAdder("g1", "a", warm_up=5),
        _ColAdder("g2", "b", warm_up=20),
        _ColAdder("g3", "c", warm_up=10),
    )
    assert engine.warm_up_periods() == 20
    assert engine.warm_up_periods(groups=["g1", "g3"]) == 10


def test_engine_warm_up_periods_empty_registry():
    """warm_up_periods() returns 0 for an empty registry."""
    engine = FeatureEngine(FeatureRegistry())
    assert engine.warm_up_periods() == 0


# ---------------------------------------------------------------------------
# Merging / column collision
# ---------------------------------------------------------------------------


def test_engine_compute_no_duplicate_columns(small_ohlcv: pl.DataFrame):
    """No column should appear more than once in the result."""
    from apex_lab.features import FeatureEngine as DefaultEngine  # noqa: PLC0415

    engine = DefaultEngine()
    result = engine.compute(small_ohlcv)
    assert len(result.columns) == len(set(result.columns))


# ---------------------------------------------------------------------------
# NaN / null behaviour
# ---------------------------------------------------------------------------


def test_engine_no_unexpected_nans_outside_warmup(small_ohlcv: pl.DataFrame):
    """After the warm-up window, no feature column should be fully null."""
    from apex_lab.features import FeatureEngine as DefaultEngine  # noqa: PLC0415

    engine = DefaultEngine()
    result = engine.compute(small_ohlcv)
    warmup = engine.warm_up_periods()
    trimmed = result.slice(warmup)

    original_cols = set(small_ohlcv.columns)
    feature_cols = [c for c in result.columns if c not in original_cols]

    for col in feature_cols:
        series = trimmed[col]
        all_null = series.is_null().all()
        assert not all_null, f"Column '{col}' is entirely null after warm-up"


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


def test_engine_performance_100k_rows(large_ohlcv: pl.DataFrame):
    """All built-in groups should process 100k rows in under 10 seconds."""
    from apex_lab.features import FeatureEngine as DefaultEngine  # noqa: PLC0415

    engine = DefaultEngine()
    t0 = time.perf_counter()
    engine.compute(large_ohlcv)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"Performance target missed: {elapsed:.2f}s for 100k rows"
