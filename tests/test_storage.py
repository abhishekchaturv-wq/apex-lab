"""Tests for Parquet storage helpers."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from apex_lab.data.storage import (
    get_chunk_path,
    get_instruments_path,
    get_metadata_path,
    get_raw_path,
    merge_and_dedup,
    read_parquet,
    write_parquet,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_get_raw_path(tmp_path: pytest.FixtureRequest) -> None:
    """get_raw_path should return the expected nested path."""
    from pathlib import Path

    data_dir = Path(str(tmp_path))
    p = get_raw_path(data_dir, "30minute", "BANKNIFTY")
    assert p == data_dir / "raw" / "30minute" / "BANKNIFTY.parquet"


def test_get_metadata_path(tmp_path: pytest.FixtureRequest) -> None:
    """get_metadata_path should return the expected sidecar path."""
    from pathlib import Path

    data_dir = Path(str(tmp_path))
    p = get_metadata_path(data_dir, "day", "NIFTY")
    assert p == data_dir / "raw" / "day" / "NIFTY.metadata.json"


def test_get_instruments_path(tmp_path: pytest.FixtureRequest) -> None:
    """get_instruments_path should point to reference/instruments.parquet."""
    from pathlib import Path

    data_dir = Path(str(tmp_path))
    p = get_instruments_path(data_dir)
    assert p == data_dir / "reference" / "instruments.parquet"


def test_get_chunk_path(tmp_path: pytest.FixtureRequest) -> None:
    """get_chunk_path should return a deterministic cache path."""
    from pathlib import Path

    data_dir = Path(str(tmp_path))
    p = get_chunk_path(data_dir, "BANKNIFTY", "30minute", 3)
    assert p == data_dir / "cache" / "BANKNIFTY_30minute" / "chunk_0003.parquet"


# ---------------------------------------------------------------------------
# write_parquet / read_parquet round-trip
# ---------------------------------------------------------------------------


def _make_small_df() -> pl.DataFrame:
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    n = 5
    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)],
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [98.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "volume": [1000 + i * 10 for i in range(n)],
        }
    )


def test_write_and_read_parquet_round_trip(tmp_path) -> None:
    """Writing then reading a Parquet file should reproduce the DataFrame."""
    df = _make_small_df()
    path = tmp_path / "test.parquet"
    write_parquet(df, path)
    assert path.exists()

    loaded = read_parquet(path)
    assert loaded.shape == df.shape
    assert loaded.columns == df.columns


def test_write_parquet_creates_parent_directories(tmp_path) -> None:
    """write_parquet should create any missing parent directories."""
    df = _make_small_df()
    path = tmp_path / "nested" / "deep" / "dir" / "file.parquet"
    write_parquet(df, path)
    assert path.exists()


def test_read_parquet_raises_for_missing_file(tmp_path) -> None:
    """read_parquet should propagate an error for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_path / "nonexistent.parquet")


# ---------------------------------------------------------------------------
# merge_and_dedup
# ---------------------------------------------------------------------------


def test_merge_and_dedup_removes_duplicate_timestamps() -> None:
    """merge_and_dedup should deduplicate rows with identical timestamps."""
    df = _make_small_df()
    result = merge_and_dedup([df, df])
    assert len(result) == len(df)


def test_merge_and_dedup_sorts_ascending() -> None:
    """merge_and_dedup should produce a timestamp-sorted DataFrame."""
    df = _make_small_df()
    reversed_df = df.sort("timestamp", descending=True)
    result = merge_and_dedup([reversed_df])
    diffs = result["timestamp"].diff().drop_nulls()
    assert all(d.total_seconds() >= 0 for d in diffs)


def test_merge_and_dedup_concatenates_non_overlapping_chunks() -> None:
    """Non-overlapping chunks should all be present after merge."""
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    chunk_a = pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(3)],
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200],
        }
    )
    chunk_b = pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(3, 6)],
            "open": [103.0, 104.0, 105.0],
            "high": [105.0, 106.0, 107.0],
            "low": [101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1300, 1400, 1500],
        }
    )
    result = merge_and_dedup([chunk_a, chunk_b])
    assert len(result) == 6


def test_merge_and_dedup_raises_on_empty_list() -> None:
    """merge_and_dedup should raise ValueError when given an empty list."""
    with pytest.raises(ValueError, match="frames must not be empty"):
        merge_and_dedup([])
