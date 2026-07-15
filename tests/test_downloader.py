"""Tests for the historical data downloader (100% offline, Kite mocked)."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apex_lab.data.downloader import (
    DataEngine,
    _candles_to_df,
    _parse_date,
    split_into_chunks,
)
from apex_lab.data.storage import get_chunk_path, get_raw_path, write_parquet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_instruments() -> list[dict]:
    return [
        {
            "instrument_token": 260105,
            "exchange_token": 1016,
            "tradingsymbol": "BANKNIFTY",
            "name": "NIFTY BANK",
            "last_price": 0.0,
            "expiry": "",
            "strike": 0.0,
            "tick_size": 0.05,
            "lot_size": 15,
            "instrument_type": "EQ",
            "segment": "INDICES",
            "exchange": "NSE",
        }
    ]


def _make_candles(n: int, base_ts: datetime.datetime | None = None) -> list[dict]:
    """Return *n* synthetic Kite historical_data dicts."""
    if base_ts is None:
        base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    candles = []
    for i in range(n):
        ts = base_ts + datetime.timedelta(minutes=30 * i)
        candles.append(
            {
                "date": ts,
                "open": 100.0 + i,
                "high": 102.0 + i,
                "low": 98.0 + i,
                "close": 101.0 + i,
                "volume": 1000 + i * 10,
            }
        )
    return candles


def _make_engine(tmp_path: Path, candles_per_chunk: list[list[dict]]) -> DataEngine:
    """Build a DataEngine with a mock Kite client returning preset candles."""
    mock_kite = MagicMock()
    mock_kite.instruments.return_value = _sample_instruments()
    mock_kite.historical_data.side_effect = candles_per_chunk

    engine = DataEngine(mock_kite, tmp_path, base_delay=0.0)
    engine.instruments.refresh()
    return engine


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


def test_parse_date_from_iso_string() -> None:
    """ISO string should parse to a date object."""
    d = _parse_date("2024-01-15")
    assert d == datetime.date(2024, 1, 15)


def test_parse_date_from_date_object() -> None:
    """A date object should be returned unchanged."""
    d = datetime.date(2024, 3, 10)
    assert _parse_date(d) == d


def test_parse_date_from_datetime_object() -> None:
    """A datetime object should have its .date() extracted."""
    dt = datetime.datetime(2024, 3, 10, 9, 0, 0)
    assert _parse_date(dt) == datetime.date(2024, 3, 10)


def test_parse_date_today() -> None:
    """The string 'today' (any case) should resolve to today's date."""
    assert _parse_date("today") == datetime.date.today()
    assert _parse_date("TODAY") == datetime.date.today()


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------


def test_split_into_chunks_single_chunk() -> None:
    """A short range that fits in one chunk should return a single tuple."""
    start = datetime.date(2024, 1, 1)
    end = datetime.date(2024, 1, 30)
    chunks = split_into_chunks(start, end, "day")
    assert len(chunks) == 1
    assert chunks[0] == (start, end)


def test_split_into_chunks_multiple_chunks() -> None:
    """A range longer than max_days should be split into multiple chunks."""
    start = datetime.date(2024, 1, 1)
    end = datetime.date(2024, 4, 30)  # ~120 days, max 100 → 2 chunks
    chunks = split_into_chunks(start, end, "30minute")
    assert len(chunks) == 2
    # Chunks are contiguous and non-overlapping
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # No gap between chunks
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][0] == chunks[i][1] + datetime.timedelta(days=1)


def test_split_into_chunks_same_day() -> None:
    """Start == end should yield exactly one chunk."""
    d = datetime.date(2024, 6, 15)
    chunks = split_into_chunks(d, d, "minute")
    assert len(chunks) == 1
    assert chunks[0] == (d, d)


def test_split_into_chunks_invalid_interval() -> None:
    """An unsupported interval should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported interval"):
        split_into_chunks(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31), "2minute")


def test_split_into_chunks_start_after_end() -> None:
    """start > end should raise ValueError."""
    with pytest.raises(ValueError, match="must not be after"):
        split_into_chunks(datetime.date(2024, 1, 31), datetime.date(2024, 1, 1), "day")


def test_split_into_chunks_covers_entire_range() -> None:
    """The union of all chunks must span the entire requested date range."""
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2024, 12, 31)
    chunks = split_into_chunks(start, end, "30minute")
    assert chunks[0][0] == start
    assert chunks[-1][1] == end


# ---------------------------------------------------------------------------
# _candles_to_df
# ---------------------------------------------------------------------------


def test_candles_to_df_empty_list_returns_empty_df() -> None:
    """Empty candle list should produce an empty DataFrame."""
    df = _candles_to_df([])
    assert len(df) == 0
    assert "timestamp" in df.columns


def test_candles_to_df_creates_correct_columns() -> None:
    """_candles_to_df should produce required OHLCV columns."""
    candles = _make_candles(3)
    df = _candles_to_df(candles)
    for col in ("timestamp", "open", "high", "low", "close", "volume"):
        assert col in df.columns


def test_candles_to_df_with_oi_creates_oi_column() -> None:
    """If raw data has 'oi' field, the DataFrame should include it."""
    candles = _make_candles(3)
    for c in candles:
        c["oi"] = 500
    df = _candles_to_df(candles)
    assert "oi" in df.columns


# ---------------------------------------------------------------------------
# KiteDownloader — chunk download and resume
# ---------------------------------------------------------------------------


def test_downloader_single_chunk_download(tmp_path: Path) -> None:
    """A single-chunk download should produce a valid Parquet file."""
    candles = _make_candles(10)
    engine = _make_engine(tmp_path, [candles])

    df = engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-10")

    assert len(df) == 10
    raw_path = get_raw_path(tmp_path, "day", "BANKNIFTY")
    assert raw_path.exists()


def test_downloader_multi_chunk_merge(tmp_path: Path) -> None:
    """Multi-chunk download should merge into a single deduplicated DataFrame."""
    base_ts = datetime.datetime(2024, 1, 2, 9, 15)
    chunk1 = _make_candles(5, base_ts)
    chunk2 = _make_candles(5, base_ts + datetime.timedelta(minutes=30 * 5))

    mock_kite = MagicMock()
    mock_kite.instruments.return_value = _sample_instruments()
    # Force two chunks by using "30minute" with a 101-day range
    mock_kite.historical_data.side_effect = [chunk1, chunk2]

    engine = DataEngine(mock_kite, tmp_path, base_delay=0.0)
    engine.instruments.refresh()

    start = datetime.date(2024, 1, 1)
    end = start + datetime.timedelta(days=105)  # >100 days → 2 chunks
    df = engine.download_symbol("BANKNIFTY", "30minute", start, end)

    assert len(df) == 10
    assert mock_kite.historical_data.call_count == 2


def test_downloader_deduplicates_overlapping_candles(tmp_path: Path) -> None:
    """Duplicate timestamps across chunks must be deduplicated."""
    candles = _make_candles(5)
    # Return the same candles twice (simulates overlap)
    engine = _make_engine(tmp_path, [candles, candles])

    start = datetime.date(2024, 1, 1)
    end = start + datetime.timedelta(days=105)
    df = engine.download_symbol("BANKNIFTY", "30minute", start, end)

    assert len(df) == len(candles)


def test_downloader_resume_skips_cached_chunks(tmp_path: Path) -> None:
    """A pre-existing chunk cache file must be loaded instead of re-fetched."""
    candles = _make_candles(5)
    # Pre-write chunk 0 to cache
    chunk_path = get_chunk_path(tmp_path, "BANKNIFTY", "30minute", 0)
    pre_written = _candles_to_df(candles)
    write_parquet(pre_written, chunk_path)

    mock_kite = MagicMock()
    mock_kite.instruments.return_value = _sample_instruments()
    # Only chunk 1 will be fetched
    candles2 = _make_candles(5, datetime.datetime(2024, 5, 1, 9, 15))
    mock_kite.historical_data.return_value = candles2

    engine = DataEngine(mock_kite, tmp_path, base_delay=0.0)
    engine.instruments.refresh()

    start = datetime.date(2024, 1, 1)
    end = start + datetime.timedelta(days=105)
    engine.download_symbol("BANKNIFTY", "30minute", start, end)

    # historical_data should only be called once (for chunk 1, not chunk 0)
    assert mock_kite.historical_data.call_count == 1


def test_downloader_cleans_up_chunk_cache_after_success(tmp_path: Path) -> None:
    """Chunk cache files should be deleted after a successful download."""
    candles = _make_candles(5)
    engine = _make_engine(tmp_path, [candles])

    engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-10")

    cache_dir = tmp_path / "cache" / "BANKNIFTY_day"
    remaining = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
    assert remaining == []


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_downloader_retries_on_network_error(tmp_path: Path) -> None:
    """historical_data failures should trigger retries with backoff."""
    from kiteconnect import exceptions as kite_exc

    candles = _make_candles(3)
    mock_kite = MagicMock()
    mock_kite.instruments.return_value = _sample_instruments()
    # First call raises, second succeeds
    mock_kite.historical_data.side_effect = [
        kite_exc.NetworkException("timeout", code=500),
        candles,
    ]

    engine = DataEngine(mock_kite, tmp_path, base_delay=0.0)
    engine.instruments.refresh()

    df = engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-05")

    assert len(df) == len(candles)
    assert mock_kite.historical_data.call_count == 2


def test_downloader_raises_after_max_retries(tmp_path: Path) -> None:
    """After exhausting retries, the exception should propagate."""
    from kiteconnect import exceptions as kite_exc

    mock_kite = MagicMock()
    mock_kite.instruments.return_value = _sample_instruments()
    mock_kite.historical_data.side_effect = kite_exc.NetworkException("timeout", code=500)

    engine = DataEngine(mock_kite, tmp_path, max_retries=2, base_delay=0.0)
    engine.instruments.refresh()

    with pytest.raises(kite_exc.NetworkException):
        engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-05")

    assert mock_kite.historical_data.call_count == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------


def test_update_symbol_downloads_only_missing_candles(tmp_path: Path) -> None:
    """update_symbol should only fetch candles after the latest stored one."""
    # Write existing data up to 2024-01-10
    base_ts = datetime.datetime(2024, 1, 2, 9, 15)
    existing_candles = _make_candles(10, base_ts)
    engine = _make_engine(tmp_path, [existing_candles])

    engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-10")

    # Now set up mock to return new candles for the next update
    new_base = datetime.datetime(2024, 1, 11, 9, 15)
    new_candles = _make_candles(5, new_base)
    engine._kite.historical_data.side_effect = [new_candles]

    updated = engine.update_symbol("BANKNIFTY", "day")

    # Should have 10 old + 5 new = 15 total
    assert len(updated) == 15


def test_update_symbol_noop_when_already_current(tmp_path: Path) -> None:
    """update_symbol should return existing data without API calls if current."""
    # Build existing data that ends today
    today = datetime.date.today()
    base_ts = datetime.datetime(today.year, today.month, today.day, 9, 15)
    existing_candles = _make_candles(3, base_ts)
    engine = _make_engine(tmp_path, [existing_candles])

    engine.download_symbol("BANKNIFTY", "day", today, today)
    initial_call_count = engine._kite.historical_data.call_count

    # update should detect no gap → no new API call
    updated = engine.update_symbol("BANKNIFTY", "day")

    assert engine._kite.historical_data.call_count == initial_call_count
    assert len(updated) == len(existing_candles)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_download_creates_metadata_file(tmp_path: Path) -> None:
    """A successful download should produce a sidecar .metadata.json file."""
    candles = _make_candles(5)
    engine = _make_engine(tmp_path, [candles])

    engine.download_symbol("BANKNIFTY", "day", "2024-01-01", "2024-01-05")

    meta_path = tmp_path / "raw" / "day" / "BANKNIFTY.metadata.json"
    assert meta_path.exists()

    import json

    meta = json.loads(meta_path.read_text())
    assert meta["symbol"] == "BANKNIFTY"
    assert meta["interval"] == "day"
    assert meta["num_candles"] == len(candles)
    assert meta["source"] == "zerodha_kite"


# ---------------------------------------------------------------------------
# download_universe
# ---------------------------------------------------------------------------


def test_download_universe_downloads_all_symbols(tmp_path: Path) -> None:
    """download_universe should process every symbol in the list."""
    extra_instruments = [
        {
            "instrument_token": 738561,
            "exchange_token": 2885,
            "tradingsymbol": "RELIANCE",
            "name": "RELIANCE",
            "last_price": 0.0,
            "expiry": "",
            "strike": 0.0,
            "tick_size": 0.05,
            "lot_size": 1,
            "instrument_type": "EQ",
            "segment": "NSE",
            "exchange": "NSE",
        }
    ] + _sample_instruments()

    mock_kite = MagicMock()
    mock_kite.instruments.return_value = extra_instruments
    candles1 = _make_candles(5)
    candles2 = _make_candles(5, datetime.datetime(2024, 2, 1, 9, 15))
    mock_kite.historical_data.side_effect = [candles1, candles2]

    engine = DataEngine(mock_kite, tmp_path, base_delay=0.0)
    engine.instruments.refresh()

    results = engine.download_universe(["BANKNIFTY", "RELIANCE"], "day", "2024-01-01", "2024-01-10")

    assert set(results.keys()) == {"BANKNIFTY", "RELIANCE"}
    assert all(len(df) == 5 for df in results.values())
