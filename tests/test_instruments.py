"""Tests for InstrumentManager (100% offline, Kite API mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from apex_lab.data.instruments import InstrumentManager, _instruments_to_df

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_kite(instruments: list[dict]) -> MagicMock:
    """Return a mock KiteConnect whose instruments() returns *instruments*."""
    mock = MagicMock()
    mock.instruments.return_value = instruments
    return mock


def _sample_instruments() -> list[dict]:
    """Return a small list of representative instrument dicts."""
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
        },
        {
            "instrument_token": 256265,
            "exchange_token": 1001,
            "tradingsymbol": "NIFTY 50",
            "name": "NIFTY 50",
            "last_price": 0.0,
            "expiry": "",
            "strike": 0.0,
            "tick_size": 0.05,
            "lot_size": 50,
            "instrument_type": "EQ",
            "segment": "INDICES",
            "exchange": "NSE",
        },
        {
            "instrument_token": 738561,
            "exchange_token": 2885,
            "tradingsymbol": "RELIANCE",
            "name": "RELIANCE INDUSTRIES",
            "last_price": 0.0,
            "expiry": "",
            "strike": 0.0,
            "tick_size": 0.05,
            "lot_size": 1,
            "instrument_type": "EQ",
            "segment": "NSE",
            "exchange": "NSE",
        },
    ]


# ---------------------------------------------------------------------------
# _instruments_to_df
# ---------------------------------------------------------------------------


def test_instruments_to_df_returns_polars_dataframe() -> None:
    """_instruments_to_df should return a non-empty Polars DataFrame."""
    raw = _sample_instruments()
    df = _instruments_to_df(raw)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == len(raw)


def test_instruments_to_df_normalises_empty_expiry() -> None:
    """Empty expiry values should be stored as empty strings."""
    raw = _sample_instruments()
    df = _instruments_to_df(raw)
    assert df.schema["expiry"] == pl.String


def test_instruments_to_df_empty_list_returns_empty_df() -> None:
    """An empty raw list should produce an empty DataFrame."""
    df = _instruments_to_df([])
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 0


def test_instruments_to_df_handles_date_expiry() -> None:
    """Date objects in the expiry field should be converted to strings."""
    import datetime

    raw = [
        {
            "instrument_token": 1,
            "exchange_token": 1,
            "tradingsymbol": "NIFTY24JAN18000CE",
            "name": "NIFTY",
            "last_price": 0.0,
            "expiry": datetime.date(2024, 1, 25),
            "strike": 18000.0,
            "tick_size": 0.05,
            "lot_size": 50,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        }
    ]
    df = _instruments_to_df(raw)
    expiry_val = df["expiry"][0]
    assert isinstance(expiry_val, str)
    assert "2024" in expiry_val


# ---------------------------------------------------------------------------
# InstrumentManager.refresh
# ---------------------------------------------------------------------------


def test_refresh_downloads_and_persists_instruments(tmp_path: Path) -> None:
    """refresh() should call kite.instruments() and write instruments.parquet."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)

    df = manager.refresh()

    mock_kite.instruments.assert_called_once()
    instruments_path = tmp_path / "reference" / "instruments.parquet"
    assert instruments_path.exists()
    assert len(df) == len(raw)


def test_refresh_updates_in_memory_cache(tmp_path: Path) -> None:
    """After refresh, get_instruments() should return the cached DataFrame."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)

    manager.refresh()
    # Second call must not hit the network again
    df = manager.get_instruments()

    assert mock_kite.instruments.call_count == 1
    assert len(df) == len(raw)


# ---------------------------------------------------------------------------
# InstrumentManager.get_instruments
# ---------------------------------------------------------------------------


def test_get_instruments_loads_from_disk_on_cache_miss(tmp_path: Path) -> None:
    """get_instruments() should load from disk when in-memory cache is empty."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)

    # First manager: refresh and write to disk
    manager1 = InstrumentManager(mock_kite, tmp_path)
    manager1.refresh()

    # Second manager (fresh cache): should load from disk, not call API
    mock_kite2 = MagicMock()
    manager2 = InstrumentManager(mock_kite2, tmp_path)
    df = manager2.get_instruments()

    mock_kite2.instruments.assert_not_called()
    assert len(df) == len(raw)


def test_get_instruments_raises_when_no_file_and_no_cache(tmp_path: Path) -> None:
    """get_instruments() should raise FileNotFoundError if no data exists."""
    manager = InstrumentManager(MagicMock(), tmp_path)
    with pytest.raises(FileNotFoundError, match="instruments"):
        manager.get_instruments()


# ---------------------------------------------------------------------------
# InstrumentManager.resolve_token
# ---------------------------------------------------------------------------


def test_resolve_token_returns_correct_integer(tmp_path: Path) -> None:
    """resolve_token should return the integer token for a known symbol."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)
    manager.refresh()

    token = manager.resolve_token("BANKNIFTY", "NSE")
    assert token == 260105


def test_resolve_token_falls_back_without_exchange(tmp_path: Path) -> None:
    """resolve_token should fall back to exchange-less search when needed."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)
    manager.refresh()

    # RELIANCE is on NSE; requesting with BSE should fall back
    token = manager.resolve_token("RELIANCE", "BSE")
    assert token == 738561


def test_resolve_token_raises_for_unknown_symbol(tmp_path: Path) -> None:
    """resolve_token should raise KeyError for an unrecognised symbol."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)
    manager.refresh()

    with pytest.raises(KeyError, match="UNKNOWN"):
        manager.resolve_token("UNKNOWN")


# ---------------------------------------------------------------------------
# InstrumentManager.resolve_exchange
# ---------------------------------------------------------------------------


def test_resolve_exchange_returns_string(tmp_path: Path) -> None:
    """resolve_exchange should return the exchange string for a known symbol."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)
    manager.refresh()

    exchange = manager.resolve_exchange("BANKNIFTY")
    assert exchange == "NSE"


# ---------------------------------------------------------------------------
# InstrumentManager.invalidate_cache
# ---------------------------------------------------------------------------


def test_invalidate_cache_clears_in_memory_state(tmp_path: Path) -> None:
    """invalidate_cache should force a disk reload on next get_instruments()."""
    raw = _sample_instruments()
    mock_kite = _make_mock_kite(raw)
    manager = InstrumentManager(mock_kite, tmp_path)
    manager.refresh()

    manager.invalidate_cache()
    # After invalidation the cache is None; next call loads from disk
    df = manager.get_instruments()
    assert len(df) == len(raw)
