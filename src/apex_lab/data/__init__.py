"""Historical market data engine for APEX Lab.

This module provides a production-grade pipeline for downloading, storing,
validating and incrementally updating market data from Zerodha Kite Connect.

It is the single source of truth for all historical market data used by APEX.

Public API
----------
::

    from apex_lab.data import (
        download_symbol,
        download_universe,
        update_symbol,
        update_universe,
        refresh_instruments,
        DataEngine,
    )

Example
-------
Download complete BANKNIFTY 30-minute history::

    download_symbol("BANKNIFTY", "30minute", "2016-01-01", "today")

Update only missing candles for a symbol::

    update_symbol("BANKNIFTY", "30minute")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.data.downloader import DataEngine

__all__ = [
    "DataEngine",
    "download_symbol",
    "download_universe",
    "update_symbol",
    "update_universe",
    "refresh_instruments",
]


# ---------------------------------------------------------------------------
# Internal factory
# ---------------------------------------------------------------------------


def _create_engine(
    kite: Any | None,
    data_dir: Path | None,
) -> DataEngine:
    """Create a :class:`DataEngine` from the provided or default settings.

    Args:
        kite: Optional authenticated :class:`kiteconnect.KiteConnect` instance.
            If ``None`` a new instance is created from settings.
        data_dir: Optional data root directory.  Uses ``settings.data_dir``
            when ``None``.

    Returns:
        Configured :class:`DataEngine` instance.
    """
    from apex_lab.config import settings  # noqa: PLC0415

    resolved_dir = data_dir or settings.data_dir
    resolved_kite = kite if kite is not None else _kite_from_settings()
    return DataEngine(resolved_kite, resolved_dir)


def _kite_from_settings() -> Any:
    """Instantiate a :class:`kiteconnect.KiteConnect` from project settings.

    Returns:
        Configured :class:`kiteconnect.KiteConnect` instance.
    """
    from kiteconnect import KiteConnect  # noqa: PLC0415

    from apex_lab.config import settings  # noqa: PLC0415

    kite = KiteConnect(api_key=settings.kite_api_key)
    if settings.kite_access_token:
        kite.set_access_token(settings.kite_access_token)
    return kite


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------


def download_symbol(
    symbol: str,
    interval: str,
    start_date: str | date,
    end_date: str | date,
    *,
    kite: Any | None = None,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """Download historical data for a single symbol.

    Args:
        symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
        interval: Candle interval. One of: ``minute``, ``3minute``,
            ``5minute``, ``10minute``, ``15minute``, ``30minute``,
            ``60minute``, ``day``.
        start_date: Inclusive start date — ISO-8601 string, ``datetime.date``,
            or the string ``"today"``.
        end_date: Inclusive end date — same formats as *start_date*.
        kite: Optional :class:`kiteconnect.KiteConnect` instance.
            Created from project settings when ``None``.
        data_dir: Optional data root directory.
            Uses ``settings.data_dir`` when ``None``.

    Returns:
        Downloaded, validated and persisted OHLCV :class:`polars.DataFrame`.

    Example:
        >>> df = download_symbol("BANKNIFTY", "30minute", "2016-01-01", "today")
    """
    engine = _create_engine(kite, data_dir)
    return engine.download_symbol(symbol, interval, start_date, end_date)


def download_universe(
    symbols: list[str],
    interval: str,
    start_date: str | date,
    end_date: str | date,
    *,
    kite: Any | None = None,
    data_dir: Path | None = None,
) -> dict[str, pl.DataFrame]:
    """Download historical data for a list of symbols.

    Args:
        symbols: List of trading symbols.
        interval: Candle interval.
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        kite: Optional :class:`kiteconnect.KiteConnect` instance.
        data_dir: Optional data root directory.

    Returns:
        Dict mapping each symbol to its OHLCV :class:`polars.DataFrame`.
    """
    engine = _create_engine(kite, data_dir)
    return engine.download_universe(symbols, interval, start_date, end_date)


def update_symbol(
    symbol: str,
    interval: str,
    *,
    kite: Any | None = None,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """Incrementally update a symbol with only the missing candles.

    Args:
        symbol: Trading symbol.
        interval: Candle interval.
        kite: Optional :class:`kiteconnect.KiteConnect` instance.
        data_dir: Optional data root directory.

    Returns:
        Updated OHLCV :class:`polars.DataFrame`.

    Example:
        >>> df = update_symbol("BANKNIFTY", "30minute")
    """
    engine = _create_engine(kite, data_dir)
    return engine.update_symbol(symbol, interval)


def update_universe(
    symbols: list[str],
    interval: str,
    *,
    kite: Any | None = None,
    data_dir: Path | None = None,
) -> dict[str, pl.DataFrame]:
    """Incrementally update multiple symbols with only missing candles.

    Args:
        symbols: List of trading symbols.
        interval: Candle interval.
        kite: Optional :class:`kiteconnect.KiteConnect` instance.
        data_dir: Optional data root directory.

    Returns:
        Dict mapping each symbol to its updated OHLCV :class:`polars.DataFrame`.
    """
    engine = _create_engine(kite, data_dir)
    return engine.update_universe(symbols, interval)


def refresh_instruments(
    *,
    kite: Any | None = None,
    data_dir: Path | None = None,
) -> pl.DataFrame:
    """Refresh the instrument master from the Kite API.

    Downloads the complete instrument list and persists it to
    ``data_dir/reference/instruments.parquet``.

    Args:
        kite: Optional :class:`kiteconnect.KiteConnect` instance.
        data_dir: Optional data root directory.

    Returns:
        Instruments :class:`polars.DataFrame`.
    """
    engine = _create_engine(kite, data_dir)
    return engine.refresh_instruments()
