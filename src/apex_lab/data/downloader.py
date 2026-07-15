"""Core historical data downloader for Zerodha Kite Connect.

Responsibilities
----------------
* Split a long date range into API-compliant chunks.
* Download each chunk with exponential-backoff retry.
* Persist completed chunks to a local cache for resume support.
* Merge chunks, deduplicate and sort.
* Validate and save final Parquet files.
* Provide the :class:`DataEngine` facade that ties all sub-modules together.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.config import get_logger
from apex_lab.data.instruments import InstrumentManager
from apex_lab.data.metadata import build_symbol_metadata, save_metadata
from apex_lab.data.storage import (
    get_chunk_path,
    get_metadata_path,
    get_raw_path,
    merge_and_dedup,
    read_parquet,
    write_parquet,
)
from apex_lab.data.validator import assert_valid_ohlcv

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Kite API rate-limit constants
# ---------------------------------------------------------------------------

#: Maximum number of calendar days per historical data request, keyed by
#: interval string.  Values are conservative to stay well within Kite limits.
INTERVAL_MAX_DAYS: dict[str, int] = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 100,
    "30minute": 100,
    "60minute": 400,
    "day": 2000,
}

#: All valid interval strings accepted by the Kite API.
VALID_INTERVALS: frozenset[str] = frozenset(INTERVAL_MAX_DAYS)

#: Default maximum retry attempts for transient API failures.
DEFAULT_MAX_RETRIES: int = 3

#: Base delay (seconds) for exponential backoff.
DEFAULT_BASE_DELAY: float = 1.0


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str | date) -> date:
    """Parse a date-like value to a :class:`datetime.date`.

    Accepts:

    * A :class:`datetime.date` or :class:`datetime.datetime` instance.
    * The string ``"today"`` (case-insensitive).
    * An ISO-8601 date string ``"YYYY-MM-DD"``.

    Args:
        value: Date value to parse.

    Returns:
        A :class:`datetime.date` instance.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value.lower() == "today":
        return date.today()
    return date.fromisoformat(value)


def split_into_chunks(start: date, end: date, interval: str) -> list[tuple[date, date]]:
    """Split a date range into API-compliant chunks for a given interval.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        interval: Kite interval string (e.g., ``"30minute"``).

    Returns:
        List of ``(chunk_start, chunk_end)`` tuples, each spanning at most
        ``INTERVAL_MAX_DAYS[interval]`` calendar days.

    Raises:
        ValueError: If *interval* is not a supported Kite interval.
        ValueError: If *start* is after *end*.
    """
    if interval not in INTERVAL_MAX_DAYS:
        raise ValueError(
            f"Unsupported interval '{interval}'. " f"Valid intervals: {sorted(INTERVAL_MAX_DAYS)}"
        )
    if start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")

    max_days = INTERVAL_MAX_DAYS[interval]
    chunks: list[tuple[date, date]] = []
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=max_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    return chunks


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _fetch_with_retry(
    fetch_fn: Any,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> Any:
    """Call *fetch_fn* with exponential-backoff retry on transient errors.

    Args:
        fetch_fn: Zero-argument callable that performs the API request.
        max_retries: Maximum number of retry attempts (not including the
            initial attempt).
        base_delay: Base delay in seconds; actual delay is
            ``base_delay * 2 ** attempt``.

    Returns:
        Return value of *fetch_fn* on success.

    Raises:
        Exception: Re-raises the last exception after *max_retries* attempts.
    """
    from kiteconnect import exceptions as kite_exc  # noqa: PLC0415

    retryable = (kite_exc.NetworkException, kite_exc.DataException)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fetch_fn()
        except retryable as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = base_delay * (2**attempt)
            logger.warning(
                "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
        except Exception:
            raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Raw Kite response → Polars DataFrame
# ---------------------------------------------------------------------------


def _candles_to_df(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert the raw Kite historical data list to a Polars DataFrame.

    Args:
        raw: List of candle dicts returned by ``kite.historical_data()``.
            Each dict has keys: ``date``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, and optionally ``oi``.

    Returns:
        Polars OHLCV DataFrame with a ``datetime`` *timestamp* column.
    """
    if not raw:
        return pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Int64),
            }
        )

    timestamps = [r["date"] for r in raw]
    opens = [float(r["open"]) for r in raw]
    highs = [float(r["high"]) for r in raw]
    lows = [float(r["low"]) for r in raw]
    closes = [float(r["close"]) for r in raw]
    volumes = [int(r["volume"]) for r in raw]

    data: dict[str, Any] = {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }

    if raw and "oi" in raw[0]:
        data["oi"] = [int(r["oi"]) for r in raw]

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# KiteDownloader
# ---------------------------------------------------------------------------


class KiteDownloader:
    """Downloads historical OHLCV data from Zerodha Kite Connect.

    Handles chunk splitting, retry, resume support and final merge/save.

    Args:
        kite: Authenticated :class:`kiteconnect.KiteConnect` instance.
        instruments: :class:`~apex_lab.data.instruments.InstrumentManager`
            for symbol→token resolution.
        data_dir: Root data directory.
        max_retries: Maximum retry attempts per chunk.
        base_delay: Base delay (seconds) for exponential backoff.
    """

    def __init__(
        self,
        kite: Any,
        instruments: InstrumentManager,
        data_dir: Path,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        self._kite = kite
        self._instruments = instruments
        self._data_dir = data_dir
        self._max_retries = max_retries
        self._base_delay = base_delay

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download(
        self,
        symbol: str,
        interval: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """Download complete historical data for *symbol* over the date range.

        Requests are automatically split into Kite-compliant chunks.
        Completed chunks are cached on disk so an interrupted download can
        be resumed without re-downloading already-fetched data.

        Args:
            symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
            interval: Kite interval string.
            start_date: Inclusive start date.
            end_date: Inclusive end date (or ``"today"``).

        Returns:
            Deduplicated, sorted OHLCV DataFrame persisted to disk.
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)

        logger.info("Starting download: %s %s from %s to %s", symbol, interval, start, end)

        token = self._instruments.resolve_token(symbol)
        exchange = self._instruments.resolve_exchange(symbol)
        chunks = split_into_chunks(start, end, interval)

        logger.info("Split into %d chunk(s) for %s [%s]", len(chunks), symbol, interval)

        frames = self._download_chunks(symbol, interval, token, chunks)

        if not frames:
            logger.warning("No data returned for %s [%s]", symbol, interval)
            df = _candles_to_df([])
        else:
            df = merge_and_dedup(frames)

        self._save(symbol, exchange, interval, df)
        self._cleanup_chunks(symbol, interval, len(chunks))

        logger.info("Download complete: %s [%s] — %d candles saved", symbol, interval, len(df))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_chunks(
        self,
        symbol: str,
        interval: str,
        token: int,
        chunks: list[tuple[date, date]],
    ) -> list[pl.DataFrame]:
        """Download all chunks, using cached files when available.

        Args:
            symbol: Trading symbol (for cache key).
            interval: Candle interval.
            token: Kite instrument token.
            chunks: List of ``(start, end)`` date tuples.

        Returns:
            List of per-chunk DataFrames.
        """
        frames: list[pl.DataFrame] = []

        for idx, (chunk_start, chunk_end) in enumerate(chunks):
            cache_path = get_chunk_path(self._data_dir, symbol, interval, idx)

            if cache_path.exists():
                logger.debug(
                    "Resuming: chunk %d/%d loaded from cache (%s)",
                    idx + 1,
                    len(chunks),
                    cache_path,
                )
                frames.append(read_parquet(cache_path))
                continue

            logger.debug(
                "Downloading chunk %d/%d: %s – %s",
                idx + 1,
                len(chunks),
                chunk_start,
                chunk_end,
            )

            raw = _fetch_with_retry(
                lambda s=chunk_start, e=chunk_end: self._kite.historical_data(
                    token,
                    from_date=s.strftime("%Y-%m-%d %H:%M:%S"),
                    to_date=e.strftime("%Y-%m-%d %H:%M:%S"),
                    interval=interval,
                    oi=True,
                ),
                max_retries=self._max_retries,
                base_delay=self._base_delay,
            )

            chunk_df = _candles_to_df(raw)
            write_parquet(chunk_df, cache_path)
            frames.append(chunk_df)

            logger.debug("Chunk %d/%d done: %d candles", idx + 1, len(chunks), len(chunk_df))

        return frames

    def _save(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        df: pl.DataFrame,
    ) -> None:
        """Validate, save Parquet and write sidecar metadata.

        Args:
            symbol: Trading symbol.
            exchange: Exchange string.
            interval: Candle interval.
            df: Final merged DataFrame.
        """
        logger.info("Validating %s [%s] (%d candles)", symbol, interval, len(df))
        assert_valid_ohlcv(df)

        raw_path = get_raw_path(self._data_dir, interval, symbol)
        write_parquet(df, raw_path)
        logger.info("Saved %s", raw_path)

        meta_path = get_metadata_path(self._data_dir, interval, symbol)
        meta = build_symbol_metadata(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            df=df,
        )
        save_metadata(meta, meta_path)
        logger.debug("Metadata saved to %s", meta_path)

    def _cleanup_chunks(self, symbol: str, interval: str, num_chunks: int) -> None:
        """Remove temporary chunk cache files after a successful download.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            num_chunks: Number of chunks that were downloaded.
        """
        for idx in range(num_chunks):
            cache_path = get_chunk_path(self._data_dir, symbol, interval, idx)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug("Removed chunk cache %s", cache_path)


# ---------------------------------------------------------------------------
# DataEngine — high-level facade
# ---------------------------------------------------------------------------


class DataEngine:
    """High-level facade for the historical data engine.

    Ties together :class:`InstrumentManager`, :class:`KiteDownloader` and the
    incremental update logic into a single cohesive API.

    Args:
        kite: Authenticated :class:`kiteconnect.KiteConnect` instance.
        data_dir: Root data directory (from settings).
        max_retries: Maximum retry attempts per download chunk.
        base_delay: Base delay (seconds) for exponential backoff.
    """

    def __init__(
        self,
        kite: Any,
        data_dir: Path,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        self._kite = kite
        self._data_dir = data_dir
        self.instruments = InstrumentManager(kite, data_dir)
        self._dl = KiteDownloader(
            kite,
            self.instruments,
            data_dir,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_symbol(
        self,
        symbol: str,
        interval: str,
        start_date: str | date,
        end_date: str | date,
    ) -> pl.DataFrame:
        """Download historical data for a single symbol.

        Args:
            symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
            interval: Candle interval.
            start_date: Inclusive start date or ``"today"``.
            end_date: Inclusive end date or ``"today"``.

        Returns:
            Downloaded OHLCV DataFrame.
        """
        return self._dl.download(symbol, interval, start_date, end_date)

    def download_universe(
        self,
        symbols: list[str],
        interval: str,
        start_date: str | date,
        end_date: str | date,
    ) -> dict[str, pl.DataFrame]:
        """Download historical data for multiple symbols.

        Args:
            symbols: List of trading symbols.
            interval: Candle interval.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            Dictionary mapping each symbol to its OHLCV DataFrame.
        """
        results: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            logger.info("Downloading universe symbol: %s", symbol)
            results[symbol] = self._dl.download(symbol, interval, start_date, end_date)
        return results

    # ------------------------------------------------------------------
    # Update (delegates to updater module)
    # ------------------------------------------------------------------

    def update_symbol(self, symbol: str, interval: str) -> pl.DataFrame:
        """Incrementally update a single symbol with only missing candles.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.

        Returns:
            Updated OHLCV DataFrame.
        """
        from apex_lab.data.updater import update_symbol as _update  # noqa: PLC0415

        return _update(symbol, interval, engine=self)

    def update_universe(self, symbols: list[str], interval: str) -> dict[str, pl.DataFrame]:
        """Incrementally update multiple symbols with only missing candles.

        Args:
            symbols: List of trading symbols.
            interval: Candle interval.

        Returns:
            Dictionary mapping each symbol to its updated OHLCV DataFrame.
        """
        from apex_lab.data.updater import update_symbol as _update  # noqa: PLC0415

        results: dict[str, pl.DataFrame] = {}
        for symbol in symbols:
            logger.info("Updating universe symbol: %s", symbol)
            results[symbol] = _update(symbol, interval, engine=self)
        return results

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def refresh_instruments(self) -> pl.DataFrame:
        """Refresh the instrument master from Kite API.

        Returns:
            Up-to-date instruments DataFrame.
        """
        return self.instruments.refresh()
