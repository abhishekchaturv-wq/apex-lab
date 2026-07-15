"""Incremental update logic for the historical data engine.

Reads the existing Parquet file for a symbol, determines the latest
timestamp already stored, downloads only the missing candles and merges
them back, keeping the dataset deduplicated and sorted.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from apex_lab.config import get_logger
from apex_lab.data.metadata import build_symbol_metadata, save_metadata
from apex_lab.data.storage import (
    get_metadata_path,
    get_raw_path,
    merge_and_dedup,
    read_parquet,
    write_parquet,
)
from apex_lab.data.validator import assert_valid_ohlcv

if TYPE_CHECKING:
    from apex_lab.data.downloader import DataEngine

logger = get_logger(__name__)


def update_symbol(
    symbol: str,
    interval: str,
    *,
    engine: DataEngine,
) -> pl.DataFrame:
    """Incrementally update a symbol by appending only missing candles.

    If no existing data is found the full history is downloaded.

    Algorithm
    ---------
    1. Read existing Parquet (if any) and find the latest timestamp.
    2. Set ``new_start = latest_date + 1 day``.
    3. Download from ``new_start`` to today via :class:`~apex_lab.data.downloader.KiteDownloader`.
    4. Concatenate old + new frames.
    5. Deduplicate, sort, validate.
    6. Overwrite the Parquet file.
    7. Update sidecar metadata.

    Args:
        symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
        interval: Candle interval string.
        engine: :class:`~apex_lab.data.downloader.DataEngine` instance that
            provides the downloader and data directory reference.

    Returns:
        Updated OHLCV DataFrame.
    """
    data_dir: Path = engine._data_dir  # noqa: SLF001
    raw_path = get_raw_path(data_dir, interval, symbol)

    if not raw_path.exists():
        logger.info("No existing data for %s [%s] — performing full download", symbol, interval)
        return engine.download_symbol(symbol, interval, "2010-01-01", "today")

    existing = read_parquet(raw_path)
    if len(existing) == 0:
        logger.warning(
            "Existing Parquet for %s [%s] is empty — performing full download",
            symbol,
            interval,
        )
        return engine.download_symbol(symbol, interval, "2010-01-01", "today")

    # Find latest stored timestamp
    latest_ts = existing.select(pl.col("timestamp").max()).item()
    latest_date: date = (
        latest_ts.date() if hasattr(latest_ts, "date") else date.fromisoformat(str(latest_ts))
    )
    new_start = latest_date + timedelta(days=1)
    today = date.today()

    if new_start > today:
        logger.info("%s [%s] is already up to date (latest: %s)", symbol, interval, latest_date)
        return existing

    logger.info(
        "Incremental update for %s [%s]: downloading from %s to %s",
        symbol,
        interval,
        new_start,
        today,
    )

    # Download only the missing window
    new_data = engine._dl.download(symbol, interval, new_start, today)  # noqa: SLF001

    if len(new_data) == 0:
        logger.info("No new candles for %s [%s]", symbol, interval)
        return existing

    # Merge, dedup, sort
    merged = merge_and_dedup([existing, new_data])

    # Validate
    assert_valid_ohlcv(merged)

    # Overwrite parquet
    write_parquet(merged, raw_path)
    logger.info(
        "Updated %s [%s]: %d new candle(s), total %d",
        symbol,
        interval,
        len(merged) - len(existing),
        len(merged),
    )

    # Update metadata
    exchange = engine.instruments.resolve_exchange(symbol)
    meta_path = get_metadata_path(data_dir, interval, symbol)
    meta = build_symbol_metadata(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        df=merged,
    )
    save_metadata(meta, meta_path)

    return merged
