"""Parquet storage helpers for market data.

All market data is stored exclusively as Parquet files.
No CSV files are written or read by this module.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Required columns for a valid OHLCV Parquet file.
OHLCV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

#: Optional additional columns.
OHLCV_OPTIONAL_COLUMNS: tuple[str, ...] = ("oi",)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_raw_path(data_dir: Path, interval: str, symbol: str) -> Path:
    """Return the Parquet path for a raw symbol dataset.

    Args:
        data_dir: Root data directory (from settings).
        interval: Candle interval string (e.g., ``"30minute"``).
        symbol: Trading symbol (e.g., ``"BANKNIFTY"``).

    Returns:
        ``data_dir/raw/{interval}/{symbol}.parquet``
    """
    return data_dir / "raw" / interval / f"{symbol}.parquet"


def get_metadata_path(data_dir: Path, interval: str, symbol: str) -> Path:
    """Return the metadata JSON path for a raw symbol dataset.

    Args:
        data_dir: Root data directory.
        interval: Candle interval string.
        symbol: Trading symbol.

    Returns:
        ``data_dir/raw/{interval}/{symbol}.metadata.json``
    """
    return data_dir / "raw" / interval / f"{symbol}.metadata.json"


def get_instruments_path(data_dir: Path) -> Path:
    """Return the path for the instruments master Parquet file.

    Args:
        data_dir: Root data directory.

    Returns:
        ``data_dir/reference/instruments.parquet``
    """
    return data_dir / "reference" / "instruments.parquet"


def get_chunk_path(data_dir: Path, symbol: str, interval: str, chunk_index: int) -> Path:
    """Return a temporary chunk file path used for resume support.

    Args:
        data_dir: Root data directory.
        symbol: Trading symbol.
        interval: Candle interval.
        chunk_index: Zero-based chunk index.

    Returns:
        ``data_dir/cache/{symbol}_{interval}/chunk_{chunk_index:04d}.parquet``
    """
    return data_dir / "cache" / f"{symbol}_{interval}" / f"chunk_{chunk_index:04d}.parquet"


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a Parquet file into a Polars DataFrame.

    Args:
        path: Path to an existing Parquet file.

    Returns:
        Polars DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    return pl.read_parquet(path)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to a Parquet file.

    Creates parent directories if they do not exist.

    Args:
        df: DataFrame to persist.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def merge_and_dedup(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate a list of DataFrames, deduplicate timestamps and sort.

    Args:
        frames: List of OHLCV DataFrames to merge.

    Returns:
        Single deduplicated DataFrame sorted by timestamp.

    Raises:
        ValueError: If *frames* is empty.
    """
    if not frames:
        raise ValueError("frames must not be empty")

    merged = pl.concat(frames, how="diagonal")
    return merged.unique(subset=["timestamp"], keep="first").sort("timestamp")
