"""Symbol-level metadata for downloaded market data.

Each downloaded symbol/interval pair has a sidecar metadata file persisted
as JSON alongside the Parquet data file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class SymbolMetadata:
    """Metadata record for a downloaded symbol dataset.

    Attributes:
        symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
        exchange: Exchange (e.g., ``"NSE"``).
        interval: Candle interval (e.g., ``"30minute"``).
        first_candle: ISO-8601 timestamp of the earliest candle, or ``None``.
        last_candle: ISO-8601 timestamp of the latest candle, or ``None``.
        num_candles: Total number of candles in the dataset.
        download_timestamp: ISO-8601 UTC timestamp of last download.
        source: Data source identifier.
    """

    symbol: str
    exchange: str
    interval: str
    first_candle: str | None
    last_candle: str | None
    num_candles: int
    download_timestamp: str
    source: str = "zerodha_kite"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a JSON-serialisable dictionary.

        Returns:
            Dictionary suitable for ``json.dumps``.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolMetadata:
        """Construct a :class:`SymbolMetadata` from a plain dictionary.

        Args:
            data: Dictionary with the same keys as the dataclass fields.

        Returns:
            SymbolMetadata instance.
        """
        return cls(**data)


def build_symbol_metadata(
    *,
    symbol: str,
    exchange: str,
    interval: str,
    df: pl.DataFrame,
    timestamp_column: str = "timestamp",
) -> SymbolMetadata:
    """Build :class:`SymbolMetadata` from a downloaded OHLCV DataFrame.

    Args:
        symbol: Trading symbol.
        exchange: Exchange string (e.g., ``"NSE"``).
        interval: Candle interval string.
        df: Downloaded OHLCV DataFrame.
        timestamp_column: Name of the timestamp column.

    Returns:
        Populated :class:`SymbolMetadata` instance.
    """
    first_candle: str | None = None
    last_candle: str | None = None

    if len(df) > 0 and timestamp_column in df.columns:
        first_ts = df.select(pl.col(timestamp_column).min()).item()
        last_ts = df.select(pl.col(timestamp_column).max()).item()
        first_candle = first_ts.isoformat() if hasattr(first_ts, "isoformat") else str(first_ts)
        last_candle = last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts)

    return SymbolMetadata(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        first_candle=first_candle,
        last_candle=last_candle,
        num_candles=len(df),
        download_timestamp=datetime.now(tz=UTC).isoformat(),
    )


def save_metadata(metadata: SymbolMetadata, path: Path) -> None:
    """Persist metadata to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        metadata: Metadata instance to persist.
        path: Destination ``.metadata.json`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2))


def load_metadata(path: Path) -> SymbolMetadata:
    """Load metadata from a JSON file.

    Args:
        path: Source ``.metadata.json`` file path.

    Returns:
        :class:`SymbolMetadata` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    data: dict[str, Any] = json.loads(path.read_text())
    return SymbolMetadata.from_dict(data)
