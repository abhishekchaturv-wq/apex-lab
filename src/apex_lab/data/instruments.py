"""Instrument master management.

Downloads the complete Kite instrument list, persists it locally as Parquet,
and provides fast cached lookups by trading symbol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.config import get_logger
from apex_lab.data.storage import get_instruments_path, read_parquet, write_parquet

logger = get_logger(__name__)


class InstrumentManager:
    """Manages the Kite instrument master.

    Downloads, caches and serves instrument metadata so that symbol strings
    can be resolved to integer instrument tokens required by the Kite
    historical data API.

    Args:
        kite: Authenticated :class:`kiteconnect.KiteConnect` instance.
        data_dir: Root data directory; instruments are stored under
            ``data_dir/reference/instruments.parquet``.
    """

    def __init__(self, kite: Any, data_dir: Path) -> None:
        self._kite = kite
        self._data_dir = data_dir
        self._cache: pl.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def refresh(self) -> pl.DataFrame:
        """Download the complete instrument list and persist to disk.

        Returns:
            Instruments DataFrame (columns mirror the Kite API response).
        """
        logger.info("Refreshing instrument master from Kite API")
        raw: list[dict[str, Any]] = self._kite.instruments()
        df = _instruments_to_df(raw)
        path = get_instruments_path(self._data_dir)
        write_parquet(df, path)
        self._cache = df
        logger.info("Instrument master refreshed: %d instruments saved to %s", len(df), path)
        return df

    def get_instruments(self) -> pl.DataFrame:
        """Return the instruments DataFrame, loading from disk when necessary.

        Returns:
            Instruments DataFrame.

        Raises:
            FileNotFoundError: If instruments have never been downloaded.
        """
        if self._cache is not None:
            return self._cache

        path = get_instruments_path(self._data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Instruments file not found at {path}. " "Call refresh_instruments() first."
            )

        logger.debug("Loading instruments from %s", path)
        self._cache = read_parquet(path)
        return self._cache

    def resolve_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Resolve a trading symbol to its Kite instrument token.

        Args:
            symbol: Trading symbol (e.g., ``"BANKNIFTY"``).
            exchange: Exchange to restrict the lookup to. Defaults to
                ``"NSE"``.

        Returns:
            Integer instrument token.

        Raises:
            KeyError: If no instrument matches ``symbol`` on ``exchange``.
        """
        instruments = self.get_instruments()
        match = instruments.filter(
            (pl.col("tradingsymbol") == symbol) & (pl.col("exchange") == exchange)
        )
        if len(match) == 0:
            # Try without exchange restriction
            match = instruments.filter(pl.col("tradingsymbol") == symbol)
        if len(match) == 0:
            raise KeyError(
                f"Symbol '{symbol}' not found in instrument master. "
                "Call refresh_instruments() to update."
            )
        token: int = match.select("instrument_token").row(0)[0]
        logger.debug("Resolved %s/%s -> token %d", symbol, exchange, token)
        return token

    def resolve_exchange(self, symbol: str, exchange: str = "NSE") -> str:
        """Resolve a symbol to its canonical exchange string.

        Args:
            symbol: Trading symbol.
            exchange: Preferred exchange (used when multiple matches exist).

        Returns:
            Exchange string (e.g., ``"NSE"``).

        Raises:
            KeyError: If the symbol is not found.
        """
        instruments = self.get_instruments()
        match = instruments.filter(
            (pl.col("tradingsymbol") == symbol) & (pl.col("exchange") == exchange)
        )
        if len(match) == 0:
            match = instruments.filter(pl.col("tradingsymbol") == symbol)
        if len(match) == 0:
            raise KeyError(f"Symbol '{symbol}' not found in instrument master.")
        resolved: str = match.select("exchange").row(0)[0]
        return resolved

    def invalidate_cache(self) -> None:
        """Clear the in-memory instruments cache.

        The next call to :meth:`get_instruments` will reload from disk.
        """
        self._cache = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _instruments_to_df(raw: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert the raw Kite instruments list to a Polars DataFrame.

    Handles heterogeneous ``expiry`` values (``datetime.date`` objects or
    empty strings) by normalising them to plain strings.

    Args:
        raw: List of instrument dicts returned by ``kite.instruments()``.

    Returns:
        Polars DataFrame with consistent column types.
    """
    if not raw:
        return pl.DataFrame()

    # Normalise values that may have mixed types
    normalised: list[dict[str, Any]] = []
    for row in raw:
        normalised_row = dict(row)
        # expiry can be a datetime.date or ""
        expiry = normalised_row.get("expiry")
        if expiry is not None and not isinstance(expiry, str):
            normalised_row["expiry"] = str(expiry) if expiry else ""
        elif expiry is None:
            normalised_row["expiry"] = ""
        # instrument_token and exchange_token to int
        for int_col in ("instrument_token", "exchange_token", "lot_size"):
            val = normalised_row.get(int_col)
            if val is not None:
                normalised_row[int_col] = int(val)
        # float columns
        for float_col in ("last_price", "strike", "tick_size"):
            val = normalised_row.get(float_col)
            if val is not None:
                normalised_row[float_col] = float(val)
        normalised.append(normalised_row)

    return pl.DataFrame(normalised)
