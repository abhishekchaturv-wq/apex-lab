"""Abstract interface for market data providers.

Defines the contract that all data providers must implement.
Allows swapping between different data sources (Kite, CSV, Parquet, etc.)
without changing consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import polars as pl


class MarketDataProvider(ABC):
    """Abstract base class for market data providers.

    All concrete implementations (KiteProvider, CSVProvider, etc.) must
    inherit from this class and implement all abstract methods.

    This ensures a consistent interface across different data sources.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source.

        Raises:
            AuthenticationError: If connection fails due to invalid credentials
            ConnectionError: If connection to data source fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source.

        Should be called when the provider is no longer needed.
        This is also called automatically when exiting a context manager.
        """
        pass

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch historical OHLC data.

        Args:
            symbol: Trading symbol (e.g., 'BANKNIFTY', 'RELIANCE')
            interval: Candle interval (e.g., '5minute', 'day')
            start: Start datetime for data retrieval
            end: End datetime for data retrieval

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValidationError: If parameters are invalid
            DownloadError: If data retrieval fails
            RateLimitError: If rate limit is exceeded
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported by this provider.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is valid, False otherwise

        Raises:
            ValidationError: If symbol format is invalid
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is healthy and operational.

        Returns:
            True if provider is healthy, False otherwise
        """
        pass

    def __enter__(self) -> MarketDataProvider:
        """Context manager entry.

        Returns:
            Self for use in 'with' statement
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.disconnect()
