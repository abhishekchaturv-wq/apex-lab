"""Kite client for Zerodha API communication.

Provides a production-grade client that abstracts Zerodha Kite Connect API.
Do NOT expose Kite SDK objects outside this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import polars as pl
from kiteconnect import KiteConnect

from apex_lab.config import get_logger, settings
from apex_lab.data.kite.auth import KiteAuthenticator
from apex_lab.data.kite.retry import retry
from apex_lab.exceptions import (
    AuthenticationError,
    DownloadError,
    ValidationError,
    RateLimitError,
)

logger = get_logger(__name__)


class KiteClient:
    """Authenticated client for Zerodha Kite API.

    This is the single entry point for all Kite API communication.
    It handles:
    - Authentication and session management
    - Retry logic with exponential backoff
    - Rate limiting
    - Data normalization

    Public methods return Polars DataFrames for type safety and performance.

    Attributes:
        _authenticator: Kite authentication handler
        _session: Authenticated KiteConnect session (cached)
    """

    def __init__(self) -> None:
        """Initialize Kite client.

        Raises:
            AuthenticationError: If credentials are not configured
        """
        self._authenticator = KiteAuthenticator()
        self._session: Optional[KiteConnect] = None

        logger.info("KiteClient initialized")

    def _get_session(self) -> KiteConnect:
        """Get or create authenticated Kite session.

        Returns:
            Authenticated KiteConnect instance

        Raises:
            AuthenticationError: If session cannot be created
        """
        if self._session is None:
            self._session = self._authenticator.create_session()
        return self._session

    @retry(max_retries=3, base_delay=1.0)
    def get_history(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        continuous: bool = False,
        oi: bool = False,
    ) -> pl.DataFrame:
        """Fetch historical OHLC data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BANKNIFTY', 'RELIANCE')
            interval: Candle interval ('minute', '5minute', '15minute', etc.)
            start: Start datetime for data retrieval
            end: End datetime for data retrieval
            continuous: Whether to return continuous contract data
            oi: Whether to include open interest data

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValidationError: If parameters are invalid
            DownloadError: If API call fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If not authenticated

        Example:
            >>> client = KiteClient()
            >>> df = client.get_history(
            ...     symbol='BANKNIFTY',
            ...     interval='5minute',
            ...     start=datetime(2024, 1, 1),
            ...     end=datetime(2024, 1, 31)
            ... )
            >>> print(df.head())
        """
        # Validate parameters
        self._validate_symbol(symbol)
        self._validate_interval(interval)
        self._validate_dates(start, end)

        try:
            session = self._get_session()

            if not self._authenticator.has_access_token:
                raise AuthenticationError(
                    "No access token available. Cannot fetch historical data."
                )

            logger.info(
                f"Fetching {symbol} {interval} data from {start.date()} to {end.date()}"
            )

            # Call Kite API
            data = session.historical_data(
                instrument_token=self._get_instrument_token(symbol),
                from_date=start,
                to_date=end,
                interval=interval,
                continuous=continuous,
                oi=oi,
            )

            # Convert to Polars DataFrame
            df = self._normalize_data(data)

            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df

        except RateLimitError:
            # Re-raise rate limit errors to trigger retry
            raise
        except AuthenticationError:
            # Don't retry authentication errors
            raise
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            raise DownloadError(f"Historical data download failed: {e}") from e

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        """Validate trading symbol format.

        Args:
            symbol: Symbol to validate

        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError(f"Invalid symbol: {symbol}")

        if len(symbol) < 2:
            raise ValidationError(f"Symbol too short: {symbol}")

    @staticmethod
    def _validate_interval(interval: str) -> None:
        """Validate candle interval.

        Args:
            interval: Interval to validate

        Raises:
            ValidationError: If interval is not supported
        """
        valid_intervals = {
            "minute",
            "5minute",
            "15minute",
            "30minute",
            "60minute",
            "day",
            "week",
            "month",
        }

        if interval not in valid_intervals:
            raise ValidationError(
                f"Invalid interval: {interval}. "
                f"Must be one of {valid_intervals}"
            )

    @staticmethod
    def _validate_dates(start: datetime, end: datetime) -> None:
        """Validate date range.

        Args:
            start: Start datetime
            end: End datetime

        Raises:
            ValidationError: If dates are invalid
        """
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValidationError("Dates must be datetime objects")

        if start >= end:
            raise ValidationError(f"Start date {start} must be before end date {end}")

        # Check if range is too large (more than 2 years)
        if (end - start).days > 730:
            logger.warning("Data range exceeds 2 years. This may take a long time.")

    @staticmethod
    def _get_instrument_token(symbol: str) -> int:
        """Get instrument token for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Instrument token (numeric ID)

        Note:
            This is a placeholder. In production, this would query
            an instrument master database or Kite API.
            For now, we'll raise a NotImplementedError.
        """
        # TODO: Implement instrument token lookup
        raise NotImplementedError(
            f"Instrument token lookup not yet implemented for {symbol}"
        )

    @staticmethod
    def _normalize_data(data: list[dict]) -> pl.DataFrame:
        """Normalize Kite API response to Polars DataFrame.

        Args:
            data: Raw data from Kite API

        Returns:
            Normalized Polars DataFrame with standard columns

        Raises:
            ValidationError: If data cannot be normalized
        """
        if not data:
            raise ValidationError("No data returned from API")

        try:
            df = pl.DataFrame(
                {
                    "timestamp": [d["date"] for d in data],
                    "open": [float(d["open"]) for d in data],
                    "high": [float(d["high"]) for d in data],
                    "low": [float(d["low"]) for d in data],
                    "close": [float(d["close"]) for d in data],
                    "volume": [int(d["volume"]) for d in data],
                }
            )

            return df.sort("timestamp")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to normalize data: {e}")
            raise ValidationError(f"Data normalization failed: {e}") from e
