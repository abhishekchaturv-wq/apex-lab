"""Candle domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Candle:
    """Represents a single OHLCV candlestick.

    Attributes:
        timestamp: Candle timestamp.
        open: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Traded volume.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """Validate OHLCV invariants."""
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")

        if not self.low <= self.open <= self.high:
            raise ValueError("open must be between low and high")

        if not self.low <= self.close <= self.high:
            raise ValueError("close must be between low and high")

        if self.volume < 0:
            raise ValueError("volume must be greater than or equal to 0")

    @property
    def body_size(self) -> float:
        """Return absolute body size: |close - open|."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Return the upper wick length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Return the lower wick length."""
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        """Return total candle range: high - low."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Return True when close is greater than open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Return True when close is less than open."""
        return self.close < self.open

    @property
    def typical_price(self) -> float:
        """Return the typical price: (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def median_price(self) -> float:
        """Return the median price: (high + low) / 2."""
        return (self.high + self.low) / 2.0

    @property
    def weighted_price(self) -> float:
        """Return the weighted close price: (high + low + 2*close) / 4."""
        return (self.high + self.low + (2.0 * self.close)) / 4.0
