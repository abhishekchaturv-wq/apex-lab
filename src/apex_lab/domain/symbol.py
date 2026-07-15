"""Symbol domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Symbol:
    """Represents a tradable symbol in an exchange namespace.

    Attributes:
        exchange: Exchange identifier (for example, NSE).
        symbol: Exchange symbol text (for example, RELIANCE, NIFTY 50).
        instrument_token: Optional broker instrument token.
    """

    exchange: str
    symbol: str
    instrument_token: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize symbol fields."""
        exchange = self.exchange.strip()
        symbol = self.symbol.strip()

        if not exchange:
            raise ValueError("exchange must not be empty")

        if not symbol:
            raise ValueError("symbol must not be empty")

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "symbol", symbol)

    @property
    def full_name(self) -> str:
        """Return canonical full symbol format: EXCHANGE:SYMBOL."""
        return f"{self.exchange}:{self.symbol}"
