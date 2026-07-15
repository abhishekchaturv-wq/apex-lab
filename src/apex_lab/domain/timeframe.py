"""Timeframe domain enum."""

from __future__ import annotations

from enum import Enum


class Timeframe(str, Enum):
    """Supported internal timeframe values."""

    ONE_MINUTE = "1m"
    THREE_MINUTE = "3m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"

    @property
    def minutes(self) -> int:
        """Return timeframe length in minutes."""
        mapping: dict[Timeframe, int] = {
            Timeframe.ONE_MINUTE: 1,
            Timeframe.THREE_MINUTE: 3,
            Timeframe.FIVE_MINUTE: 5,
            Timeframe.FIFTEEN_MINUTE: 15,
            Timeframe.THIRTY_MINUTE: 30,
            Timeframe.ONE_HOUR: 60,
            Timeframe.ONE_DAY: 1440,
        }
        return mapping[self]

    @property
    def label(self) -> str:
        """Return a human-readable label."""
        mapping: dict[Timeframe, str] = {
            Timeframe.ONE_MINUTE: "1m",
            Timeframe.THREE_MINUTE: "3m",
            Timeframe.FIVE_MINUTE: "5m",
            Timeframe.FIFTEEN_MINUTE: "15m",
            Timeframe.THIRTY_MINUTE: "30m",
            Timeframe.ONE_HOUR: "1h",
            Timeframe.ONE_DAY: "1d",
        }
        return mapping[self]
