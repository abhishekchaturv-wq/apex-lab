"""Interval conversion helpers.

This module is the single source of truth for translating interval strings
between TradingView, Kite, and internal Timeframe representations.
"""

from __future__ import annotations

from apex_lab.domain.timeframe import Timeframe

_TRADINGVIEW_TO_KITE: dict[str, str] = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "1d": "day",
}

_KITE_TO_TIMEFRAME: dict[str, Timeframe] = {
    "minute": Timeframe.ONE_MINUTE,
    "3minute": Timeframe.THREE_MINUTE,
    "5minute": Timeframe.FIVE_MINUTE,
    "15minute": Timeframe.FIFTEEN_MINUTE,
    "30minute": Timeframe.THIRTY_MINUTE,
    "60minute": Timeframe.ONE_HOUR,
    "day": Timeframe.ONE_DAY,
}

_TIMEFRAME_TO_TRADINGVIEW: dict[Timeframe, str] = {
    timeframe: tradingview
    for tradingview, kite in _TRADINGVIEW_TO_KITE.items()
    for timeframe in [_KITE_TO_TIMEFRAME[kite]]
}


def tradingview_to_kite(interval: str) -> str:
    """Convert TradingView interval to Kite interval.

    Args:
        interval: TradingView interval string.

    Returns:
        Kite-compatible interval string.

    Raises:
        ValueError: If interval is unsupported.
    """
    key = interval.strip().lower()
    if key not in _TRADINGVIEW_TO_KITE:
        raise ValueError(f"unsupported TradingView interval: {interval}")
    return _TRADINGVIEW_TO_KITE[key]


def kite_to_timeframe(interval: str) -> Timeframe:
    """Convert Kite interval string to internal timeframe."""
    key = interval.strip().lower()
    if key not in _KITE_TO_TIMEFRAME:
        raise ValueError(f"unsupported Kite interval: {interval}")
    return _KITE_TO_TIMEFRAME[key]


def tradingview_to_timeframe(interval: str) -> Timeframe:
    """Convert TradingView interval directly to internal timeframe."""
    return kite_to_timeframe(tradingview_to_kite(interval))


def timeframe_to_tradingview(timeframe: Timeframe) -> str:
    """Convert internal timeframe to TradingView interval string."""
    if timeframe not in _TIMEFRAME_TO_TRADINGVIEW:
        raise ValueError(f"unsupported Timeframe: {timeframe}")
    return _TIMEFRAME_TO_TRADINGVIEW[timeframe]


def timeframe_to_kite(timeframe: Timeframe) -> str:
    """Convert internal timeframe to Kite interval string."""
    tradingview_interval = timeframe_to_tradingview(timeframe)
    return _TRADINGVIEW_TO_KITE[tradingview_interval]
