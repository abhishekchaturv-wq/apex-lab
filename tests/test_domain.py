"""Tests for core domain models."""

from __future__ import annotations

from datetime import datetime

import pytest

from apex_lab.domain.candle import Candle
from apex_lab.domain.interval import (
    kite_to_timeframe,
    timeframe_to_kite,
    timeframe_to_tradingview,
    tradingview_to_kite,
    tradingview_to_timeframe,
)
from apex_lab.domain.symbol import Symbol
from apex_lab.domain.timeframe import Timeframe


def test_candle_validates_and_computed_properties() -> None:
    candle = Candle(
        timestamp=datetime(2026, 1, 1, 9, 15),
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1_000.0,
    )

    assert candle.body_size == 5.0
    assert candle.upper_wick == 5.0
    assert candle.lower_wick == 5.0
    assert candle.range == 15.0
    assert candle.is_bullish is True
    assert candle.is_bearish is False
    assert candle.typical_price == pytest.approx((110.0 + 95.0 + 105.0) / 3.0)
    assert candle.median_price == 102.5
    assert candle.weighted_price == pytest.approx((110.0 + 95.0 + 2.0 * 105.0) / 4.0)


def test_candle_invalid_high_low_rejected() -> None:
    with pytest.raises(ValueError, match="high"):
        Candle(
            timestamp=datetime(2026, 1, 1, 9, 15),
            open=100.0,
            high=90.0,
            low=95.0,
            close=98.0,
            volume=100.0,
        )


def test_candle_invalid_open_close_rejected() -> None:
    with pytest.raises(ValueError, match="open"):
        Candle(
            timestamp=datetime(2026, 1, 1, 9, 15),
            open=120.0,
            high=110.0,
            low=95.0,
            close=100.0,
            volume=100.0,
        )

    with pytest.raises(ValueError, match="close"):
        Candle(
            timestamp=datetime(2026, 1, 1, 9, 15),
            open=100.0,
            high=110.0,
            low=95.0,
            close=120.0,
            volume=100.0,
        )


def test_candle_negative_volume_rejected() -> None:
    with pytest.raises(ValueError, match="volume"):
        Candle(
            timestamp=datetime(2026, 1, 1, 9, 15),
            open=100.0,
            high=110.0,
            low=95.0,
            close=100.0,
            volume=-1.0,
        )


def test_timeframe_properties() -> None:
    assert Timeframe.ONE_MINUTE.minutes == 1
    assert Timeframe.ONE_HOUR.minutes == 60
    assert Timeframe.ONE_DAY.minutes == 1440
    assert Timeframe.THIRTY_MINUTE.label == "30m"


def test_interval_conversions() -> None:
    assert tradingview_to_kite("30m") == "30minute"
    assert kite_to_timeframe("30minute") == Timeframe.THIRTY_MINUTE
    assert tradingview_to_timeframe("1h") == Timeframe.ONE_HOUR
    assert timeframe_to_tradingview(Timeframe.FIFTEEN_MINUTE) == "15m"
    assert timeframe_to_kite(Timeframe.ONE_DAY) == "day"


def test_interval_invalid_values_rejected() -> None:
    with pytest.raises(ValueError):
        tradingview_to_kite("2m")

    with pytest.raises(ValueError):
        kite_to_timeframe("2minute")


def test_symbol_formatting_and_validation() -> None:
    symbol = Symbol(exchange="NSE", symbol="RELIANCE")
    assert symbol.full_name == "NSE:RELIANCE"

    nifty = Symbol(exchange="NSE", symbol="NIFTY 50")
    assert nifty.full_name == "NSE:NIFTY 50"

    bank = Symbol(exchange=" NSE ", symbol=" NIFTY BANK ")
    assert bank.full_name == "NSE:NIFTY BANK"

    with pytest.raises(ValueError):
        Symbol(exchange="", symbol="RELIANCE")

    with pytest.raises(ValueError):
        Symbol(exchange="NSE", symbol="")
