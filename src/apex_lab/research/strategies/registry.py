"""Strategy registry and built-in strategy implementations.

Six strategies are registered by default:

- EMA Crossover
- Opening Range Breakout
- VWAP Trend
- EMA + VWAP
- EMA + RSI
- EMA + ATR Expansion
"""

from __future__ import annotations

import polars as pl

from apex_lab.research.factors.atr_volatility import AtrVolatilityFactor
from apex_lab.research.factors.ema_trend import EmaTrendFactor
from apex_lab.research.factors.rsi import RsiFactor
from apex_lab.research.factors.vwap import VwapFactor
from apex_lab.research.strategies.base import Strategy

# Shared factor instances (stateless — safe to share)
_EMA = EmaTrendFactor()
_VWAP = VwapFactor()
_RSI = RsiFactor()
_ATR = AtrVolatilityFactor()

# ORB opening range: first N bars of each session
_ORB_BARS = 2


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


class EmaCrossoverStrategy(Strategy):
    """EMA 20/50 bullish crossover strategy.

    Entry: EMA20 crosses above EMA50.
    Exit:  EMA20 crosses below EMA50.
    """

    @property
    def name(self) -> str:
        return "EMA Crossover"

    @property
    def description(self) -> str:
        return "Enter on EMA20/EMA50 bullish crossover; exit on bearish crossover."

    @property
    def required_features(self) -> list[str]:
        return ["EMA"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        return _EMA.compute(df)

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        return df["bullish_crossover"].fill_null(False)

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        return df["bearish_crossover"].fill_null(False)


class OpeningRangeBreakoutStrategy(Strategy):
    """Opening Range Breakout (ORB) strategy.

    Entry: Close crosses above the high of the first bar of each session.
    Exit:  Close crosses below the low of the first bar of each session.

    EMA columns are computed as infrastructure to satisfy the backtester's
    required-column contract (``ema_200``, ``atr_pct``).
    """

    @property
    def name(self) -> str:
        return "Opening Range Breakout"

    @property
    def description(self) -> str:
        return (
            "Enter when close crosses above the session opening-range high; "
            "exit when close crosses below the opening-range low."
        )

    @property
    def required_features(self) -> list[str]:
        return ["EMA"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        enriched = _EMA.compute(df)
        if "orb_high" in enriched.columns:
            return enriched

        # Compute session opening-range high/low (first _ORB_BARS bars per day).
        date_col = pl.col("timestamp").dt.date()
        session_orb = (
            enriched.with_columns(date_col.alias("_orb_date"))
            .with_row_index("_orb_idx")
            .filter(
                pl.col("_orb_idx")
                < (
                    pl.col("_orb_idx")
                    .min()
                    .over("_orb_date")
                    + _ORB_BARS
                )
            )
            .group_by("_orb_date")
            .agg(
                [
                    pl.col("high").max().alias("orb_high"),
                    pl.col("low").min().alias("orb_low"),
                ]
            )
        )
        return (
            enriched.with_columns(date_col.alias("_orb_date"))
            .join(session_orb, on="_orb_date", how="left")
            .drop("_orb_date")
        )

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        above = df["close"] > df["orb_high"]
        prev_above = df["close"].shift(1) > df["orb_high"].shift(1)
        return (above & ~prev_above.fill_null(True)).fill_null(False)

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        below = df["close"] < df["orb_low"]
        prev_below = df["close"].shift(1) < df["orb_low"].shift(1)
        return (below & ~prev_below.fill_null(True)).fill_null(False)


class VwapTrendStrategy(Strategy):
    """VWAP Trend strategy.

    Entry: Close crosses above the session VWAP.
    Exit:  Close crosses below the session VWAP.
    """

    @property
    def name(self) -> str:
        return "VWAP Trend"

    @property
    def description(self) -> str:
        return "Enter on close crossing above session VWAP; exit on crossing below."

    @property
    def required_features(self) -> list[str]:
        return ["EMA", "VWAP"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        enriched = _EMA.compute(df)
        return _VWAP.compute(enriched)

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        above = df["close"] > df["vwap"]
        prev_above = df["close"].shift(1) > df["vwap"].shift(1)
        return (above & ~prev_above.fill_null(True)).fill_null(False)

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        below = df["close"] < df["vwap"]
        prev_below = df["close"].shift(1) < df["vwap"].shift(1)
        return (below & ~prev_below.fill_null(True)).fill_null(False)


class EmaVwapStrategy(Strategy):
    """EMA crossover confirmed by VWAP.

    Entry: EMA20/50 bullish crossover AND close above session VWAP.
    Exit:  EMA20/50 bearish crossover.
    """

    @property
    def name(self) -> str:
        return "EMA + VWAP"

    @property
    def description(self) -> str:
        return "EMA bullish crossover confirmed by close above VWAP; exit on bearish crossover."

    @property
    def required_features(self) -> list[str]:
        return ["EMA", "VWAP"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        enriched = _EMA.compute(df)
        return _VWAP.compute(enriched)

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        ema_signal = _EMA.signal(df).fill_null(False)
        vwap_signal = _VWAP.signal(df).fill_null(False)
        return ema_signal & vwap_signal

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        return df["bearish_crossover"].fill_null(False)


class EmaRsiStrategy(Strategy):
    """EMA crossover confirmed by RSI momentum.

    Entry: EMA20/50 bullish crossover AND RSI > 50.
    Exit:  EMA20/50 bearish crossover.
    """

    @property
    def name(self) -> str:
        return "EMA + RSI"

    @property
    def description(self) -> str:
        return "EMA bullish crossover confirmed by RSI > 50; exit on bearish crossover."

    @property
    def required_features(self) -> list[str]:
        return ["EMA", "RSI"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        enriched = _EMA.compute(df)
        return _RSI.compute(enriched)

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        ema_signal = _EMA.signal(df).fill_null(False)
        rsi_signal = _RSI.signal(df).fill_null(False)
        return ema_signal & rsi_signal

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        return df["bearish_crossover"].fill_null(False)


class EmaAtrExpansionStrategy(Strategy):
    """EMA crossover filtered by ATR volatility regime.

    Entry: EMA20/50 bullish crossover AND ATR percentile in [20, 80].
    Exit:  EMA20/50 bearish crossover.
    """

    @property
    def name(self) -> str:
        return "EMA + ATR Expansion"

    @property
    def description(self) -> str:
        return (
            "EMA bullish crossover when ATR percentile is between 20 and 80 "
            "(normal volatility); exit on bearish crossover."
        )

    @property
    def required_features(self) -> list[str]:
        return ["EMA", "ATR"]

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        # EmaTrendFactor already computes atr_pct — AtrVolatilityFactor is idempotent.
        return _EMA.compute(df)

    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        ema_signal = _EMA.signal(df).fill_null(False)
        atr_signal = _ATR.signal(df).fill_null(False)
        return ema_signal & atr_signal

    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        return df["bearish_crossover"].fill_null(False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class StrategyRegistry:
    """Registry mapping strategy names to :class:`~base.Strategy` instances.

    Strategies are stored in insertion order.  Names are case-insensitive for
    lookups via :meth:`get`.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        """Register a strategy.

        Args:
            strategy: The :class:`~base.Strategy` instance to add.

        Raises:
            ValueError: If a strategy with the same name is already registered.
        """
        key = strategy.name.lower()
        if key in self._strategies:
            raise ValueError(f"Strategy '{strategy.name}' is already registered.")
        self._strategies[key] = strategy

    def get(self, name: str) -> Strategy:
        """Return the strategy registered under *name* (case-insensitive).

        Raises:
            KeyError: If no strategy with that name is registered.
        """
        key = name.lower()
        if key not in self._strategies:
            available = ", ".join(self._strategies)
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return self._strategies[key]

    def all_strategies(self) -> list[Strategy]:
        """Return all registered strategies in insertion order."""
        return list(self._strategies.values())

    def names(self) -> list[str]:
        """Return all registered strategy names in insertion order."""
        return [s.name for s in self._strategies.values()]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        return f"StrategyRegistry({self.names()})"


def _build_default_registry() -> StrategyRegistry:
    """Create a new :class:`StrategyRegistry` pre-loaded with all built-in strategies."""
    registry = StrategyRegistry()
    registry.register(EmaCrossoverStrategy())
    registry.register(OpeningRangeBreakoutStrategy())
    registry.register(VwapTrendStrategy())
    registry.register(EmaVwapStrategy())
    registry.register(EmaRsiStrategy())
    registry.register(EmaAtrExpansionStrategy())
    return registry


def get_default_registry() -> StrategyRegistry:
    """Return a fresh default :class:`StrategyRegistry` with all built-in strategies."""
    return _build_default_registry()
