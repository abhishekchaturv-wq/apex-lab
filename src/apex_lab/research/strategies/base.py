"""Abstract base class for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """Interface every strategy must implement.

    A strategy encapsulates entry and exit logic for a single trading idea.
    It is responsible for:

    - Adding required indicator columns to a raw OHLCV DataFrame (``prepare``).
    - Producing a boolean entry-signal Series (``entry_condition``).
    - Producing a boolean exit-signal Series (``exit_condition``).
    - Declaring which feature keys it depends on (``required_features``).
    - Describing itself for reporting (``description``).

    Strategies must not contain any reporting logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name used as a key in the strategy registry."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the strategy logic."""

    @property
    @abstractmethod
    def required_features(self) -> list[str]:
        """List of factor keys this strategy depends on (e.g. ``["EMA", "RSI"]``)."""

    @abstractmethod
    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append all indicator columns required by this strategy.

        Implementations must be idempotent: columns that are already present
        must not be re-added.

        Args:
            df: Raw OHLCV DataFrame sorted by ``timestamp``.

        Returns:
            New DataFrame with all original columns plus strategy indicators.
        """

    @abstractmethod
    def entry_condition(self, df: pl.DataFrame) -> pl.Series:
        """Return a boolean Series that is ``True`` at valid entry bars.

        Args:
            df: DataFrame already processed by :meth:`prepare`.

        Returns:
            Boolean ``pl.Series`` aligned with *df*.
        """

    @abstractmethod
    def exit_condition(self, df: pl.DataFrame) -> pl.Series:
        """Return a boolean Series that is ``True`` at valid exit bars.

        Args:
            df: DataFrame already processed by :meth:`prepare`.

        Returns:
            Boolean ``pl.Series`` aligned with *df*.
        """
