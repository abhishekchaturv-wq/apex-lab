"""Abstract base class for research factors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class Factor(ABC):
    """Interface every research factor must implement.

    A factor is responsible for:

    - Adding indicator columns to a raw OHLCV DataFrame (``compute``).
    - Producing a boolean entry-signal Series from the enriched DataFrame (``signal``).
    - Describing itself for reporting purposes (``metadata``).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name used as a key in the factor registry."""

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append indicator columns to *df* and return the enriched DataFrame.

        Implementations must be idempotent: if the required columns are
        already present they should not be re-added (to avoid conflicts when
        the engine calls multiple factors in sequence).

        Args:
            df: OHLCV DataFrame that may already contain columns added by
                previously-evaluated factors.

        Returns:
            New DataFrame with all input columns plus the factor's indicator
            columns.
        """

    @abstractmethod
    def signal(self, df: pl.DataFrame) -> pl.Series:
        """Return a boolean Series representing the entry condition.

        The engine ANDs the signals from all factors in a combination to
        produce a combined entry signal.  Null values are treated as ``False``.

        Args:
            df: DataFrame that has already been passed through ``compute``.

        Returns:
            Boolean ``pl.Series`` with the same length as *df*.
        """

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a dictionary describing the factor's parameters and logic.

        Returns:
            Arbitrary key/value pairs suitable for logging or report headers.
        """
