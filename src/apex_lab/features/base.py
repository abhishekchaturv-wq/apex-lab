"""Abstract base class for feature groups.

Defines the contract that all feature groups must implement. Each feature group
is responsible for computing a cohesive set of related technical indicators and
returning them as new columns appended to the input DataFrame.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import polars as pl

logger = logging.getLogger(__name__)


class FeatureGroup(ABC):
    """Abstract base class for all feature groups.

    A feature group encapsulates the logic for computing a set of related
    technical indicators. Subclasses must implement :meth:`compute` and
    declare a unique :attr:`name`.

    All computations must be fully vectorised using Polars expressions and
    must not rely on Python-level loops over rows.

    Example:
        >>> class MyFeatures(FeatureGroup):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_features"
        ...
        ...     def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        ...         return df.with_columns(
        ...             (pl.col("close") * 2).alias("double_close")
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this feature group.

        Returns:
            A lowercase string identifier (e.g. ``"price"``, ``"trend"``).
        """

    @property
    def warm_up_periods(self) -> int:
        """Minimum number of leading rows required for stable outputs.

        Rows within the warm-up window may contain ``null`` values in the
        computed features and should typically be excluded from model training.

        Returns:
            Number of warm-up rows (default ``0``).
        """
        return 0

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute features and return a DataFrame with new columns appended.

        Args:
            df: Input OHLCV DataFrame.  Must contain at least the columns
                required by this feature group (see each concrete class for
                its column requirements).

        Returns:
            The input DataFrame with additional feature columns appended.
            The row order and index are preserved.

        Raises:
            ValueError: If required columns are missing from *df*.
        """

    def _require_columns(self, df: pl.DataFrame, columns: list[str]) -> None:
        """Validate that all required columns are present in the DataFrame.

        Args:
            df: DataFrame to validate.
            columns: List of required column names.

        Raises:
            ValueError: If any required column is absent.
        """
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required columns: {missing}. "
                f"Available columns: {df.columns}"
            )
        logger.debug("[%s] Required columns validated: %s", self.name, columns)
