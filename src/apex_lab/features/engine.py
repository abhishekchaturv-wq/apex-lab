"""Feature computation engine.

The :class:`FeatureEngine` orchestrates the execution of :class:`FeatureGroup`
instances retrieved from a :class:`FeatureRegistry`.  It contains *zero*
feature-specific logic: all domain knowledge lives inside the feature groups.

Example:
    >>> import polars as pl
    >>> from apex_lab.features.engine import FeatureEngine
    >>> engine = FeatureEngine()
    >>> df_with_features = engine.compute(raw_ohlcv_df)
"""

from __future__ import annotations

import logging
import time

import polars as pl

from apex_lab.features.registry import FeatureRegistry, default_registry

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Orchestrates feature computation across multiple :class:`FeatureGroup` instances.

    The engine iterates over the requested groups in registration order,
    calls each group's :meth:`~FeatureGroup.compute` method, and merges
    the resulting columns back into the accumulated DataFrame.

    Args:
        registry: Registry of feature groups to draw from.  If ``None``,
            the module-level :data:`~apex_lab.features.registry.default_registry`
            is used.

    Example:
        >>> from apex_lab.features.engine import FeatureEngine
        >>> engine = FeatureEngine()
        >>> out = engine.compute(df, groups=["price", "trend"])
    """

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        """Initialise the engine with an optional registry.

        Args:
            registry: :class:`FeatureRegistry` to use.  Defaults to
                :data:`~apex_lab.features.registry.default_registry`.
        """
        self._registry: FeatureRegistry = registry if registry is not None else default_registry
        logger.debug(
            "FeatureEngine initialised with registry containing %d groups",
            len(self._registry),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pl.DataFrame,
        groups: list[str] | None = None,
    ) -> pl.DataFrame:
        """Compute features for the requested groups and return the enriched DataFrame.

        Groups are applied in the order they appear in the registry.  When
        *groups* is ``None`` every registered group is applied.

        Args:
            df: Input OHLCV DataFrame with at minimum the columns required by
                the requested feature groups.
            groups: Optional list of group names to compute.  If ``None``,
                all registered groups are computed.

        Returns:
            A new DataFrame equal to *df* with additional feature columns
            appended.  The original columns and row order are preserved.

        Raises:
            KeyError: If any name in *groups* is not registered.
            ValueError: If a feature group is missing required columns in *df*.
        """
        requested: list[str] = groups if groups is not None else self._registry.list_groups()

        if not requested:
            logger.warning("FeatureEngine.compute called but no groups are registered/requested.")
            return df

        logger.info(
            "Computing features for %d group(s): %s  [rows=%d]",
            len(requested),
            requested,
            len(df),
        )

        result = df
        wall_start = time.perf_counter()

        for name in requested:
            group = self._registry.get(name)
            t0 = time.perf_counter()
            result = group.compute(result)
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            logger.debug(
                "  %-12s  warm_up=%3d  elapsed=%.1f ms",
                name,
                group.warm_up_periods,
                elapsed_ms,
            )

        total_ms = (time.perf_counter() - wall_start) * 1_000
        new_cols = len(result.columns) - len(df.columns)
        logger.info(
            "Feature computation complete: %d new column(s) in %.1f ms",
            new_cols,
            total_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def warm_up_periods(self, groups: list[str] | None = None) -> int:
        """Return the maximum warm-up window across the requested groups.

        Args:
            groups: Group names to inspect.  Defaults to all registered groups.

        Returns:
            Maximum :attr:`~FeatureGroup.warm_up_periods` value.
        """
        names = groups if groups is not None else self._registry.list_groups()
        if not names:
            return 0
        return max(self._registry.get(n).warm_up_periods for n in names)

    @property
    def registry(self) -> FeatureRegistry:
        """Expose the underlying registry (read-only accessor).

        Returns:
            The :class:`FeatureRegistry` this engine is bound to.
        """
        return self._registry

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"FeatureEngine(registry={self._registry!r})"
