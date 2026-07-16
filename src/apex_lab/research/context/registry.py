"""Context feature registry for the Alpha Discovery Engine.

Features are registered at import time by calling :func:`register_context_feature`.
The engine iterates over all registered features without hardcoding them, making
it possible to add new features in future PRs without modifying the engine.

Example::

    from apex_lab.research.context.registry import (
        ContextFeature,
        register_context_feature,
    )

    class MyNewFeature(ContextFeature):
        ...

    register_context_feature(MyNewFeature())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

_FEATURE_REGISTRY: dict[str, ContextFeature] = {}


class ContextFeature(ABC):
    """Abstract base class for all context features.

    Each subclass must:

    - Declare a unique ``name`` (registry key and column prefix).
    - Declare a ``group`` for ``best_features.json`` grouping.
    - Implement :meth:`compute` to idempotently append indicator columns.
    - Implement :meth:`label` to return categorical bucket strings.
    - Implement :meth:`numeric` to return raw numeric values for correlation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique key used in the registry."""

    @property
    @abstractmethod
    def group(self) -> str:
        """Feature group; used as key in ``best_features.json``."""

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Idempotently append indicator columns required by this feature.

        If all required columns are already present, return *df* unchanged.

        Args:
            df: OHLCV DataFrame (may already contain columns from earlier features).

        Returns:
            DataFrame with this feature's indicator columns added (or unchanged).
        """

    @abstractmethod
    def label(self, df: pl.DataFrame) -> pl.Series:
        """Return a ``pl.Utf8`` Series of bucket labels, one per row.

        Rows where the underlying indicator is null yield a null label.

        Args:
            df: DataFrame already passed through :meth:`compute`.

        Returns:
            String Series of the same length as *df*.
        """

    @abstractmethod
    def numeric(self, df: pl.DataFrame) -> pl.Series:
        """Return a ``pl.Float64`` Series of raw numeric values for correlation.

        Args:
            df: DataFrame already passed through :meth:`compute`.

        Returns:
            Float64 Series of the same length as *df*, nulls where missing.
        """

    def metadata(self) -> dict[str, Any]:
        """Return a dict describing the feature's parameters and logic."""
        return {"name": self.name, "group": self.group}


def register_context_feature(feature: ContextFeature) -> ContextFeature:
    """Register *feature* in the global registry and return it.

    This is the only mechanism by which a feature becomes known to the engine.
    Future PRs can add features by calling this function in :mod:`features`
    without modifying the engine.

    Args:
        feature: A :class:`ContextFeature` instance.

    Returns:
        The same *feature* (allows use as a one-liner after instantiation).

    Example::

        register_context_feature(MyNewFeature())
    """
    _FEATURE_REGISTRY[feature.name] = feature
    return feature


def get_registry() -> dict[str, ContextFeature]:
    """Return a snapshot of the current global feature registry.

    Returns:
        Dict mapping feature name → :class:`ContextFeature` instance.
    """
    return dict(_FEATURE_REGISTRY)
