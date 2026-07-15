"""Feature engineering module for computing technical indicators and features.

Importing this package triggers automatic registration of all built-in feature
groups into :data:`~apex_lab.features.registry.default_registry`.

Quick start::

    import polars as pl
    from apex_lab.features import FeatureEngine

    engine = FeatureEngine()
    df_enriched = engine.compute(raw_ohlcv_df)
"""

# Importing groups triggers self-registration into default_registry.
import apex_lab.features.groups  # noqa: F401, E402
from apex_lab.features.base import FeatureGroup
from apex_lab.features.engine import FeatureEngine
from apex_lab.features.registry import FeatureRegistry, default_registry

__all__ = [
    "FeatureEngine",
    "FeatureGroup",
    "FeatureRegistry",
    "default_registry",
]
