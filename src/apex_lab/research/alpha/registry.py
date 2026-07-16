"""Registry helpers for Alpha Scoring Engine categories."""

from __future__ import annotations

from dataclasses import dataclass

import apex_lab.research.context.features as _context_features_module  # noqa: F401
from apex_lab.research.context.registry import ContextFeature, get_registry

_GROUP_TO_CATEGORY = {
    "trend": "trend",
    "momentum": "momentum",
    "volatility": "volatility",
    "vwap": "vwap",
    "market_structure": "market_structure",
    "opening_session": "opening_range",
    "time": "time",
}

CATEGORY_ORDER: tuple[str, ...] = (
    "trend",
    "momentum",
    "volatility",
    "vwap",
    "market_structure",
    "opening_range",
    "time",
)


@dataclass(frozen=True)
class AlphaFeatureSpec:
    """Context feature metadata required by alpha scoring."""

    name: str
    group: str
    category: str
    feature: ContextFeature


def get_alpha_feature_registry() -> dict[str, AlphaFeatureSpec]:
    """Return context features annotated with alpha scoring categories."""
    registry = get_registry()
    return {
        name: AlphaFeatureSpec(
            name=name,
            group=feature.group,
            category=_GROUP_TO_CATEGORY.get(feature.group, feature.group),
            feature=feature,
        )
        for name, feature in registry.items()
    }
