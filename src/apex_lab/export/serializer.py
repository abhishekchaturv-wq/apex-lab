"""Serialization helpers for the Pine Script strategy summary JSON.

The summary document captures all metadata about a generation run so that
the exact Pine Script output can be reproduced and audited.
"""

from __future__ import annotations

import datetime
from typing import Any

GENERATOR_VERSION = "1.0.0"
PINE_VERSION = "5"
_RESEARCH_VERSION = "0.1.0"


def build_summary(
    best_params: dict[str, Any],
    best_features: dict[str, Any],
    weights_data: dict[str, Any],
    input_files: list[str],
) -> dict[str, Any]:
    """Build the strategy summary dictionary.

    Args:
        best_params: Walk-forward optimisation output (``best_parameters.json``).
        best_features: Context discovery best features (``best_features.json``).
        weights_data: Alpha scoring weights (``weights.json``).
        input_files: Absolute or relative paths of the files consumed.

    Returns:
        A JSON-serialisable dictionary describing the generation run.
    """
    weights: list[dict[str, Any]] = weights_data.get("weights", [])

    # Summarise alpha weights by category
    alpha_weights_by_category: dict[str, float] = {}
    for entry in weights:
        cat = entry.get("category", "unknown")
        alpha_weights_by_category[cat] = round(
            alpha_weights_by_category.get(cat, 0.0) + entry.get("weight", 0.0),
            6,
        )

    return {
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "research_version": _RESEARCH_VERSION,
        "pine_version": PINE_VERSION,
        "ema_parameters": {
            "fast": int(best_params.get("fast_ema", 50)),
            "slow": int(best_params.get("slow_ema", 200)),
        },
        "context_filters": list(best_features.keys()),
        "alpha_weights": alpha_weights_by_category,
        "input_files_used": input_files,
    }
