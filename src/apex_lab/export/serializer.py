"""Serialization helpers for the Pine Script strategy summary JSON.

The summary document captures all metadata about a generation run so that
the exact Pine Script output can be reproduced and audited.
"""

from __future__ import annotations

import datetime
from typing import Any

from apex_lab.export.signal_pattern_loader import SignalPattern

GENERATOR_VERSION = "1.0.0"
PINE_VERSION = "5"
_RESEARCH_VERSION = "0.1.0"


def build_summary(
    best_params: dict[str, Any] | None,
    best_features: dict[str, Any] | None,
    weights_data: dict[str, Any] | None,
    input_files: list[str],
    signal_pattern: SignalPattern | None = None,
) -> dict[str, Any]:
    """Build the strategy summary dictionary.

    Args:
        best_params: Walk-forward optimisation output (``best_parameters.json``).
        best_features: Context discovery best features (``best_features.json``).
        weights_data: Alpha scoring weights (``weights.json``).
        input_files: Absolute or relative paths of the files consumed.
        signal_pattern: Optional discovered signal pattern for signal export mode.

    Returns:
        A JSON-serialisable dictionary describing the generation run.
    """
    weights: list[dict[str, Any]] = (weights_data or {}).get("weights", [])

    # Summarise alpha weights by category
    alpha_weights_by_category: dict[str, float] = {}
    for entry in weights:
        cat = entry.get("category", "unknown")
        alpha_weights_by_category[cat] = round(
            alpha_weights_by_category.get(cat, 0.0) + entry.get("weight", 0.0),
            6,
        )

    summary = {
        "generation_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "research_version": _RESEARCH_VERSION,
        "pine_version": PINE_VERSION,
        "ema_parameters": {
            "fast": int((best_params or {}).get("fast_ema", 50)),
            "slow": int((best_params or {}).get("slow_ema", 200)),
        },
        "context_filters": list((best_features or {}).keys()),
        "alpha_weights": alpha_weights_by_category,
        "input_files_used": input_files,
    }
    if signal_pattern is not None:
        summary["signal_pattern"] = {
            "rank": signal_pattern.rank,
            "rule_label": signal_pattern.rule_label,
            "features": list(signal_pattern.features),
            "conditions": list(signal_pattern.conditions),
            "combination_size": signal_pattern.combination_size,
            "signal_frequency": signal_pattern.signal_frequency,
            "win_rate": signal_pattern.win_rate,
            "average_return": signal_pattern.average_return,
            "expectancy": signal_pattern.expectancy,
            "average_mfe": signal_pattern.average_mfe,
            "average_mae": signal_pattern.average_mae,
            "robustness": signal_pattern.robustness,
            "diversity_score": signal_pattern.diversity_score,
            "composite_score": signal_pattern.composite_score,
        }
    return summary
