"""Target labels and output schema for reversal supervision."""

from __future__ import annotations

from enum import StrEnum


class LabelType(StrEnum):
    """Supported reversal label classes."""

    TOP = "TOP"
    BOTTOM = "BOTTOM"
    NONE = "NONE"


TARGET_COLUMNS: tuple[str, ...] = (
    "label",
    "confidence",
    "future_return",
    "bars_to_target",
    "bars_to_failure",
)
