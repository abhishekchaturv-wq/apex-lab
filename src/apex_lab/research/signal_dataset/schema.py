"""Schema helpers for the signal discovery dataset."""

from __future__ import annotations

from typing import Any

import polars as pl


def build_schema_payload(
    df: pl.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    metadata_columns: list[str],
) -> dict[str, Any]:
    """Build a serializable schema payload for dataset artifacts."""
    return {
        "columns": [
            {
                "name": name,
                "dtype": str(df.schema[name]),
                "role": _role_for_column(name, feature_columns, label_columns, metadata_columns),
            }
            for name in df.columns
        ],
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "metadata_columns": metadata_columns,
    }


def _role_for_column(
    name: str,
    feature_columns: list[str],
    label_columns: list[str],
    metadata_columns: list[str],
) -> str:
    if name in label_columns:
        return "label"
    if name in metadata_columns:
        return "metadata"
    if name in feature_columns:
        return "feature"
    return "raw"
