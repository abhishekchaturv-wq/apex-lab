"""Validation helpers for reproducible ML datasets."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def validate_dataset(
    df: pl.DataFrame,
    *,
    warm_up_rows: int = 0,
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    nullable_columns: Iterable[str] | None = None,
    expected_schema: dict[str, pl.DataType] | None = None,
) -> None:
    """Validate dataset quality constraints.

    Args:
        df: Dataset to validate.
        warm_up_rows: Number of initial rows exempt from NaN/null checks.
        timestamp_column: Timestamp column name.
        label_column: Label column name.
        nullable_columns: Columns allowed to contain null/NaN after warm-up.
        expected_schema: Optional schema to enforce.

    Raises:
        ValueError: If any validation rule fails.
    """
    errors = collect_validation_errors(
        df,
        warm_up_rows=warm_up_rows,
        timestamp_column=timestamp_column,
        label_column=label_column,
        nullable_columns=nullable_columns,
        expected_schema=expected_schema,
    )
    if errors:
        raise ValueError("Dataset validation failed: " + " | ".join(errors))


def collect_validation_errors(
    df: pl.DataFrame,
    *,
    warm_up_rows: int = 0,
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    nullable_columns: Iterable[str] | None = None,
    expected_schema: dict[str, pl.DataType] | None = None,
) -> list[str]:
    """Collect validation errors without raising."""
    errors: list[str] = []
    allowed_nulls = set(nullable_columns or [])

    if timestamp_column not in df.columns:
        errors.append(f"missing timestamp column '{timestamp_column}'")
    else:
        duplicate_count = len(df) - df.select(pl.col(timestamp_column).n_unique()).item()
        if duplicate_count > 0:
            errors.append(f"duplicate timestamps found: {duplicate_count}")

    if label_column not in df.columns:
        errors.append(f"missing label column '{label_column}'")
    else:
        missing_labels = df.select(pl.col(label_column).is_null().sum()).item()
        if missing_labels > 0:
            errors.append(f"missing labels found: {missing_labels}")

    if expected_schema is not None and df.schema != expected_schema:
        errors.append("inconsistent schema detected")

    if warm_up_rows < 0:
        errors.append("warm_up_rows must be >= 0")

    trimmed = df.slice(offset=warm_up_rows)
    if len(trimmed) > 0:
        for column, dtype in trimmed.schema.items():
            if column in allowed_nulls:
                continue

            null_count = trimmed.select(pl.col(column).is_null().sum()).item()
            if null_count > 0:
                errors.append(f"null values found outside warm-up in '{column}': {null_count}")

            if dtype in (pl.Float32, pl.Float64):
                nan_count = trimmed.select(pl.col(column).is_nan().sum()).item()
                if nan_count > 0:
                    errors.append(f"nan values found outside warm-up in '{column}': {nan_count}")

    return errors
