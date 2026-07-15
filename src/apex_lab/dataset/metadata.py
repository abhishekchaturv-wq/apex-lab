"""Dataset metadata definitions and deterministic metadata builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json

import polars as pl


@dataclass(frozen=True)
class DatasetMetadata:
    """Versioned metadata for a generated ML dataset.

    Attributes:
        dataset_id: Deterministic dataset identifier.
        git_sha: Optional git SHA used at generation time.
        feature_version: Feature pipeline version identifier.
        label_version: Label pipeline version identifier.
        date_range: Date range covered by the dataset.
        symbols: Symbols included in the dataset.
        timeframe: Candle timeframe string.
        number_of_rows: Total number of rows in the dataset.
        class_balance: Label counts by class.
        generation_timestamp: ISO-8601 generation timestamp (UTC).
    """

    dataset_id: str
    git_sha: str | None
    feature_version: str
    label_version: str
    date_range: dict[str, str | None]
    symbols: list[str]
    timeframe: str
    number_of_rows: int
    class_balance: dict[str, int]
    generation_timestamp: str

    def to_dict(self) -> dict[str, object]:
        """Return metadata as a JSON-serializable dictionary."""
        return asdict(self)


def build_metadata(
    df: pl.DataFrame,
    *,
    feature_version: str,
    label_version: str,
    symbols: list[str],
    timeframe: str,
    git_sha: str | None = None,
    timestamp_column: str = "timestamp",
    generation_timestamp: datetime | None = None,
) -> DatasetMetadata:
    """Create dataset metadata for a labeled feature DataFrame.

    Args:
        df: Final dataset DataFrame.
        feature_version: Feature pipeline version identifier.
        label_version: Label pipeline version identifier.
        symbols: Symbols included in the dataset.
        timeframe: Candle timeframe string.
        git_sha: Optional source commit SHA.
        timestamp_column: Timestamp column name.
        generation_timestamp: Optional timestamp override for deterministic tests.

    Returns:
        Fully-populated :class:`DatasetMetadata`.

    Raises:
        ValueError: If label column is missing.
    """
    if "label" not in df.columns:
        raise ValueError("Metadata generation requires a 'label' column")

    date_range = _extract_date_range(df, timestamp_column)
    class_balance = _extract_class_balance(df)
    normalized_symbols = sorted(set(symbols))

    dataset_id = _build_dataset_id(
        feature_version=feature_version,
        label_version=label_version,
        date_range=date_range,
        symbols=normalized_symbols,
        timeframe=timeframe,
        number_of_rows=len(df),
        class_balance=class_balance,
        git_sha=git_sha,
    )

    ts = generation_timestamp or datetime.now(tz=UTC)

    return DatasetMetadata(
        dataset_id=dataset_id,
        git_sha=git_sha,
        feature_version=feature_version,
        label_version=label_version,
        date_range=date_range,
        symbols=normalized_symbols,
        timeframe=timeframe,
        number_of_rows=len(df),
        class_balance=class_balance,
        generation_timestamp=ts.isoformat(),
    )


def _extract_date_range(df: pl.DataFrame, timestamp_column: str) -> dict[str, str | None]:
    """Extract start/end timestamps from the dataset."""
    if timestamp_column not in df.columns or len(df) == 0:
        return {"start": None, "end": None}

    start = df.select(pl.col(timestamp_column).min()).item()
    end = df.select(pl.col(timestamp_column).max()).item()

    return {
        "start": start.isoformat() if hasattr(start, "isoformat") else str(start),
        "end": end.isoformat() if hasattr(end, "isoformat") else str(end),
    }


def _extract_class_balance(df: pl.DataFrame) -> dict[str, int]:
    """Extract class balance from the label column."""
    return {
        row["label"]: int(row["count"])
        for row in (
            df.group_by("label")
            .len()
            .rename({"len": "count"})
            .sort("label")
            .to_dicts()
        )
    }


def _build_dataset_id(
    *,
    feature_version: str,
    label_version: str,
    date_range: dict[str, str | None],
    symbols: list[str],
    timeframe: str,
    number_of_rows: int,
    class_balance: dict[str, int],
    git_sha: str | None,
) -> str:
    """Build a deterministic dataset ID from metadata-defining fields."""
    payload = {
        "feature_version": feature_version,
        "label_version": label_version,
        "date_range": date_range,
        "symbols": symbols,
        "timeframe": timeframe,
        "number_of_rows": number_of_rows,
        "class_balance": class_balance,
        "git_sha": git_sha,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ds_{digest[:16]}"
