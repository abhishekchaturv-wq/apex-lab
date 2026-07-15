"""Tests for reproducible dataset builder pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from apex_lab.dataset import (
    DatasetBuildConfig,
    DatasetBuilder,
    SplitConfig,
    build_metadata,
    build_reproducible_dataset,
    collect_validation_errors,
    save_dataset,
    split_dataset,
)


def test_metadata_is_deterministic_for_same_inputs(small_ohlcv: pl.DataFrame) -> None:
    """Dataset ID should be deterministic for identical metadata inputs."""
    labeled = small_ohlcv.with_columns(pl.lit("NONE").alias("label"))

    first = build_metadata(
        labeled,
        feature_version="feat-v1",
        label_version="label-v1",
        symbols=["NIFTY"],
        timeframe="5m",
        generation_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = build_metadata(
        labeled,
        feature_version="feat-v1",
        label_version="label-v1",
        symbols=["NIFTY"],
        timeframe="5m",
        generation_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert first.dataset_id == second.dataset_id


def test_splitter_splits_chronologically(small_ohlcv: pl.DataFrame) -> None:
    """Chronological split should preserve order and full row coverage."""
    splits = split_dataset(small_ohlcv, config=SplitConfig(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2))

    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(small_ohlcv)

    train_last = splits.train["timestamp"][-1]
    validation_first = splits.validation["timestamp"][0]
    validation_last = splits.validation["timestamp"][-1]
    test_first = splits.test["timestamp"][0]

    assert train_last <= validation_first
    assert validation_last <= test_first


def test_validator_detects_duplicates_and_missing_labels(small_ohlcv: pl.DataFrame) -> None:
    """Validator should report duplicate timestamps and missing labels."""
    bad_df = (
        small_ohlcv
        .with_columns(pl.lit("NONE").alias("label"))
        .with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("label"))
            .alias("label")
        )
        .with_columns(
            pl.when(pl.arange(0, pl.len()) == 1)
            .then(pl.col("timestamp").first())
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
    )

    errors = collect_validation_errors(bad_df)

    assert any("duplicate timestamps" in error for error in errors)
    assert any("missing labels" in error for error in errors)


def test_serializer_persists_parquet_and_json(tmp_path: Path, small_ohlcv: pl.DataFrame) -> None:
    """Serializer should write dataset/splits parquet and metadata json."""
    dataset = small_ohlcv.with_columns(pl.lit("NONE").alias("label"))
    splits = split_dataset(dataset)
    metadata = build_metadata(
        dataset,
        feature_version="feat-v1",
        label_version="label-v1",
        symbols=["NIFTY"],
        timeframe="5m",
    )

    paths = save_dataset(dataset, splits, metadata, Path(tmp_path))

    assert paths.dataset_parquet.exists()
    assert paths.train_parquet.exists()
    assert paths.validation_parquet.exists()
    assert paths.test_parquet.exists()
    assert paths.metadata_json.exists()

    payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == metadata.dataset_id


def test_builder_builds_complete_dataset_with_one_call(small_ohlcv: pl.DataFrame, tmp_path: Path) -> None:
    """Single build call should return dataset, metadata, splits, and persisted files."""
    config = DatasetBuildConfig(
        symbols=["NIFTY"],
        timeframe="5m",
        feature_version="features-v1",
        label_version="labels-v1",
        output_dir=Path(tmp_path),
    )

    result = build_reproducible_dataset(small_ohlcv, config)

    assert len(result.dataset) > 0
    assert result.metadata.number_of_rows == len(result.dataset)
    assert result.metadata.dataset_id.startswith("ds_")
    assert len(result.splits.train) + len(result.splits.validation) + len(result.splits.test) == len(result.dataset)
    assert result.paths is not None
    assert result.paths.dataset_parquet.exists()


def test_builder_respects_schema_validation(small_ohlcv: pl.DataFrame) -> None:
    """Builder output should be schema-consistent with its own schema signature."""
    builder = DatasetBuilder()
    config = DatasetBuildConfig(
        symbols=["NIFTY"],
        timeframe="5m",
        feature_version="features-v1",
        label_version="labels-v1",
    )

    result = builder.build(small_ohlcv, config)
    errors = collect_validation_errors(
        result.dataset,
        expected_schema=result.dataset.schema,
        nullable_columns={"future_return", "bars_to_target", "bars_to_failure"},
    )

    assert errors == []
