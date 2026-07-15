"""Reproducible end-to-end dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from apex_lab.dataset.metadata import DatasetMetadata, build_metadata
from apex_lab.dataset.serializer import SerializedDatasetPaths, save_dataset
from apex_lab.dataset.splitter import DatasetSplits, SplitConfig, split_dataset
from apex_lab.dataset.validator import validate_dataset
from apex_lab.features import FeatureEngine
from apex_lab.labels import LabelEngine


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Configuration for reproducible dataset building."""

    symbols: list[str]
    timeframe: str
    feature_version: str
    label_version: str
    git_sha: str | None = None
    feature_groups: list[str] | None = None
    split: SplitConfig = field(default_factory=SplitConfig)
    timestamp_column: str = "timestamp"
    drop_warm_up_rows: bool = True
    output_dir: Path | None = None


@dataclass(frozen=True)
class DatasetBuildResult:
    """Artifacts returned by the dataset builder."""

    dataset: pl.DataFrame
    metadata: DatasetMetadata
    splits: DatasetSplits
    paths: SerializedDatasetPaths | None


class DatasetBuilder:
    """Compose historical data, features, and labels into a reproducible dataset."""

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        label_engine: LabelEngine | None = None,
    ) -> None:
        """Initialize dataset builder dependencies."""
        self.feature_engine = feature_engine or FeatureEngine()
        self.label_engine = label_engine or LabelEngine()

    def build(self, historical_df: pl.DataFrame, config: DatasetBuildConfig) -> DatasetBuildResult:
        """Build a complete reproducible dataset.

        Args:
            historical_df: Raw historical OHLCV data.
            config: Dataset generation configuration.

        Returns:
            Complete dataset, metadata, splits, and optional persisted paths.

        Raises:
            ValueError: If required columns are missing.
        """
        df = _sort_if_timestamp_present(historical_df, config.timestamp_column)

        featured = self.feature_engine.compute(df, groups=config.feature_groups)
        warm_up_rows = self.feature_engine.warm_up_periods(config.feature_groups)
        labeled = self.label_engine.label(featured)

        if config.drop_warm_up_rows and warm_up_rows > 0:
            labeled = labeled.slice(warm_up_rows)
            warm_up_rows = 0

        validate_dataset(
            labeled,
            warm_up_rows=warm_up_rows,
            timestamp_column=config.timestamp_column,
            nullable_columns={"future_return", "bars_to_target", "bars_to_failure"},
        )

        splits = split_dataset(labeled, config=config.split)

        metadata = build_metadata(
            labeled,
            feature_version=config.feature_version,
            label_version=config.label_version,
            symbols=config.symbols,
            timeframe=config.timeframe,
            git_sha=config.git_sha,
            timestamp_column=config.timestamp_column,
        )

        paths = None
        if config.output_dir is not None:
            paths = save_dataset(labeled, splits, metadata, config.output_dir)

        return DatasetBuildResult(dataset=labeled, metadata=metadata, splits=splits, paths=paths)


def build_reproducible_dataset(historical_df: pl.DataFrame, config: DatasetBuildConfig) -> DatasetBuildResult:
    """Single-command API to build a complete reproducible dataset."""
    return DatasetBuilder().build(historical_df, config)


def _sort_if_timestamp_present(df: pl.DataFrame, timestamp_column: str) -> pl.DataFrame:
    """Sort DataFrame chronologically when timestamp column exists."""
    if timestamp_column in df.columns:
        return df.sort(timestamp_column)
    return df
