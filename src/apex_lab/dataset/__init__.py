"""Dataset generation package for reproducible ML datasets."""

from apex_lab.dataset.builder import (
    DatasetBuildConfig,
    DatasetBuilder,
    DatasetBuildResult,
    build_reproducible_dataset,
)
from apex_lab.dataset.metadata import DatasetMetadata, build_metadata
from apex_lab.dataset.serializer import SerializedDatasetPaths, save_dataset
from apex_lab.dataset.splitter import DatasetSplits, SplitConfig, split_dataset
from apex_lab.dataset.validator import collect_validation_errors, validate_dataset

__all__ = [
    "DatasetBuildConfig",
    "DatasetBuildResult",
    "DatasetBuilder",
    "DatasetMetadata",
    "DatasetSplits",
    "SerializedDatasetPaths",
    "SplitConfig",
    "build_metadata",
    "build_reproducible_dataset",
    "collect_validation_errors",
    "save_dataset",
    "split_dataset",
    "validate_dataset",
]
