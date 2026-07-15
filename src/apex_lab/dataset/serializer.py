"""Persistence helpers for reproducible datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import polars as pl

from apex_lab.dataset.metadata import DatasetMetadata
from apex_lab.dataset.splitter import DatasetSplits


@dataclass(frozen=True)
class SerializedDatasetPaths:
    """Filesystem paths for serialized dataset artifacts."""

    root_dir: Path
    dataset_parquet: Path
    train_parquet: Path
    validation_parquet: Path
    test_parquet: Path
    metadata_json: Path


def save_dataset(
    dataset: pl.DataFrame,
    splits: DatasetSplits,
    metadata: DatasetMetadata,
    output_dir: Path,
) -> SerializedDatasetPaths:
    """Persist dataset artifacts to disk.

    Args:
        dataset: Full dataset DataFrame.
        splits: Train/validation/test splits.
        metadata: Dataset metadata object.
        output_dir: Root directory where artifacts are stored.

    Returns:
        Paths to serialized artifacts.
    """
    dataset_dir = output_dir / metadata.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "dataset.parquet"
    train_path = dataset_dir / "train.parquet"
    validation_path = dataset_dir / "validation.parquet"
    test_path = dataset_dir / "test.parquet"
    metadata_path = dataset_dir / "metadata.json"

    dataset.write_parquet(dataset_path)
    splits.train.write_parquet(train_path)
    splits.validation.write_parquet(validation_path)
    splits.test.write_parquet(test_path)

    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    return SerializedDatasetPaths(
        root_dir=dataset_dir,
        dataset_parquet=dataset_path,
        train_parquet=train_path,
        validation_parquet=validation_path,
        test_parquet=test_path,
        metadata_json=metadata_path,
    )
