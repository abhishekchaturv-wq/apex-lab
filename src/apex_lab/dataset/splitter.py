"""Chronological dataset split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SplitConfig:
    """Ratios for train/validation/test splits."""

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        """Validate split ratios."""
        for name, value in {
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        ratio_sum = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")


@dataclass(frozen=True)
class DatasetSplits:
    """Container for chronologically split datasets."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame


def split_dataset(df: pl.DataFrame, config: SplitConfig | None = None) -> DatasetSplits:
    """Split dataset into train/validation/test partitions by row order.

    Args:
        df: Input dataset already sorted chronologically.
        config: Split ratio configuration.

    Returns:
        Chronological dataset splits.
    """
    cfg = config or SplitConfig()
    n_rows = len(df)

    train_end = int(n_rows * cfg.train_ratio)
    validation_end = train_end + int(n_rows * cfg.validation_ratio)

    train = df.slice(0, train_end)
    validation = df.slice(train_end, validation_end - train_end)
    test = df.slice(validation_end, n_rows - validation_end)

    return DatasetSplits(train=train, validation=validation, test=test)
