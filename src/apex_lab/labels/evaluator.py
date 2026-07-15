"""Evaluation helpers for generated label targets."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apex_lab.labels.targets import LabelType


@dataclass(frozen=True)
class LabelStats:
    """Summary metrics for labeled data."""

    total_rows: int
    total_labels: int
    positive_pct: float
    negative_pct: float
    class_balance: dict[str, int]
    average_move: float | None
    median_move: float | None


def evaluate_labels(df: pl.DataFrame) -> LabelStats:
    """Compute aggregate statistics for generated labels.

    Args:
        df: DataFrame with ``label`` and ``future_return`` columns.

    Returns:
        Summary statistics for monitoring label distribution quality.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"label", "future_return"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    total_rows = len(df)

    label_counts = (
        df.group_by("label")
        .len()
        .rename({"len": "count"})
        .select(["label", "count"])
        .to_dicts()
    )
    class_balance = {row["label"]: int(row["count"]) for row in label_counts}

    positive = class_balance.get(LabelType.BOTTOM.value, 0)
    negative = class_balance.get(LabelType.TOP.value, 0)
    total_labels = positive + negative

    labeled_moves = df.filter(pl.col("label") != LabelType.NONE.value).select(pl.col("future_return").abs())

    average_move = None
    median_move = None
    if len(labeled_moves) > 0:
        average_move = float(labeled_moves.select(pl.col("future_return").mean()).item())
        median_move = float(labeled_moves.select(pl.col("future_return").median()).item())

    return LabelStats(
        total_rows=total_rows,
        total_labels=total_labels,
        positive_pct=(positive / total_rows * 100.0) if total_rows else 0.0,
        negative_pct=(negative / total_rows * 100.0) if total_rows else 0.0,
        class_balance=class_balance,
        average_move=average_move,
        median_move=median_move,
    )
