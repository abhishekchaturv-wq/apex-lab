"""Artifact writer for signal discovery datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_OUTPUT_DIR = Path("reports/lab/signal_dataset")


def write_dataset_artifacts(
    df: pl.DataFrame,
    output_dir: Path,
    schema_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, Path]:
    """Write parquet + JSON artifact set and return file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.parquet"
    schema_path = output_dir / "schema.json"
    summary_path = output_dir / "summary.json"
    feature_list_path = output_dir / "feature_list.json"

    df.write_parquet(dataset_path)
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    feature_list_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    return {
        "dataset": dataset_path,
        "schema": schema_path,
        "summary": summary_path,
        "feature_list": feature_list_path,
    }
