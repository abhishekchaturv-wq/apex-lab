"""Tests for the reproducible baseline analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from apex_lab.models.experiment import run_baseline_predictive_analysis


def test_baseline_experiment_exports_required_reports(tmp_path: Path) -> None:
    """Baseline experiment should generate all required report artifacts."""
    input_dir = tmp_path / "historical"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_symbol_data(input_dir / "NIFTY.parquet", offset=0.0)
    _write_symbol_data(input_dir / "BANKNIFTY.parquet", offset=15.0)

    reports_dir = tmp_path / "reports"
    result = run_baseline_predictive_analysis(
        input_dir=input_dir,
        output_dir=reports_dir,
        timeframe="30m",
        feature_version="features-v1",
        label_version="labels-v1",
    )

    assert result.reports_dir == reports_dir
    assert result.dataset_id.startswith("ds_")
    assert result.models_trained == 6
    assert result.symbols_processed == ["BANKNIFTY", "NIFTY"]

    expected_files = {
        "metrics.json",
        "feature_importance.csv",
        "threshold_analysis.csv",
        "confusion_matrix.csv",
        "calibration.csv",
        "probability_distribution.csv",
        "feature_correlation.csv",
        "experiment.json",
    }
    assert expected_files.issubset({path.name for path in reports_dir.iterdir()})

    metrics_payload = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["dataset_id"] == result.dataset_id
    assert len(metrics_payload["runs"]) == 6

    experiment_payload = json.loads((reports_dir / "experiment.json").read_text(encoding="utf-8"))
    assert experiment_payload["dataset_id"] == result.dataset_id
    assert sorted(experiment_payload["model"]) == [
        "gradient_boosting",
        "logistic_regression",
        "random_forest",
    ]


def _write_symbol_data(path: Path, *, offset: float) -> None:
    """Write deterministic oscillating OHLCV data for one symbol."""
    points = 1_200
    base = np.linspace(0.0, 40.0 * np.pi, points)
    close = 100.0 + offset + (8.0 * np.sin(base)) + (1.5 * np.sin(base / 2.0))
    open_ = close + (0.2 * np.cos(base))
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = 75_000 + (1_500 * (1.0 + np.sin(base / 3.0)))

    frame = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 9, 15),
                end=pl.datetime(2024, 1, 26, 15, 30),
                interval="30m",
                eager=True,
            )[:points],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    frame.write_parquet(path)
