"""Tests for chronological baseline experiment evaluation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from apex_lab.dataset import DatasetBuildConfig, DatasetBuilder
from apex_lab.models.experiment import DEFAULT_PURGE_GAP_BARS, run_baseline_predictive_analysis
from apex_lab.models.trainer import TrainingResult


def test_baseline_experiment_uses_chronological_purged_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experiment should pass chronological train/test frames with a purge gap."""
    input_dir = tmp_path / "historical"
    input_dir.mkdir(parents=True, exist_ok=True)
    data_path = input_dir / "NIFTY.parquet"
    raw_frame = _write_symbol_data(data_path, offset=0.0, points=600)

    captured_calls: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    def fake_train_baseline_model(
        dataset: pl.DataFrame,
        *,
        target_column: str,
        model_name: str = "logistic_regression",
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: Path | None = None,
        test_df: pl.DataFrame | None = None,
    ) -> TrainingResult:
        del target_column, test_size, random_state, output_dir
        assert test_df is not None
        captured_calls.append((dataset.clone(), test_df.clone()))
        return TrainingResult(
            model_name=model_name,
            model=object(),
            feature_names=["atr_14"],
            class_mapping={"0": 0, "1": 1},
            metrics={
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "roc_auc": 1.0,
                "pr_auc": 1.0,
                "confusion_matrix": [[1, 0], [0, 1]],
                "calibration": {
                    "brier_score": 0.0,
                    "curve": {
                        "mean_predicted_probability": [0.1, 0.9],
                        "fraction_of_positives": [0.0, 1.0],
                    },
                },
            },
            feature_importance=pl.DataFrame(
                {
                    "feature": ["atr_14"],
                    "model_importance": [0.1],
                    "permutation_importance_mean": [0.2],
                    "permutation_importance_std": [0.0],
                }
            ),
            prediction_probabilities=pl.DataFrame(
                {
                    "y_true": [0, 1],
                    "y_pred": [0, 1],
                    "probability": [0.1, 0.9],
                }
            ),
            paths=None,
        )

    monkeypatch.setattr("apex_lab.models.experiment.train_baseline_model", fake_train_baseline_model)

    run_baseline_predictive_analysis(input_dir=input_dir, output_dir=tmp_path / "reports")

    assert len(captured_calls) == 6

    train_frame, test_frame = captured_calls[0]
    train_timestamps = train_frame["timestamp"].to_list()
    test_timestamps = test_frame["timestamp"].to_list()

    assert train_timestamps == sorted(train_timestamps)
    assert test_timestamps == sorted(test_timestamps)
    assert train_timestamps[-1] < test_timestamps[0]
    assert test_timestamps[0] - train_timestamps[-1] >= timedelta(minutes=30 * DEFAULT_PURGE_GAP_BARS)

    builder = DatasetBuilder()
    built = builder.build(
        raw_frame,
        DatasetBuildConfig(
            symbols=["NIFTY"],
            timeframe="30m",
            feature_version="features-v1",
            label_version="labels-v1",
        ),
    )
    assert len(train_frame) == len(built.splits.train) - DEFAULT_PURGE_GAP_BARS
    assert len(test_frame) == len(built.splits.test)


def _write_symbol_data(path: Path, *, offset: float, points: int) -> pl.DataFrame:
    """Write deterministic oscillating OHLCV data for one symbol."""
    base = np.linspace(0.0, 20.0 * np.pi, points)
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
    return frame
