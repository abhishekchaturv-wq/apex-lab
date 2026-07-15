"""Tests for baseline predictive model pipeline."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

import apex_lab.models.trainer as trainer_module
from apex_lab.models import (
    evaluate_binary_classifier,
    load_model,
    predict_from_artifact,
    predict_probabilities,
    train_baseline_model,
)


@pytest.fixture()
def binary_dataset() -> pl.DataFrame:
    """Generate a deterministic binary classification dataset."""
    x, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )

    payload = {f"feature_{idx}": x[:, idx] for idx in range(x.shape[1])}
    payload["target"] = y
    return pl.DataFrame(payload)


def test_train_baseline_model_writes_expected_outputs(binary_dataset: pl.DataFrame, tmp_path) -> None:
    """Training should produce model, metrics, importances, and probabilities artifacts."""
    result = train_baseline_model(
        binary_dataset,
        target_column="target",
        model_name="logistic_regression",
        output_dir=tmp_path,
    )

    assert result.paths is not None
    assert result.paths.model_artifact.exists()
    assert result.paths.metrics_json.exists()
    assert result.paths.feature_importance_csv.exists()
    assert result.paths.prediction_probabilities_csv.exists()

    payload = json.loads(result.paths.metrics_json.read_text(encoding="utf-8"))
    assert {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "calibration",
    }.issubset(payload.keys())


def test_inference_probability_output(binary_dataset: pl.DataFrame, tmp_path) -> None:
    """Inference module should load model and return valid probability scores."""
    result = train_baseline_model(
        binary_dataset,
        target_column="target",
        model_name="random_forest",
        output_dir=tmp_path,
    )

    assert result.paths is not None

    subset = binary_dataset.select(result.feature_names).head(25)
    probs = predict_from_artifact(result.paths.model_artifact, subset, feature_names=result.feature_names)

    assert probs.shape == (25,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_metrics_cover_expected_binary_suite() -> None:
    """Metrics helper should compute all expected evaluation keys."""
    y_true = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    y_pred = np.array([0, 1, 0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.2, 0.8, 0.4, 0.3, 0.7, 0.6], dtype=float)

    metrics = evaluate_binary_classifier(y_true, y_pred, y_prob)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]
    assert metrics["calibration"]["brier_score"] is not None


def test_model_serialization_roundtrip(binary_dataset: pl.DataFrame, tmp_path) -> None:
    """Serialized model should preserve probability predictions."""
    result = train_baseline_model(
        binary_dataset,
        target_column="target",
        model_name="gradient_boosting",
        output_dir=tmp_path,
    )

    assert result.paths is not None

    features = binary_dataset.select(result.feature_names).head(30)
    loaded_model = load_model(result.paths.model_artifact)

    original_probs = predict_probabilities(result.model, features, feature_names=result.feature_names)
    loaded_probs = predict_probabilities(loaded_model, features, feature_names=result.feature_names)

    assert np.allclose(original_probs, loaded_probs)


def test_train_baseline_model_skips_random_split_when_test_df_is_supplied(
    binary_dataset: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit test frames should bypass sklearn's random train/test split."""

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("train_test_split should not be called when test_df is supplied")

    monkeypatch.setattr(trainer_module, "train_test_split", fail_if_called)

    train_df = binary_dataset.head(300)
    test_df = binary_dataset.tail(200)

    result = train_baseline_model(
        train_df,
        target_column="target",
        model_name="logistic_regression",
        test_df=test_df,
    )

    assert len(result.prediction_probabilities) == len(test_df)
