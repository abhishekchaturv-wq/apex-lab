"""Tests for the Feature Importance & Signal Discovery Engine."""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl

from apex_lab.research.signal_discovery.combination import rank_feature_combinations
from apex_lab.research.signal_discovery.engine import run_signal_discovery
from apex_lab.research.signal_discovery.importance import (
    analyze_categorical_features,
    analyze_feature_importance,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _signal_class(value: float) -> str:
    if value >= 1.5:
        return "Strong Bull Move"
    if value >= 0.4:
        return "Bull Move"
    if value <= -1.5:
        return "Strong Bear Move"
    if value <= -0.4:
        return "Bear Move"
    return "Neutral"


def _make_signal_dataset(rows: int = 240) -> pl.DataFrame:
    base_ts = datetime.datetime(2024, 1, 1, 9, 15, 0)

    signal_strength = [((index % 20) - 10) / 5.0 for index in range(rows)]
    noise = [((index * 7) % 13) / 13.0 - 0.5 for index in range(rows)]
    atr_state = ["high" if value > 0.5 else "low" for value in signal_strength]
    rsi_bucket = ["60-70" if value > 0.5 else "40-50" for value in signal_strength]
    opening_range = ["above_or" if index % 3 else "inside_or" for index in range(rows)]
    market_regime = ["above_ema200_high" if index % 4 else "below_ema200_low" for index in range(rows)]
    symbol = ["NIFTY BANK" if index % 2 == 0 else "NIFTY 50" for index in range(rows)]

    future_return_20 = [
        (strength * 1.2) + (0.3 if state == "high" else -0.2) + (0.05 * n)
        for strength, state, n in zip(signal_strength, atr_state, noise, strict=True)
    ]
    future_return_5 = [value * 0.6 for value in future_return_20]
    future_return_10 = [value * 0.8 for value in future_return_20]
    future_return_40 = [value * 1.1 for value in future_return_20]

    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * index) for index in range(rows)],
            "open": [100.0 + index * 0.1 for index in range(rows)],
            "high": [101.0 + index * 0.1 for index in range(rows)],
            "low": [99.0 + index * 0.1 for index in range(rows)],
            "close": [100.5 + index * 0.1 for index in range(rows)],
            "volume": [10_000 + index for index in range(rows)],
            "symbol": symbol,
            "market_regime": market_regime,
            "hour": [9 + (index % 7) for index in range(rows)],
            "day": [1 + (index % 5) for index in range(rows)],
            "month": [1 + (index % 3) for index in range(rows)],
            "quarter": [1 + (index % 4) for index in range(rows)],
            "ema_signal_strength": signal_strength,
            "atr_state": atr_state,
            "rsi_bucket": rsi_bucket,
            "opening_range": opening_range,
            "noise_feature": noise,
            "future_return_5": future_return_5,
            "future_return_10": future_return_10,
            "future_return_20": future_return_20,
            "future_return_40": future_return_40,
            "signal_class": [_signal_class(value) for value in future_return_20],
        }
    )


def test_feature_importance_ranking_prefers_predictive_features() -> None:
    dataset = _make_signal_dataset()
    features = [
        "ema_signal_strength",
        "atr_state",
        "rsi_bucket",
        "opening_range",
        "noise_feature",
        "hour",
        "day",
        "month",
        "quarter",
    ]

    ranked = analyze_feature_importance(dataset, features)

    assert ranked.height == len(features)
    top_features = ranked.head(3).get_column("feature").to_list()
    assert "ema_signal_strength" in top_features
    assert "noise_feature" in ranked.tail(3).get_column("feature").to_list()


def test_categorical_analysis_produces_expected_columns() -> None:
    dataset = _make_signal_dataset()
    features = ["atr_state", "rsi_bucket", "opening_range", "hour", "day", "month", "quarter"]

    report = analyze_categorical_features(dataset, features)

    assert report.height > 0
    assert set(report.columns) == {
        "feature",
        "category",
        "trade_frequency",
        "win_rate",
        "average_return",
        "expectancy",
    }
    assert "atr_state" in set(report.get_column("feature").to_list())


def test_combination_ranking_returns_two_and_three_feature_sets() -> None:
    dataset = _make_signal_dataset()
    features = [
        "ema_signal_strength",
        "atr_state",
        "rsi_bucket",
        "opening_range",
        "noise_feature",
        "hour",
    ]

    importance = analyze_feature_importance(dataset, features)
    combinations = rank_feature_combinations(dataset, importance)

    assert combinations.height > 0
    sizes = set(combinations.get_column("combination_size").to_list())
    assert 2 in sizes
    assert 3 in sizes


def test_engine_generates_required_artifacts_and_summary(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    dataset_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "signal_discovery"
    dataset.write_parquet(dataset_path)

    result = run_signal_discovery(dataset_path, output_dir=output_dir)

    expected = {
        "feature_importance.csv",
        "feature_importance.json",
        "top_combinations.csv",
        "top_combinations.json",
        "stability_report.json",
        "summary.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert result.feature_importance.height > 0
    assert result.top_combinations.height > 0
    assert "top_features" in result.summary
    assert "recommended_pine_features" in result.summary
    assert "features_to_ignore" in result.summary


def test_engine_output_is_deterministic(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    dataset_path = tmp_path / "dataset.parquet"
    dataset.write_parquet(dataset_path)

    run1 = run_signal_discovery(dataset_path, output_dir=tmp_path / "run1")
    run2 = run_signal_discovery(dataset_path, output_dir=tmp_path / "run2")

    assert run1.feature_importance.equals(run2.feature_importance)
    assert run1.top_combinations.equals(run2.top_combinations)
    assert run1.summary == run2.summary


def test_research_lab_signal_discovery_wrapper(tmp_path: Path) -> None:
    module = _load_script_module()
    dataset = _make_signal_dataset()
    dataset_path = tmp_path / "dataset.parquet"
    output_dir = tmp_path / "wrapper_out"
    dataset.write_parquet(dataset_path)

    summary = module.run_signal_discovery(dataset_path=dataset_path, output_dir=output_dir)

    assert (output_dir / "summary.json").exists()
    assert len(summary.get("top_features", [])) > 0
