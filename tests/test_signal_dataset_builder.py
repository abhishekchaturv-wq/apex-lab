"""Tests for signal discovery dataset builder."""

from __future__ import annotations

import datetime
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import polars as pl

from apex_lab.research.signal_dataset.builder import SignalDatasetBuilder, SignalDatasetConfig
from apex_lab.research.signal_dataset.labels import SignalLabelConfig, append_signal_classes

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_ohlcv(n: int = 900) -> pl.DataFrame:
    base_ts = datetime.datetime(2021, 1, 1, 9, 15, 0)
    closes = [
        41_000.0 + i * 0.8 + 280.0 * math.sin(i / 12.0) + 90.0 * math.sin(i / 31.0)
        for i in range(n)
    ]
    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)],
            "open": [close - 12.0 for close in closes],
            "high": [close + 18.0 for close in closes],
            "low": [close - 22.0 for close in closes],
            "close": closes,
            "volume": [80_000 + int(15_000 * (1.0 + math.sin(i / 17.0))) for i in range(n)],
        }
    )


def test_label_generation_columns_and_classes() -> None:
    config = SignalLabelConfig(horizons=(5, 10, 20, 40))
    labeled = append_signal_classes(_make_ohlcv(120), config)

    expected = {
        "future_return_5",
        "future_return_10",
        "future_return_20",
        "future_return_40",
        "future_high_return",
        "future_low_return",
        "maximum_favorable_excursion",
        "maximum_adverse_excursion",
        "direction",
        "signal_class",
        "label_strong_bull_move",
        "label_bull_move",
        "label_neutral",
        "label_bear_move",
        "label_strong_bear_move",
    }
    assert expected.issubset(set(labeled.columns))

    classes = set(labeled.get_column("signal_class").drop_nulls().unique().to_list())
    assert classes.issubset(
        {"Strong Bull Move", "Bull Move", "Neutral", "Bear Move", "Strong Bear Move"}
    )


def test_signal_dataset_builder_generates_required_artifacts(tmp_path: Path) -> None:
    builder = SignalDatasetBuilder()
    result = builder.build(
        _make_ohlcv(),
        SignalDatasetConfig(
            symbol="NIFTY BANK",
            interval="30minute",
            output_dir=tmp_path,
            session_id="test-session",
            generation_timestamp="2026-01-01T00:00:00+00:00",
        ),
    )

    assert result.dataset.height > 0
    assert (tmp_path / "dataset.parquet").exists()
    assert (tmp_path / "schema.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "feature_list.json").exists()

    assert "symbol" in result.dataset.columns
    assert "interval" in result.dataset.columns
    assert "session_id" in result.dataset.columns
    assert "weekday" in result.dataset.columns
    assert "market_regime" in result.dataset.columns
    assert "alpha_score" in result.dataset.columns


def test_summary_and_schema_payload_fields(tmp_path: Path) -> None:
    result = SignalDatasetBuilder().build(
        _make_ohlcv(),
        SignalDatasetConfig(output_dir=tmp_path, generation_timestamp="2026-01-01T00:00:00+00:00"),
    )

    summary = result.summary
    assert summary["row_count"] == result.dataset.height
    assert summary["feature_count"] == len(result.feature_columns)
    assert summary["label_count"] == len(result.label_columns)
    assert summary["symbol"] == "UNKNOWN"
    assert summary["interval"] == "UNKNOWN"
    assert "start" in summary["date_range"] and "end" in summary["date_range"]

    schema = result.schema
    assert "columns" in schema
    assert "feature_columns" in schema
    assert "label_columns" in schema
    assert "metadata_columns" in schema


def test_duplicate_timestamp_handling_drops_duplicates(tmp_path: Path) -> None:
    df = _make_ohlcv(300)
    duplicate_row = df.slice(100, 1)
    with_duplicate = pl.concat([df, duplicate_row], how="vertical")

    result = SignalDatasetBuilder().build(
        with_duplicate,
        SignalDatasetConfig(output_dir=tmp_path, generation_timestamp="2026-01-01T00:00:00+00:00"),
    )

    assert result.summary["duplicate_timestamps"] == 1
    assert result.dataset.get_column("timestamp").n_unique() == result.dataset.height


def test_deterministic_output_with_fixed_metadata(tmp_path: Path) -> None:
    config = SignalDatasetConfig(
        symbol="NIFTY BANK",
        interval="30minute",
        session_id="stable-session",
        generation_timestamp="2026-01-01T00:00:00+00:00",
    )
    df = _make_ohlcv()

    run1 = SignalDatasetBuilder().build(df, SignalDatasetConfig(**{**config.__dict__, "output_dir": tmp_path / "r1"}))
    run2 = SignalDatasetBuilder().build(df, SignalDatasetConfig(**{**config.__dict__, "output_dir": tmp_path / "r2"}))

    assert run1.dataset.equals(run2.dataset)
    assert run1.summary == run2.summary
    assert run1.feature_columns == run2.feature_columns


def test_research_lab_signal_dataset_mode_wrapper(tmp_path: Path) -> None:
    module = _load_script_module()
    data_path = tmp_path / "input.parquet"
    _make_ohlcv().write_parquet(data_path)

    dataset, summary, schema, feature_columns = module.run_signal_dataset(
        data_path=data_path,
        output_dir=tmp_path / "signal_dataset",
        symbol="NIFTY BANK",
        interval="30minute",
    )

    assert dataset.height > 0
    assert summary["symbol"] == "NIFTY BANK"
    assert summary["interval"] == "30minute"
    assert summary["feature_count"] == len(feature_columns)
    assert "columns" in schema
