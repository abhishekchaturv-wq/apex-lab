"""Tests for the research lab EMA crossover script."""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("research_lab_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_ohlcv(closes: list[float]) -> pl.DataFrame:
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * index) for index in range(len(closes))],
            "open": [close - 0.5 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [10_000 + index * 10 for index in range(len(closes))],
        }
    )


def test_run_research_lab_writes_expected_reports(tmp_path: Path) -> None:
    """The script should export bullish returns and summary metrics."""
    research_lab = _load_script_module()
    closes = ([100.0] * 30) + ([130.0] * 15) + ([90.0] * 25)
    source_df = _make_ohlcv(closes)

    data_path = tmp_path / "input.parquet"
    csv_output = tmp_path / "reports" / "ema_cross_returns.csv"
    json_output = tmp_path / "reports" / "ema_cross_summary.json"
    source_df.write_parquet(data_path)

    bullish_returns, summary = research_lab.run_research_lab(data_path, csv_output, json_output)

    assert csv_output.exists()
    assert json_output.exists()
    assert bullish_returns.height == 1
    assert summary["bullish_crossovers"] == 1
    assert summary["bearish_crossovers"] == 1

    signal_timestamp = bullish_returns[0, "timestamp"]
    source_index = source_df.get_column("timestamp").to_list().index(signal_timestamp)

    for horizon in research_lab.FORWARD_RETURN_HORIZONS:
        expected_return = ((closes[source_index + horizon] / closes[source_index]) - 1.0) * 100.0
        actual_return = bullish_returns[0, f"forward_return_{horizon}"]
        assert actual_return == pytest.approx(expected_return)

        metrics = summary["forward_returns"][str(horizon)]
        assert metrics["num_signals"] == 1
        assert metrics["mean_return"] == pytest.approx(expected_return)
        assert metrics["median_return"] == pytest.approx(expected_return)
        assert metrics["standard_deviation"] == pytest.approx(0.0)
        assert metrics["maximum_gain"] == pytest.approx(expected_return)
        assert metrics["maximum_loss"] == pytest.approx(expected_return)

    persisted_summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert persisted_summary == summary

    persisted_returns = pl.read_csv(csv_output, try_parse_dates=True)
    assert persisted_returns.height == 1
    assert persisted_returns["signal"].to_list() == ["bullish_crossover"]


def test_run_research_lab_handles_no_bullish_crossovers(tmp_path: Path) -> None:
    """The script should emit empty reports when no bullish crossover exists."""
    research_lab = _load_script_module()
    closes = [150.0 - index for index in range(70)]
    source_df = _make_ohlcv(closes)

    data_path = tmp_path / "input.parquet"
    csv_output = tmp_path / "reports" / "ema_cross_returns.csv"
    json_output = tmp_path / "reports" / "ema_cross_summary.json"
    source_df.write_parquet(data_path)

    bullish_returns, summary = research_lab.run_research_lab(data_path, csv_output, json_output)

    assert bullish_returns.is_empty()
    assert summary["bullish_crossovers"] == 0

    for horizon in research_lab.FORWARD_RETURN_HORIZONS:
        metrics = summary["forward_returns"][str(horizon)]
        assert metrics["num_signals"] == 0
        assert metrics["win_rate"] is None
        assert metrics["mean_return"] is None
        assert metrics["median_return"] is None
        assert metrics["standard_deviation"] is None
        assert metrics["maximum_gain"] is None
        assert metrics["maximum_loss"] is None

    persisted_returns = pl.read_csv(csv_output, try_parse_dates=True)
    assert persisted_returns.is_empty()
