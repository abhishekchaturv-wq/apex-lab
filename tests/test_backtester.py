"""Unit tests for the event-driven EMA crossover backtester."""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from apex_lab.research.backtest.backtester import (
    compute_metrics,
    run_backtest,
    write_backtest_reports,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_enriched(closes: list[float]) -> pl.DataFrame:
    """Build a minimal enriched DataFrame with EMA signals."""
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    n = len(closes)
    df = pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)],
            "open": [c - 0.5 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [10_000 + i * 10 for i in range(n)],
        }
    )
    # Use the script's compute_ema_signals to get proper crossover columns
    rl = _load_script_module()
    return rl.compute_ema_signals(df)


# ---------------------------------------------------------------------------
# run_backtest – opposite_crossover exit
# ---------------------------------------------------------------------------


class TestRunBacktestOppositeCrossover:
    """Tests for opposite_crossover exit mode."""

    def test_single_trade_captured(self) -> None:
        """One bullish then one bearish crossover should produce exactly one trade."""
        # 30 flat bars → EMA converges, then a spike forces bullish crossover,
        # then a drop forces bearish crossover.
        closes = [100.0] * 30 + [130.0] * 15 + [90.0] * 25
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="opposite_crossover")

        assert trades.height == 1
        assert trades["exit_reason"][0] == "opposite_crossover"
        assert trades["bars_held"][0] > 0

    def test_no_trades_when_no_crossover(self) -> None:
        """A monotonically increasing series should have no bearish exit → no completed trades."""
        closes = [100.0 + i * 0.5 for i in range(80)]
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="opposite_crossover")
        # There may or may not be an entry, but no bearish exit → no completed trades
        assert trades.height == 0

    def test_no_overlap(self) -> None:
        """Trades must not overlap."""
        closes = [100.0] * 30 + [130.0] * 10 + [90.0] * 10 + [130.0] * 10 + [90.0] * 20
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="opposite_crossover")

        for i in range(len(trades) - 1):
            assert trades["exit_time"][i] <= trades["entry_time"][i + 1]

    def test_return_pct_correct(self) -> None:
        """return_pct should equal (exit_price / entry_price - 1) * 100."""
        closes = [100.0] * 30 + [130.0] * 15 + [90.0] * 25
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="opposite_crossover")

        assert trades.height == 1
        expected = (trades["exit_price"][0] / trades["entry_price"][0] - 1.0) * 100.0
        assert trades["return_pct"][0] == pytest.approx(expected)

    def test_missing_columns_raises(self) -> None:
        """Missing required columns should raise ValueError."""
        df = pl.DataFrame({"timestamp": [1, 2], "close": [100.0, 101.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            run_backtest(df)

    def test_invalid_exit_mode_raises(self) -> None:
        """An unknown exit_mode should raise ValueError."""
        closes = [100.0] * 60
        enriched = _make_enriched(closes)
        with pytest.raises(ValueError, match="Unknown exit_mode"):
            run_backtest(enriched, exit_mode="bad_mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_backtest – fixed_bars exit
# ---------------------------------------------------------------------------


class TestRunBacktestFixedBars:
    """Tests for fixed_bars exit mode."""

    def test_exits_after_n_bars(self) -> None:
        """Each trade must be held for exactly fixed_bars bars."""
        closes = [100.0] * 30 + [130.0] * 20 + [90.0] * 20
        enriched = _make_enriched(closes)
        n_bars = 5
        trades = run_backtest(enriched, exit_mode="fixed_bars", fixed_bars=n_bars)

        assert trades.height >= 1
        for bars_held in trades["bars_held"].to_list():
            assert bars_held == n_bars

    def test_exit_reason_label(self) -> None:
        """exit_reason must be 'fixed_bars'."""
        closes = [100.0] * 30 + [130.0] * 20
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="fixed_bars", fixed_bars=3)

        if trades.height > 0:
            assert all(r == "fixed_bars" for r in trades["exit_reason"].to_list())

    def test_no_overlap_fixed_bars(self) -> None:
        """Trades must not overlap even with fixed_bars exit."""
        closes = [100.0] * 30 + [130.0] * 15 + [90.0] * 25
        enriched = _make_enriched(closes)
        trades = run_backtest(enriched, exit_mode="fixed_bars", fixed_bars=3)

        for i in range(len(trades) - 1):
            assert trades["exit_time"][i] <= trades["entry_time"][i + 1]


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the metrics computation function."""

    def test_empty_trades_returns_none_metrics(self) -> None:
        """Empty trade log should return all-None metrics."""
        empty = pl.DataFrame(
            {
                "entry_time": pl.Series([], dtype=pl.Datetime),
                "exit_time": pl.Series([], dtype=pl.Datetime),
                "entry_price": pl.Series([], dtype=pl.Float64),
                "exit_price": pl.Series([], dtype=pl.Float64),
                "bars_held": pl.Series([], dtype=pl.Int64),
                "return_pct": pl.Series([], dtype=pl.Float64),
                "exit_reason": pl.Series([], dtype=pl.Utf8),
            }
        )
        metrics = compute_metrics(empty)
        assert metrics["number_of_trades"] == 0
        assert metrics["win_rate"] is None
        assert metrics["average_return"] is None
        assert metrics["profit_factor"] is None
        assert metrics["maximum_drawdown"] is None

    def test_all_wins(self) -> None:
        """All positive returns should yield win_rate = 1.0 and no drawdown."""
        returns = [2.0, 3.0, 1.5]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, i + 1) for i in range(3)],
                "exit_time": [datetime.datetime(2024, 1, i + 2) for i in range(3)],
                "entry_price": [100.0, 102.0, 105.0],
                "exit_price": [102.0, 105.0, 106.575],
                "bars_held": [1, 1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover"] * 3,
            }
        )
        metrics = compute_metrics(trades)
        assert metrics["number_of_trades"] == 3
        assert metrics["win_rate"] == pytest.approx(1.0)
        assert metrics["average_return"] == pytest.approx(sum(returns) / 3)
        assert metrics["maximum_drawdown"] == pytest.approx(0.0)

    def test_all_losses(self) -> None:
        """All negative returns should yield win_rate = 0.0 and None profit_factor."""
        returns = [-1.0, -2.0, -0.5]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, i + 1) for i in range(3)],
                "exit_time": [datetime.datetime(2024, 1, i + 2) for i in range(3)],
                "entry_price": [100.0, 99.0, 97.0],
                "exit_price": [99.0, 97.02, 96.515],
                "bars_held": [1, 1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover"] * 3,
            }
        )
        metrics = compute_metrics(trades)
        assert metrics["win_rate"] == pytest.approx(0.0)
        assert metrics["profit_factor"] == pytest.approx(0.0)  # 0 gross wins / positive losses

    def test_profit_factor_calculation(self) -> None:
        """profit_factor should equal gross wins / |gross losses|."""
        returns = [4.0, -2.0]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)],
                "exit_time": [datetime.datetime(2024, 1, 2), datetime.datetime(2024, 1, 3)],
                "entry_price": [100.0, 104.0],
                "exit_price": [104.0, 101.92],
                "bars_held": [1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover", "opposite_crossover"],
            }
        )
        metrics = compute_metrics(trades)
        assert metrics["profit_factor"] == pytest.approx(4.0 / 2.0)

    def test_maximum_drawdown_positive(self) -> None:
        """Equity-curve drawdown should be positive and correct."""
        # Cumulative returns: 3, 1 (peak 3, trough 1) → max drawdown = 2
        returns = [3.0, -2.0, 2.0]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, i + 1) for i in range(3)],
                "exit_time": [datetime.datetime(2024, 1, i + 2) for i in range(3)],
                "entry_price": [100.0] * 3,
                "exit_price": [103.0, 100.94, 102.94],
                "bars_held": [1, 1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover"] * 3,
            }
        )
        metrics = compute_metrics(trades)
        assert metrics["maximum_drawdown"] == pytest.approx(2.0)

    def test_expectancy_formula(self) -> None:
        """expectancy = win_rate * avg_win + loss_rate * avg_loss."""
        returns = [4.0, -2.0]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)],
                "exit_time": [datetime.datetime(2024, 1, 2), datetime.datetime(2024, 1, 3)],
                "entry_price": [100.0, 104.0],
                "exit_price": [104.0, 101.92],
                "bars_held": [1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover", "opposite_crossover"],
            }
        )
        metrics = compute_metrics(trades)
        expected_exp = 0.5 * 4.0 + 0.5 * (-2.0)
        assert metrics["expectancy"] == pytest.approx(expected_exp)


# ---------------------------------------------------------------------------
# write_backtest_reports
# ---------------------------------------------------------------------------


class TestWriteBacktestReports:
    """Tests for persistence of backtest outputs."""

    def test_writes_csv_and_json(self, tmp_path: Path) -> None:
        """Both trades.csv and summary.json should be created."""
        returns = [2.0, -1.0, 3.0]
        trades = pl.DataFrame(
            {
                "entry_time": [datetime.datetime(2024, 1, i + 1) for i in range(3)],
                "exit_time": [datetime.datetime(2024, 1, i + 2) for i in range(3)],
                "entry_price": [100.0, 102.0, 101.0],
                "exit_price": [102.0, 100.98, 104.03],
                "bars_held": [1, 1, 1],
                "return_pct": returns,
                "exit_reason": ["opposite_crossover"] * 3,
            }
        )
        metrics = compute_metrics(trades)
        trades_path = tmp_path / "trades.csv"
        summary_path = tmp_path / "summary.json"

        write_backtest_reports(trades, metrics, trades_path, summary_path)

        assert trades_path.exists()
        assert summary_path.exists()

        persisted_trades = pl.read_csv(trades_path, try_parse_dates=True)
        assert persisted_trades.height == 3

        persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert persisted_summary["number_of_trades"] == 3


# ---------------------------------------------------------------------------
# CLI integration – run_event_backtest via script module
# ---------------------------------------------------------------------------


class TestRunEventBacktest:
    """Integration tests for run_event_backtest in the CLI script."""

    def test_event_mode_produces_reports(self, tmp_path: Path) -> None:
        """run_event_backtest should write trades and summary to disk."""
        rl = _load_script_module()
        closes = [100.0] * 30 + [130.0] * 15 + [90.0] * 25
        base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
        df = pl.DataFrame(
            {
                "timestamp": [
                    base_ts + datetime.timedelta(minutes=30 * i) for i in range(len(closes))
                ],
                "open": [c - 0.5 for c in closes],
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [10_000 + i * 10 for i in range(len(closes))],
            }
        )
        data_path = tmp_path / "input.parquet"
        trades_path = tmp_path / "trades.csv"
        summary_path = tmp_path / "summary.json"
        df.write_parquet(data_path)

        trades, metrics = rl.run_event_backtest(
            data_path=data_path,
            exit_mode="opposite_crossover",
            trades_output=trades_path,
            summary_output=summary_path,
        )

        assert trades_path.exists()
        assert summary_path.exists()
        assert trades.height >= 1
        assert metrics["number_of_trades"] >= 1

    def test_fixed_bars_mode_produces_reports(self, tmp_path: Path) -> None:
        """run_event_backtest with fixed_bars should produce valid reports."""
        rl = _load_script_module()
        closes = [100.0] * 30 + [130.0] * 20 + [90.0] * 20
        base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
        df = pl.DataFrame(
            {
                "timestamp": [
                    base_ts + datetime.timedelta(minutes=30 * i) for i in range(len(closes))
                ],
                "open": [c - 0.5 for c in closes],
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [10_000 + i * 10 for i in range(len(closes))],
            }
        )
        data_path = tmp_path / "input.parquet"
        trades_path = tmp_path / "trades.csv"
        summary_path = tmp_path / "summary.json"
        df.write_parquet(data_path)

        trades, metrics = rl.run_event_backtest(
            data_path=data_path,
            exit_mode="fixed_bars",
            fixed_bars=5,
            trades_output=trades_path,
            summary_output=summary_path,
        )

        assert trades_path.exists()
        assert metrics["number_of_trades"] >= 0

    def test_forward_return_mode_still_works(self, tmp_path: Path) -> None:
        """The original run_research_lab must still function correctly."""
        rl = _load_script_module()
        closes = ([100.0] * 30) + ([130.0] * 15) + ([90.0] * 25)
        base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
        df = pl.DataFrame(
            {
                "timestamp": [
                    base_ts + datetime.timedelta(minutes=30 * i) for i in range(len(closes))
                ],
                "open": [c - 0.5 for c in closes],
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [10_000 + i * 10 for i in range(len(closes))],
            }
        )
        data_path = tmp_path / "input.parquet"
        csv_path = tmp_path / "ema.csv"
        json_path = tmp_path / "ema.json"
        df.write_parquet(data_path)

        bullish_returns, summary = rl.run_research_lab(data_path, csv_path, json_path)

        assert csv_path.exists()
        assert json_path.exists()
        assert bullish_returns.height == 1
        assert summary["bullish_crossovers"] == 1
