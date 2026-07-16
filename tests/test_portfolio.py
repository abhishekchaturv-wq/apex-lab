"""Tests for the portfolio simulation module."""

from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from apex_lab.research.portfolio.metrics import (
    compute_monthly_returns,
    compute_portfolio_metrics,
    compute_rolling_drawdown,
    compute_rolling_sharpe,
    compute_yearly_returns,
)
from apex_lab.research.portfolio.portfolio import simulate_portfolio
from apex_lab.research.portfolio.report import write_portfolio_reports

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


def _make_trades(returns: list[float]) -> pl.DataFrame:
    """Build a minimal trades DataFrame with given return_pct values."""
    base = datetime.datetime(2024, 1, 1, 9, 15)
    n = len(returns)
    return pl.DataFrame(
        {
            "entry_time": [base + datetime.timedelta(days=i * 2) for i in range(n)],
            "exit_time": [base + datetime.timedelta(days=i * 2 + 1) for i in range(n)],
            "return_pct": [float(r) for r in returns],
        }
    )


def _make_multi_month_trades() -> pl.DataFrame:
    """Build trades spanning multiple months and years."""
    # 4 trades per month across 3 months in 2024
    returns = [2.0, -1.0, 3.0, 1.5, -0.5, 2.5, 1.0, -2.0, 4.0, 0.5, -1.5, 2.0]
    dates = [
        # Jan 2024: 4 trades
        datetime.datetime(2024, 1, 10),
        datetime.datetime(2024, 1, 20),
        datetime.datetime(2024, 2, 5),
        datetime.datetime(2024, 2, 15),
        datetime.datetime(2024, 2, 25),
        datetime.datetime(2024, 3, 5),
        datetime.datetime(2024, 3, 15),
        datetime.datetime(2024, 3, 25),
        datetime.datetime(2024, 4, 5),
        datetime.datetime(2024, 4, 15),
        datetime.datetime(2024, 4, 25),
        datetime.datetime(2024, 5, 5),
    ]
    return pl.DataFrame(
        {
            "entry_time": [d - datetime.timedelta(days=1) for d in dates],
            "exit_time": dates,
            "return_pct": returns,
        }
    )


# ---------------------------------------------------------------------------
# simulate_portfolio – fixed sizing
# ---------------------------------------------------------------------------


class TestFixedPositionSizing:
    """Tests for fixed position sizing."""

    def test_position_size_is_always_initial_capital(self) -> None:
        trades = _make_trades([5.0, -3.0, 10.0])
        equity_df = simulate_portfolio(trades, initial_capital=25_000.0, position_sizing="fixed")
        assert equity_df.height == 3
        # Position size is always initial_capital
        for ps in equity_df["position_size"].to_list():
            assert ps == pytest.approx(25_000.0)

    def test_pnl_computed_from_initial_capital(self) -> None:
        trades = _make_trades([10.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0, position_sizing="fixed")
        assert equity_df["pnl"][0] == pytest.approx(1_000.0)

    def test_equity_accumulates_correctly(self) -> None:
        # 3 trades: +10%, -5%, +20% on fixed 10000 capital
        trades = _make_trades([10.0, -5.0, 20.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0, position_sizing="fixed")
        rows = equity_df.to_dicts()
        # Trade 1: start=10000, pnl=+1000, end=11000
        assert rows[0]["starting_equity"] == pytest.approx(10_000.0)
        assert rows[0]["pnl"] == pytest.approx(1_000.0)
        assert rows[0]["ending_equity"] == pytest.approx(11_000.0)
        # Trade 2: start=11000, pnl=-500 (10000 * -5%), end=10500
        assert rows[1]["starting_equity"] == pytest.approx(11_000.0)
        assert rows[1]["pnl"] == pytest.approx(-500.0)
        assert rows[1]["ending_equity"] == pytest.approx(10_500.0)
        # Trade 3: start=10500, pnl=+2000 (10000 * 20%), end=12500
        assert rows[2]["pnl"] == pytest.approx(2_000.0)
        assert rows[2]["ending_equity"] == pytest.approx(12_500.0)

    def test_trade_ids_start_at_one(self) -> None:
        trades = _make_trades([1.0, 2.0])
        equity_df = simulate_portfolio(trades, initial_capital=1_000.0, position_sizing="fixed")
        assert equity_df["trade_id"].to_list() == [1, 2]


# ---------------------------------------------------------------------------
# simulate_portfolio – percent_equity sizing
# ---------------------------------------------------------------------------


class TestPercentEquitySizing:
    """Tests for percent_equity position sizing."""

    def test_position_size_equals_current_equity(self) -> None:
        trades = _make_trades([10.0, -5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=20_000.0, position_sizing="percent_equity"
        )
        rows = equity_df.to_dicts()
        # Trade 1: position_size = 20000 (starting equity)
        assert rows[0]["position_size"] == pytest.approx(20_000.0)
        # Trade 2: position_size = 22000 (ending equity of trade 1)
        assert rows[1]["position_size"] == pytest.approx(22_000.0)

    def test_compounding_effect(self) -> None:
        trades = _make_trades([10.0, 10.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        rows = equity_df.to_dicts()
        # Trade 1: 10000 * 10% = +1000 → ending = 11000
        assert rows[0]["ending_equity"] == pytest.approx(11_000.0)
        # Trade 2: 11000 * 10% = +1100 → ending = 12100
        assert rows[1]["ending_equity"] == pytest.approx(12_100.0)

    def test_equity_never_negative(self) -> None:
        # A catastrophic loss that would push equity below zero
        trades = _make_trades([-200.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        assert equity_df["ending_equity"][0] >= 0.0


# ---------------------------------------------------------------------------
# simulate_portfolio – risk_percent sizing
# ---------------------------------------------------------------------------


class TestRiskPercentSizing:
    """Tests for risk_percent position sizing."""

    def test_position_size_is_fraction_of_equity(self) -> None:
        trades = _make_trades([5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=50_000.0, position_sizing="risk_percent", risk_percent=2.0
        )
        # position_size = 50000 * 2% = 1000
        assert equity_df["position_size"][0] == pytest.approx(1_000.0)

    def test_pnl_uses_risk_position(self) -> None:
        trades = _make_trades([100.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="risk_percent", risk_percent=1.0
        )
        # position = 10000 * 1% = 100; pnl = 100 * 100% = 100
        assert equity_df["pnl"][0] == pytest.approx(100.0)
        assert equity_df["ending_equity"][0] == pytest.approx(10_100.0)

    def test_position_shrinks_with_equity(self) -> None:
        # After a loss, position should be smaller
        trades = _make_trades([-50.0, 5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="risk_percent", risk_percent=1.0
        )
        rows = equity_df.to_dicts()
        # Trade 1 position: 10000 * 1% = 100; pnl: -50; ending: 9950
        assert rows[0]["position_size"] == pytest.approx(100.0)
        # Trade 2 position: 9950 * 1% = 99.5
        assert rows[1]["position_size"] == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# simulate_portfolio – empty trades
# ---------------------------------------------------------------------------


def test_simulate_portfolio_empty_trades() -> None:
    """Empty trades input should return empty equity DataFrame."""
    trades = _make_trades([])
    equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
    assert equity_df.height == 0
    assert set(equity_df.columns) == {
        "trade_id",
        "entry_time",
        "exit_time",
        "starting_equity",
        "position_size",
        "return_pct",
        "pnl",
        "ending_equity",
        "drawdown",
    }


# ---------------------------------------------------------------------------
# Validation constraints
# ---------------------------------------------------------------------------


class TestPortfolioValidation:
    """Validate the required invariants on the equity DataFrame."""

    def test_ending_equity_is_final_portfolio_value(self) -> None:
        trades = _make_trades([5.0, -2.0, 8.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        # The last ending_equity is the true final portfolio value
        expected_final = equity_df["ending_equity"][-1]
        assert equity_df["ending_equity"][-1] == pytest.approx(expected_final)

    def test_equity_never_negative(self) -> None:
        trades = _make_trades([-100.0, -100.0, -100.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=5_000.0, position_sizing="percent_equity"
        )
        assert all(e >= 0.0 for e in equity_df["ending_equity"].to_list())

    def test_drawdown_never_positive(self) -> None:
        trades = _make_trades([5.0, -10.0, 3.0, -2.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        for dd in equity_df["drawdown"].to_list():
            assert dd <= 0.0, f"Drawdown should be ≤ 0, got {dd}"

    def test_drawdown_zero_at_peak(self) -> None:
        trades = _make_trades([10.0, 10.0, 10.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        # All gains, no drawdown
        for dd in equity_df["drawdown"].to_list():
            assert dd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_portfolio_metrics
# ---------------------------------------------------------------------------


class TestPortfolioSummary:
    """Tests for compute_portfolio_metrics."""

    def test_initial_and_ending_capital(self) -> None:
        trades = _make_trades([10.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["initial_capital"] == 10_000.0
        assert summary["ending_capital"] == pytest.approx(11_000.0)

    def test_total_return_pct(self) -> None:
        trades = _make_trades([20.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["total_return_pct"] == pytest.approx(20.0)

    def test_number_of_trades(self) -> None:
        trades = _make_trades([1.0, 2.0, 3.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["number_of_trades"] == 3

    def test_empty_trades_returns_defaults(self) -> None:
        equity_df = simulate_portfolio(_make_trades([]), initial_capital=5_000.0)
        summary = compute_portfolio_metrics(equity_df, initial_capital=5_000.0)
        assert summary["number_of_trades"] == 0
        assert summary["ending_capital"] == 5_000.0
        assert summary["total_return_pct"] == 0.0

    def test_profit_factor(self) -> None:
        trades = _make_trades([10.0, -5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        # gross wins: 10*100 = 1000; gross losses: 5*110 = 550
        assert summary["profit_factor"] is not None
        assert summary["profit_factor"] > 1.0


class TestCAGR:
    """Tests for CAGR computation."""

    def test_cagr_one_year_doubling(self) -> None:
        # 1 trade that doubles capital over roughly 1 year
        base = datetime.datetime(2023, 1, 1)
        trades = pl.DataFrame(
            {
                "entry_time": [base],
                "exit_time": [base + datetime.timedelta(days=365)],
                "return_pct": [100.0],
            }
        )
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        # CAGR ≈ 100% for doubling in 1 year
        assert summary["cagr"] is not None
        assert summary["cagr"] == pytest.approx(100.0, abs=1.0)

    def test_cagr_none_when_no_time_elapsed(self) -> None:
        base = datetime.datetime(2024, 1, 1)
        trades = pl.DataFrame(
            {
                "entry_time": [base],
                "exit_time": [base],  # same time
                "return_pct": [5.0],
            }
        )
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["cagr"] is None


class TestMaximumDrawdown:
    """Tests for maximum drawdown computation."""

    def test_max_drawdown_reported_as_positive(self) -> None:
        trades = _make_trades([10.0, -20.0, 5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["maximum_drawdown"] >= 0.0

    def test_max_drawdown_zero_for_all_wins(self) -> None:
        trades = _make_trades([5.0, 3.0, 7.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["maximum_drawdown"] == pytest.approx(0.0)

    def test_max_drawdown_value(self) -> None:
        # Trade 1: +10% → equity = 11000, peak = 11000, dd = 0
        # Trade 2: -20% → equity = 8800, peak = 11000, dd = -20%
        # Trade 3: +5% → equity = 9240, peak = 11000, dd ≈ -16%
        trades = _make_trades([10.0, -20.0, 5.0])
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["maximum_drawdown"] == pytest.approx(20.0, abs=0.01)


class TestSharpeRatio:
    """Tests for Sharpe ratio computation."""

    def test_sharpe_none_when_zero_std(self) -> None:
        # All identical returns → std = 0 → Sharpe = None
        trades = _make_trades([5.0, 5.0, 5.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["sharpe_ratio"] is None

    def test_sharpe_positive_for_good_strategy(self) -> None:
        # Mostly winning trades
        trades = _make_trades([5.0, 3.0, 4.0, -1.0, 6.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["sharpe_ratio"] is not None
        assert summary["sharpe_ratio"] > 0.0

    def test_sharpe_negative_for_losing_strategy(self) -> None:
        trades = _make_trades([-5.0, -3.0, -4.0, 1.0, -6.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        summary = compute_portfolio_metrics(equity_df, initial_capital=10_000.0)
        assert summary["sharpe_ratio"] is not None
        assert summary["sharpe_ratio"] < 0.0


# ---------------------------------------------------------------------------
# Monthly returns
# ---------------------------------------------------------------------------


class TestMonthlyReport:
    """Tests for compute_monthly_returns."""

    def test_monthly_aggregation(self) -> None:
        trades = _make_multi_month_trades()
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        monthly = compute_monthly_returns(equity_df)
        assert "year" in monthly.columns
        assert "month" in monthly.columns
        assert "return_pct" in monthly.columns
        # Should have at least 2 months worth of data
        assert monthly.height >= 2

    def test_monthly_aggregation_correct_order(self) -> None:
        trades = _make_multi_month_trades()
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        monthly = compute_monthly_returns(equity_df)
        years = monthly["year"].to_list()
        months = monthly["month"].to_list()
        # Verify chronological order
        for i in range(1, len(years)):
            assert (years[i], months[i]) >= (years[i - 1], months[i - 1])

    def test_monthly_returns_columns(self) -> None:
        trades = _make_trades([2.0, -1.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        monthly = compute_monthly_returns(equity_df)
        assert set(monthly.columns) == {"year", "month", "return_pct"}

    def test_empty_produces_empty_monthly(self) -> None:
        equity_df = simulate_portfolio(_make_trades([]), initial_capital=10_000.0)
        monthly = compute_monthly_returns(equity_df)
        assert monthly.height == 0


# ---------------------------------------------------------------------------
# Yearly returns
# ---------------------------------------------------------------------------


class TestYearlyReport:
    """Tests for compute_yearly_returns."""

    def test_yearly_aggregation(self) -> None:
        trades = _make_multi_month_trades()
        equity_df = simulate_portfolio(
            trades, initial_capital=10_000.0, position_sizing="percent_equity"
        )
        yearly = compute_yearly_returns(equity_df)
        assert "year" in yearly.columns
        assert "return_pct" in yearly.columns

    def test_yearly_returns_columns(self) -> None:
        trades = _make_trades([5.0, -2.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        yearly = compute_yearly_returns(equity_df)
        assert set(yearly.columns) == {"year", "return_pct"}

    def test_empty_produces_empty_yearly(self) -> None:
        equity_df = simulate_portfolio(_make_trades([]), initial_capital=10_000.0)
        yearly = compute_yearly_returns(equity_df)
        assert yearly.height == 0

    def test_yearly_order(self) -> None:
        base_2023 = datetime.datetime(2023, 6, 1)
        base_2024 = datetime.datetime(2024, 6, 1)
        trades = pl.DataFrame(
            {
                "entry_time": [base_2023, base_2024],
                "exit_time": [base_2023 + datetime.timedelta(days=1), base_2024 + datetime.timedelta(days=1)],
                "return_pct": [10.0, -5.0],
            }
        )
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        yearly = compute_yearly_returns(equity_df)
        years = yearly["year"].to_list()
        assert years == sorted(years)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


class TestRollingMetrics:
    """Tests for rolling Sharpe and drawdown."""

    def test_rolling_sharpe_columns(self) -> None:
        trades = _make_trades([1.0, 2.0, -1.0, 3.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        rs = compute_rolling_sharpe(equity_df)
        assert set(rs.columns) == {"date", "rolling_sharpe"}
        assert rs.height == 4

    def test_rolling_sharpe_empty(self) -> None:
        equity_df = simulate_portfolio(_make_trades([]), initial_capital=10_000.0)
        rs = compute_rolling_sharpe(equity_df)
        assert rs.height == 0

    def test_rolling_sharpe_null_for_single_trade(self) -> None:
        trades = _make_trades([5.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        rs = compute_rolling_sharpe(equity_df)
        # With window=20, a single trade has no std → null
        assert rs["rolling_sharpe"][0] is None

    def test_rolling_drawdown_columns(self) -> None:
        trades = _make_trades([5.0, -3.0, 2.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        rd = compute_rolling_drawdown(equity_df)
        assert set(rd.columns) == {"date", "rolling_drawdown"}
        assert rd.height == 3

    def test_rolling_drawdown_matches_equity_drawdown(self) -> None:
        trades = _make_trades([5.0, -3.0, 2.0])
        equity_df = simulate_portfolio(trades, initial_capital=10_000.0)
        rd = compute_rolling_drawdown(equity_df)
        assert rd["rolling_drawdown"].to_list() == pytest.approx(
            equity_df["drawdown"].to_list()
        )

    def test_rolling_drawdown_empty(self) -> None:
        equity_df = simulate_portfolio(_make_trades([]), initial_capital=10_000.0)
        rd = compute_rolling_drawdown(equity_df)
        assert rd.height == 0


# ---------------------------------------------------------------------------
# write_portfolio_reports
# ---------------------------------------------------------------------------


def test_write_portfolio_reports(tmp_path: Path) -> None:
    """All six report files should be written to the output directory."""
    trades = _make_multi_month_trades()
    equity_df = simulate_portfolio(
        trades, initial_capital=25_000.0, position_sizing="percent_equity"
    )
    summary = compute_portfolio_metrics(equity_df, initial_capital=25_000.0)
    monthly = compute_monthly_returns(equity_df)
    yearly = compute_yearly_returns(equity_df)
    rs = compute_rolling_sharpe(equity_df)
    rd = compute_rolling_drawdown(equity_df)

    write_portfolio_reports(equity_df, summary, monthly, yearly, rs, rd, output_dir=tmp_path)

    assert (tmp_path / "equity.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "monthly_returns.csv").exists()
    assert (tmp_path / "yearly_returns.csv").exists()
    assert (tmp_path / "rolling_sharpe.csv").exists()
    assert (tmp_path / "rolling_drawdown.csv").exists()

    # Verify summary JSON structure
    written = json.loads((tmp_path / "summary.json").read_text())
    assert "initial_capital" in written
    assert "ending_capital" in written
    assert "total_return_pct" in written
    assert "cagr" in written
    assert "maximum_drawdown" in written
    assert "sharpe_ratio" in written
    assert "sortino_ratio" in written
    assert "calmar_ratio" in written
    assert "profit_factor" in written
    assert "number_of_trades" in written

    # Verify equity CSV columns
    eq_read = pl.read_csv(tmp_path / "equity.csv")
    assert "trade_id" in eq_read.columns
    assert "entry_time" in eq_read.columns
    assert "exit_time" in eq_read.columns
    assert "starting_equity" in eq_read.columns
    assert "position_size" in eq_read.columns
    assert "return_pct" in eq_read.columns
    assert "pnl" in eq_read.columns
    assert "ending_equity" in eq_read.columns
    assert "drawdown" in eq_read.columns
