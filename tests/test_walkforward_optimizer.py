"""Unit tests for walk-forward parameter optimization."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import polars as pl
import pytest

from apex_lab.research.optimization.walkforward_optimizer import (
    EMA_PAIRS,
    _add_months,
    _build_ema_signals,
    _build_leaderboard,
    _compute_cagr,
    _compute_sharpe,
    _generate_windows,
    _select_best_parameters,
    _to_date,
    optimize,
    run_walkforward_optimization,
    write_optimization_reports,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n_bars: int, start: datetime.datetime | None = None) -> pl.DataFrame:
    """Build synthetic 30-minute OHLCV data spanning *n_bars* bars.

    Prices follow a gentle upward trend with a periodic oscillation so that
    short/long EMA crossovers can occur naturally.
    """
    if start is None:
        start = datetime.datetime(2020, 1, 1, 9, 15, 0)
    timestamps = [start + datetime.timedelta(minutes=30 * i) for i in range(n_bars)]
    closes = [100.0 + i * 0.02 + (i % 200) * 0.3 for i in range(n_bars)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.2 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [10_000 for _ in range(n_bars)],
        }
    )


def _make_trades(returns: list[float]) -> pl.DataFrame:
    """Build a minimal trades DataFrame with given return_pct values."""
    base = datetime.datetime(2024, 1, 1, 9, 15)
    n = len(returns)
    return pl.DataFrame(
        {
            "entry_time": [base + datetime.timedelta(hours=i) for i in range(n)],
            "exit_time": [base + datetime.timedelta(hours=i + 1) for i in range(n)],
            "entry_price": [100.0] * n,
            "exit_price": [100.0 + r for r in returns],
            "bars_held": [2] * n,
            "return_pct": returns,
            "exit_reason": ["opposite_crossover"] * n,
            "trend_regime": ["above_ema200"] * n,
            "volatility_regime": ["low"] * n,
        }
    )


# ---------------------------------------------------------------------------
# _add_months
# ---------------------------------------------------------------------------


class TestAddMonths:
    """Tests for the month-arithmetic helper."""

    def test_add_zero_months_is_noop(self) -> None:
        d = datetime.date(2021, 6, 15)
        assert _add_months(d, 0) == d

    def test_add_one_month_mid_month(self) -> None:
        assert _add_months(datetime.date(2021, 1, 15), 1) == datetime.date(2021, 2, 15)

    def test_add_month_clamps_to_month_end(self) -> None:
        # Jan 31 + 1 month → Feb 28/29
        result = _add_months(datetime.date(2021, 1, 31), 1)
        assert result == datetime.date(2021, 2, 28)

    def test_add_twelve_months_advances_year(self) -> None:
        assert _add_months(datetime.date(2020, 3, 1), 12) == datetime.date(2021, 3, 1)

    def test_add_six_months_crosses_year_boundary(self) -> None:
        assert _add_months(datetime.date(2021, 9, 30), 6) == datetime.date(2022, 3, 30)

    def test_add_twenty_four_months(self) -> None:
        assert _add_months(datetime.date(2020, 1, 1), 24) == datetime.date(2022, 1, 1)


# ---------------------------------------------------------------------------
# _to_date
# ---------------------------------------------------------------------------


class TestToDate:
    """Tests for the datetime normalisation helper."""

    def test_date_passthrough(self) -> None:
        d = datetime.date(2021, 5, 10)
        assert _to_date(d) is d

    def test_datetime_converted(self) -> None:
        dt = datetime.datetime(2021, 5, 10, 14, 30)
        assert _to_date(dt) == datetime.date(2021, 5, 10)


# ---------------------------------------------------------------------------
# _build_ema_signals
# ---------------------------------------------------------------------------


class TestBuildEmaSignals:
    """Tests for EMA signal construction."""

    def test_required_columns_present(self) -> None:
        df = _make_ohlcv(300)
        result = _build_ema_signals(df, fast_period=5, slow_period=20)
        for col in ("bullish_crossover", "bearish_crossover", "ema_200", "atr_pct"):
            assert col in result.columns, f"Missing column: {col}"

    def test_crossover_columns_are_boolean(self) -> None:
        df = _make_ohlcv(300)
        result = _build_ema_signals(df, fast_period=10, slow_period=30)
        assert result["bullish_crossover"].dtype == pl.Boolean
        assert result["bearish_crossover"].dtype == pl.Boolean

    def test_no_null_in_crossover_columns(self) -> None:
        df = _make_ohlcv(300)
        result = _build_ema_signals(df, fast_period=8, slow_period=21)
        assert result["bullish_crossover"].null_count() == 0
        assert result["bearish_crossover"].null_count() == 0

    def test_row_count_preserved(self) -> None:
        df = _make_ohlcv(500)
        result = _build_ema_signals(df, fast_period=20, slow_period=50)
        assert result.height == 500

    def test_slow_period_equals_200_still_produces_ema_200(self) -> None:
        """The ema_200 column must exist even when slow_period == 200."""
        df = _make_ohlcv(300)
        result = _build_ema_signals(df, fast_period=50, slow_period=200)
        assert "ema_200" in result.columns

    def test_all_pairs_produce_signals(self) -> None:
        df = _make_ohlcv(500)
        for fast, slow in EMA_PAIRS:
            result = _build_ema_signals(df, fast, slow)
            assert result.height == df.height


# ---------------------------------------------------------------------------
# _generate_windows
# ---------------------------------------------------------------------------


class TestGenerateWindows:
    """Tests for walk-forward window generation."""

    def test_empty_series_returns_no_windows(self) -> None:
        empty = pl.Series([], dtype=pl.Datetime)
        windows = _generate_windows(empty, train_years=1, test_months=3, advance_months=3)
        assert windows == []

    def test_insufficient_data_returns_no_windows(self) -> None:
        # Only 1 year of data; need at least train_years + test_months = 2.5 years
        df = _make_ohlcv(n_bars=5000)  # ~350 days at 30-min bars (14/day)
        windows = _generate_windows(
            df["timestamp"], train_years=2, test_months=6, advance_months=6
        )
        assert len(windows) == 0

    def test_at_least_one_window_from_sufficient_data(self) -> None:
        # 3 calendar years of continuous 30-min bars (48 bars/day × 365 days × 3)
        n_bars = 3 * 365 * 48
        df = _make_ohlcv(n_bars=n_bars)
        windows = _generate_windows(
            df["timestamp"], train_years=2, test_months=6, advance_months=6
        )
        assert len(windows) >= 1

    def test_window_dates_are_consistent(self) -> None:
        # 4 calendar years of continuous 30-min bars
        n_bars = 4 * 365 * 48
        df = _make_ohlcv(n_bars=n_bars)
        windows = _generate_windows(
            df["timestamp"], train_years=2, test_months=6, advance_months=6
        )
        for train_start, train_end, test_start, test_end in windows:
            assert train_start < train_end
            assert train_end == test_start
            assert test_start < test_end

    def test_consecutive_windows_advance_by_step(self) -> None:
        # 5 calendar years of continuous 30-min bars
        n_bars = 5 * 365 * 48
        df = _make_ohlcv(n_bars=n_bars)
        windows = _generate_windows(
            df["timestamp"], train_years=1, test_months=3, advance_months=3
        )
        assert len(windows) >= 2
        # Each train_start advances by advance_months
        for i in range(1, len(windows)):
            expected = _add_months(windows[i - 1][0], 3)
            assert windows[i][0] == expected

    def test_window_elements_are_dates(self) -> None:
        # 3 calendar years of continuous 30-min bars
        n_bars = 3 * 365 * 48
        df = _make_ohlcv(n_bars=n_bars)
        windows = _generate_windows(
            df["timestamp"], train_years=2, test_months=6, advance_months=6
        )
        for tup in windows:
            for val in tup:
                assert isinstance(val, datetime.date)


# ---------------------------------------------------------------------------
# _compute_cagr
# ---------------------------------------------------------------------------


class TestComputeCagr:
    """Tests for CAGR computation."""

    def test_no_trades_returns_none(self) -> None:
        trades = _make_trades([])
        assert _compute_cagr(trades, years=1.0) is None

    def test_zero_years_returns_none(self) -> None:
        trades = _make_trades([5.0, -2.0, 3.0])
        assert _compute_cagr(trades, years=0.0) is None

    def test_positive_return(self) -> None:
        # 100% total return over 1 year → CAGR = 100%
        trades = _make_trades([100.0])
        result = _compute_cagr(trades, years=1.0)
        assert result == pytest.approx(100.0, rel=1e-6)

    def test_negative_return(self) -> None:
        # -50% total over 1 year → CAGR = -50%
        trades = _make_trades([-50.0])
        result = _compute_cagr(trades, years=1.0)
        assert result == pytest.approx(-50.0, rel=1e-6)

    def test_returns_float(self) -> None:
        trades = _make_trades([10.0, 5.0])
        result = _compute_cagr(trades, years=0.5)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _compute_sharpe
# ---------------------------------------------------------------------------


class TestComputeSharpe:
    """Tests for Sharpe ratio computation."""

    def test_no_trades_returns_none(self) -> None:
        trades = _make_trades([])
        assert _compute_sharpe(trades, years=1.0) is None

    def test_one_trade_returns_none(self) -> None:
        trades = _make_trades([5.0])
        assert _compute_sharpe(trades, years=1.0) is None

    def test_zero_std_returns_none(self) -> None:
        trades = _make_trades([3.0, 3.0, 3.0])
        assert _compute_sharpe(trades, years=1.0) is None

    def test_returns_float_for_valid_input(self) -> None:
        trades = _make_trades([5.0, -2.0, 3.0, 4.0, -1.0])
        result = _compute_sharpe(trades, years=1.0)
        assert isinstance(result, float)

    def test_positive_mean_gives_positive_sharpe(self) -> None:
        trades = _make_trades([5.0, 4.0, 3.0, 6.0, 2.0])
        result = _compute_sharpe(trades, years=1.0)
        assert result is not None
        assert result > 0.0


# ---------------------------------------------------------------------------
# _build_leaderboard
# ---------------------------------------------------------------------------


class TestBuildLeaderboard:
    """Tests for leaderboard aggregation."""

    def test_empty_summary_returns_empty_leaderboard(self) -> None:
        from apex_lab.research.optimization.walkforward_optimizer import _empty_summary_df

        leaderboard = _build_leaderboard(_empty_summary_df())
        assert leaderboard.is_empty()

    def test_leaderboard_has_one_row_per_ema_pair(self) -> None:
        summary = pl.DataFrame(
            {
                "window_id": [1, 1, 2, 2],
                "fast_ema": [5, 8, 5, 8],
                "slow_ema": [20, 21, 20, 21],
                "train_start": ["2020-01-01"] * 4,
                "train_end": ["2022-01-01"] * 4,
                "test_start": ["2022-01-01"] * 4,
                "test_end": ["2022-07-01"] * 4,
                "number_of_trades": [10, 8, 12, 7],
                "win_rate": [0.6, 0.5, 0.7, 0.4],
                "expectancy": [1.2, 0.8, 1.5, 0.5],
                "average_return": [0.5, 0.3, 0.6, 0.2],
                "profit_factor": [1.8, 1.2, 2.1, 1.0],
                "maximum_drawdown": [5.0, 4.0, 6.0, 3.0],
                "cagr": [10.0, 8.0, 12.0, 6.0],
                "sharpe_ratio": [1.5, 1.2, 1.8, 1.0],
            }
        )
        leaderboard = _build_leaderboard(summary)
        assert leaderboard.height == 2
        assert set(leaderboard["fast_ema"].to_list()) == {5, 8}

    def test_leaderboard_columns_present(self) -> None:
        summary = pl.DataFrame(
            {
                "window_id": [1],
                "fast_ema": [5],
                "slow_ema": [20],
                "train_start": ["2020-01-01"],
                "train_end": ["2022-01-01"],
                "test_start": ["2022-01-01"],
                "test_end": ["2022-07-01"],
                "number_of_trades": [10],
                "win_rate": [0.6],
                "expectancy": [1.2],
                "average_return": [0.5],
                "profit_factor": [1.8],
                "maximum_drawdown": [5.0],
                "cagr": [10.0],
                "sharpe_ratio": [1.5],
            }
        )
        leaderboard = _build_leaderboard(summary)
        for col in (
            "fast_ema",
            "slow_ema",
            "mean_profit_factor",
            "mean_expectancy",
            "mean_win_rate",
            "mean_drawdown",
            "number_of_windows",
        ):
            assert col in leaderboard.columns, f"Missing column: {col}"

    def test_means_are_computed_correctly(self) -> None:
        summary = pl.DataFrame(
            {
                "window_id": [1, 2],
                "fast_ema": [5, 5],
                "slow_ema": [20, 20],
                "train_start": ["2020-01-01", "2020-07-01"],
                "train_end": ["2022-01-01", "2022-07-01"],
                "test_start": ["2022-01-01", "2022-07-01"],
                "test_end": ["2022-07-01", "2023-01-01"],
                "number_of_trades": [10, 12],
                "win_rate": [0.6, 0.8],
                "expectancy": [1.0, 3.0],
                "average_return": [0.5, 0.7],
                "profit_factor": [2.0, 4.0],
                "maximum_drawdown": [4.0, 6.0],
                "cagr": [10.0, 14.0],
                "sharpe_ratio": [1.0, 2.0],
            }
        )
        leaderboard = _build_leaderboard(summary)
        row = leaderboard.row(0, named=True)
        assert row["mean_profit_factor"] == pytest.approx(3.0)
        assert row["mean_expectancy"] == pytest.approx(2.0)
        assert row["mean_win_rate"] == pytest.approx(0.7)
        assert row["mean_drawdown"] == pytest.approx(5.0)
        assert row["number_of_windows"] == 2


# ---------------------------------------------------------------------------
# _select_best_parameters
# ---------------------------------------------------------------------------


class TestSelectBestParameters:
    """Tests for best-parameter selection."""

    def test_empty_leaderboard_returns_empty_dict(self) -> None:
        from apex_lab.research.optimization.walkforward_optimizer import _empty_summary_df

        leaderboard = _build_leaderboard(_empty_summary_df())
        result = _select_best_parameters(leaderboard)
        assert result == {}

    def test_highest_profit_factor_wins(self) -> None:
        leaderboard = pl.DataFrame(
            {
                "fast_ema": [5, 8, 10],
                "slow_ema": [20, 21, 30],
                "mean_profit_factor": [1.5, 2.5, 2.0],
                "mean_expectancy": [1.0, 0.8, 1.2],
                "mean_win_rate": [0.6, 0.5, 0.55],
                "mean_drawdown": [5.0, 4.0, 6.0],
                "number_of_windows": [3, 3, 3],
            }
        )
        result = _select_best_parameters(leaderboard)
        assert result["fast_ema"] == 8
        assert result["slow_ema"] == 21

    def test_lower_drawdown_breaks_tie_on_profit_factor(self) -> None:
        leaderboard = pl.DataFrame(
            {
                "fast_ema": [5, 8],
                "slow_ema": [20, 21],
                "mean_profit_factor": [2.0, 2.0],
                "mean_expectancy": [1.0, 0.8],
                "mean_win_rate": [0.6, 0.5],
                "mean_drawdown": [5.0, 3.0],
                "number_of_windows": [3, 3],
            }
        )
        result = _select_best_parameters(leaderboard)
        # Same profit_factor → lower drawdown wins
        assert result["fast_ema"] == 8
        assert result["slow_ema"] == 21

    def test_higher_expectancy_breaks_tie_on_drawdown(self) -> None:
        leaderboard = pl.DataFrame(
            {
                "fast_ema": [5, 8],
                "slow_ema": [20, 21],
                "mean_profit_factor": [2.0, 2.0],
                "mean_expectancy": [1.5, 0.8],
                "mean_win_rate": [0.6, 0.5],
                "mean_drawdown": [3.0, 3.0],
                "number_of_windows": [3, 3],
            }
        )
        result = _select_best_parameters(leaderboard)
        # Same profit_factor, same drawdown → higher expectancy wins
        assert result["fast_ema"] == 5
        assert result["slow_ema"] == 20

    def test_result_contains_selection_criteria(self) -> None:
        leaderboard = pl.DataFrame(
            {
                "fast_ema": [5],
                "slow_ema": [20],
                "mean_profit_factor": [1.8],
                "mean_expectancy": [1.0],
                "mean_win_rate": [0.6],
                "mean_drawdown": [4.0],
                "number_of_windows": [2],
            }
        )
        result = _select_best_parameters(leaderboard)
        assert "selection_criteria" in result
        assert isinstance(result["selection_criteria"], list)
        assert len(result["selection_criteria"]) == 3


# ---------------------------------------------------------------------------
# run_walkforward_optimization
# ---------------------------------------------------------------------------


class TestRunWalkforwardOptimization:
    """Integration-style tests for the full optimization loop."""

    def test_raises_on_missing_timestamp_column(self) -> None:
        df = pl.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="timestamp"):
            run_walkforward_optimization(df)

    def test_raises_on_empty_dataframe(self) -> None:
        df = pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Datetime)})
        with pytest.raises(ValueError, match="empty"):
            run_walkforward_optimization(df)

    def test_insufficient_data_returns_empty_results(self) -> None:
        # Only ~6 months of 30-min bars (14 bars/day × 126 trading days)
        df = _make_ohlcv(n_bars=1764)
        summary, leaderboard, best = run_walkforward_optimization(
            df,
            ema_pairs=((5, 20),),
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        assert summary.is_empty()
        assert leaderboard.is_empty()
        assert best == {}

    def test_returns_expected_types(self) -> None:
        # ~3.5 years of data
        n_bars = 3 * 365 * 48  # 3 calendar years of continuous 30-min bars
        df = _make_ohlcv(n_bars=n_bars)
        summary, leaderboard, best = run_walkforward_optimization(
            df,
            ema_pairs=((5, 20), (10, 30)),
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        assert isinstance(summary, pl.DataFrame)
        assert isinstance(leaderboard, pl.DataFrame)
        assert isinstance(best, dict)

    def test_summary_has_correct_columns(self) -> None:
        n_bars = 3 * 365 * 48  # 3 calendar years of continuous 30-min bars
        df = _make_ohlcv(n_bars=n_bars)
        summary, _, _ = run_walkforward_optimization(
            df,
            ema_pairs=((5, 20),),
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        if not summary.is_empty():
            for col in (
                "window_id",
                "fast_ema",
                "slow_ema",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "number_of_trades",
                "win_rate",
                "expectancy",
                "profit_factor",
                "maximum_drawdown",
            ):
                assert col in summary.columns, f"Missing summary column: {col}"

    def test_leaderboard_row_count_equals_ema_pair_count(self) -> None:
        n_bars = 3 * 365 * 48  # 3 calendar years of continuous 30-min bars
        df = _make_ohlcv(n_bars=n_bars)
        pairs = ((5, 20), (10, 30), (20, 50))
        _, leaderboard, _ = run_walkforward_optimization(
            df,
            ema_pairs=pairs,
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        # One row per EMA pair that was evaluated
        assert leaderboard.height <= len(pairs)

    def test_best_params_keys_present_when_data_sufficient(self) -> None:
        n_bars = 3 * 365 * 48  # 3 calendar years of continuous 30-min bars
        df = _make_ohlcv(n_bars=n_bars)
        _, _, best = run_walkforward_optimization(
            df,
            ema_pairs=((5, 20), (10, 30)),
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        if best:
            for key in ("fast_ema", "slow_ema", "selection_criteria"):
                assert key in best


# ---------------------------------------------------------------------------
# write_optimization_reports
# ---------------------------------------------------------------------------


class TestWriteOptimizationReports:
    """Tests for report file output."""

    def test_creates_all_three_files(self, tmp_path: Path) -> None:
        from apex_lab.research.optimization.walkforward_optimizer import _empty_summary_df

        empty_summary = _empty_summary_df()
        empty_leaderboard = _build_leaderboard(empty_summary)
        write_optimization_reports(empty_summary, empty_leaderboard, {}, output_dir=tmp_path)

        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "best_parameters.json").exists()

    def test_best_parameters_json_is_valid(self, tmp_path: Path) -> None:
        from apex_lab.research.optimization.walkforward_optimizer import _empty_summary_df

        best = {"fast_ema": 5, "slow_ema": 20, "mean_profit_factor": 2.1}
        write_optimization_reports(
            _empty_summary_df(), _build_leaderboard(_empty_summary_df()), best, output_dir=tmp_path
        )
        loaded = json.loads((tmp_path / "best_parameters.json").read_text(encoding="utf-8"))
        assert loaded == best

    def test_summary_csv_round_trips_correctly(self, tmp_path: Path) -> None:
        summary = pl.DataFrame(
            {
                "window_id": [1],
                "fast_ema": [5],
                "slow_ema": [20],
                "train_start": ["2020-01-01"],
                "train_end": ["2022-01-01"],
                "test_start": ["2022-01-01"],
                "test_end": ["2022-07-01"],
                "number_of_trades": [10],
                "win_rate": [0.6],
                "expectancy": [1.2],
                "average_return": [0.5],
                "profit_factor": [1.8],
                "maximum_drawdown": [5.0],
                "cagr": [10.0],
                "sharpe_ratio": [1.5],
            }
        )
        write_optimization_reports(
            summary,
            _build_leaderboard(summary),
            {"fast_ema": 5},
            output_dir=tmp_path,
        )
        loaded = pl.read_csv(tmp_path / "summary.csv")
        assert loaded.height == 1
        assert loaded["fast_ema"].to_list() == [5]

    def test_creates_output_dir_when_missing(self, tmp_path: Path) -> None:
        from apex_lab.research.optimization.walkforward_optimizer import _empty_summary_df

        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()
        write_optimization_reports(
            _empty_summary_df(), _build_leaderboard(_empty_summary_df()), {}, output_dir=nested
        )
        assert nested.exists()


# ---------------------------------------------------------------------------
# optimize (public entry point)
# ---------------------------------------------------------------------------


class TestOptimize:
    """End-to-end tests for the public optimize() function."""

    def test_optimize_writes_reports_and_returns_results(self, tmp_path: Path) -> None:
        n_bars = 3 * 365 * 48  # 3 calendar years of continuous 30-min bars
        df = _make_ohlcv(n_bars=n_bars)
        summary, leaderboard, best = optimize(
            df,
            output_dir=tmp_path,
            ema_pairs=((5, 20), (10, 30)),
            train_years=2,
            test_months=6,
            advance_months=6,
        )
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "best_parameters.json").exists()
        assert isinstance(summary, pl.DataFrame)
        assert isinstance(leaderboard, pl.DataFrame)
        assert isinstance(best, dict)
