"""Tests for the strategy research framework.

Covers:
- Strategy registry (register, get, list, dispatch)
- All six built-in strategy names and feature declarations
- Engine execution with mock datasets (no Kite contact)
- Leaderboard generation and ranking
- Single-strategy mode
- Report file generation
"""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from apex_lab.research.strategies.base import Strategy
from apex_lab.research.strategies.engine import (
    _build_metrics_df,
    _compute_composite_scores,
    run_strategy_research,
)
from apex_lab.research.strategies.registry import (
    EmaAtrExpansionStrategy,
    EmaCrossoverStrategy,
    EmaRsiStrategy,
    EmaVwapStrategy,
    OpeningRangeBreakoutStrategy,
    StrategyRegistry,
    VwapTrendStrategy,
    get_default_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TREND_UP = [100.0 + i * 0.5 for i in range(300)]
_TREND_DOWN = [150.0 - i * 0.5 for i in range(300)]


def _make_ohlcv(closes: list[float], start_date: datetime.datetime | None = None) -> pl.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    if start_date is None:
        start_date = datetime.datetime(2022, 1, 3, 9, 15, 0)
    timestamps = [start_date + datetime.timedelta(minutes=30 * i) for i in range(len(closes))]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [c - 0.3 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [10_000 + i * 10 for i in range(len(closes))],
        }
    )


def _crossover_closes(n: int = 300) -> list[float]:
    """Generate closes that produce at least one EMA crossover."""
    # Flat then rising then flat
    flat = [100.0] * 60
    rising = [100.0 + i * 1.5 for i in range(120)]
    high_flat = [280.0] * 60
    falling = [280.0 - i * 1.5 for i in range(60)]
    return (flat + rising + high_flat + falling)[:n]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_register_and_get(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossoverStrategy())
        strategy = registry.get("EMA Crossover")
        assert isinstance(strategy, EmaCrossoverStrategy)

    def test_get_case_insensitive(self) -> None:
        registry = get_default_registry()
        s1 = registry.get("ema crossover")
        s2 = registry.get("EMA Crossover")
        assert s1.name == s2.name

    def test_register_duplicate_raises(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossoverStrategy())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(EmaCrossoverStrategy())

    def test_get_missing_raises(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_all_strategies_insertion_order(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossoverStrategy())
        registry.register(EmaRsiStrategy())
        names = [s.name for s in registry.all_strategies()]
        assert names == ["EMA Crossover", "EMA + RSI"]

    def test_contains(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossoverStrategy())
        assert "ema crossover" in registry
        assert "nonexistent" not in registry

    def test_len(self) -> None:
        registry = StrategyRegistry()
        assert len(registry) == 0
        registry.register(EmaCrossoverStrategy())
        assert len(registry) == 1

    def test_repr_contains_names(self) -> None:
        registry = StrategyRegistry()
        registry.register(EmaCrossoverStrategy())
        assert "EMA Crossover" in repr(registry)


# ---------------------------------------------------------------------------
# Built-in strategy declarations
# ---------------------------------------------------------------------------


class TestBuiltinStrategies:
    @pytest.mark.parametrize(
        "strategy_cls, expected_features",
        [
            (EmaCrossoverStrategy, ["EMA"]),
            (OpeningRangeBreakoutStrategy, ["EMA"]),
            (VwapTrendStrategy, ["EMA", "VWAP"]),
            (EmaVwapStrategy, ["EMA", "VWAP"]),
            (EmaRsiStrategy, ["EMA", "RSI"]),
            (EmaAtrExpansionStrategy, ["EMA", "ATR"]),
        ],
    )
    def test_required_features(self, strategy_cls: type, expected_features: list[str]) -> None:
        strategy = strategy_cls()
        assert strategy.required_features == expected_features

    def test_default_registry_has_six_strategies(self) -> None:
        registry = get_default_registry()
        assert len(registry) == 6

    def test_default_registry_names(self) -> None:
        registry = get_default_registry()
        expected = {
            "EMA Crossover",
            "Opening Range Breakout",
            "VWAP Trend",
            "EMA + VWAP",
            "EMA + RSI",
            "EMA + ATR Expansion",
        }
        assert set(registry.names()) == expected

    def test_all_strategies_have_non_empty_description(self) -> None:
        registry = get_default_registry()
        for strategy in registry.all_strategies():
            assert strategy.description, f"{strategy.name} has empty description"

    def test_all_strategies_are_strategy_instances(self) -> None:
        registry = get_default_registry()
        for strategy in registry.all_strategies():
            assert isinstance(strategy, Strategy)


# ---------------------------------------------------------------------------
# Strategy dispatch (prepare / entry / exit)
# ---------------------------------------------------------------------------


class TestStrategyDispatch:
    def setup_method(self) -> None:
        self.df = _make_ohlcv(_crossover_closes())

    def test_ema_crossover_prepare_adds_required_columns(self) -> None:
        strategy = EmaCrossoverStrategy()
        enriched = strategy.prepare(self.df)
        for col in ("ema_200", "atr_pct", "bullish_crossover", "bearish_crossover"):
            assert col in enriched.columns, f"Missing column: {col}"

    def test_ema_crossover_entry_exit_are_boolean_series(self) -> None:
        strategy = EmaCrossoverStrategy()
        enriched = strategy.prepare(self.df)
        entry = strategy.entry_condition(enriched)
        exit_ = strategy.exit_condition(enriched)
        assert entry.dtype == pl.Boolean
        assert exit_.dtype == pl.Boolean
        assert len(entry) == len(self.df)
        assert len(exit_) == len(self.df)

    @pytest.mark.parametrize(
        "strategy_cls",
        [
            EmaCrossoverStrategy,
            OpeningRangeBreakoutStrategy,
            VwapTrendStrategy,
            EmaVwapStrategy,
            EmaRsiStrategy,
            EmaAtrExpansionStrategy,
        ],
    )
    def test_all_strategies_produce_required_backtester_columns(
        self, strategy_cls: type
    ) -> None:
        strategy = strategy_cls()
        enriched = strategy.prepare(self.df)
        for col in ("ema_200", "atr_pct"):
            assert col in enriched.columns, f"{strategy.name} missing '{col}'"


# ---------------------------------------------------------------------------
# Engine and leaderboard
# ---------------------------------------------------------------------------


class TestEngine:
    def test_run_strategy_research_returns_two_dataframes(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, metrics_df = run_strategy_research(df, output_dir=tmp_path)
        assert isinstance(leaderboard, pl.DataFrame)
        assert isinstance(metrics_df, pl.DataFrame)

    def test_leaderboard_has_all_strategies(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, _ = run_strategy_research(df, output_dir=tmp_path)
        assert leaderboard.height == 6

    def test_leaderboard_has_rank_column(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, _ = run_strategy_research(df, output_dir=tmp_path)
        assert "rank" in leaderboard.columns
        assert "composite_score" in leaderboard.columns

    def test_leaderboard_ranks_are_sequential(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, _ = run_strategy_research(df, output_dir=tmp_path)
        ranks = leaderboard["rank"].to_list()
        assert ranks == list(range(1, leaderboard.height + 1))

    def test_leaderboard_sorted_descending_by_composite_score(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, _ = run_strategy_research(df, output_dir=tmp_path)
        scores = leaderboard["composite_score"].drop_nulls().to_list()
        assert scores == sorted(scores, reverse=True)

    def test_ranking_is_deterministic(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        lb1, _ = run_strategy_research(df, output_dir=tmp_path / "run1")
        lb2, _ = run_strategy_research(df, output_dir=tmp_path / "run2")
        assert lb1["strategy"].to_list() == lb2["strategy"].to_list()
        assert lb1["composite_score"].to_list() == lb2["composite_score"].to_list()

    def test_single_strategy_mode(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, metrics_df = run_strategy_research(
            df, output_dir=tmp_path, strategy_name="EMA Crossover"
        )
        assert metrics_df.height == 1
        assert metrics_df["strategy"].to_list() == ["EMA Crossover"]

    def test_single_strategy_mode_case_insensitive(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        leaderboard, metrics_df = run_strategy_research(
            df, output_dir=tmp_path, strategy_name="ema crossover"
        )
        assert metrics_df.height == 1

    def test_single_strategy_unknown_raises(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        with pytest.raises(KeyError):
            run_strategy_research(df, output_dir=tmp_path, strategy_name="Nonexistent")

    def test_metrics_df_has_all_scorecard_columns(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        _, metrics_df = run_strategy_research(df, output_dir=tmp_path)
        required = {
            "strategy",
            "cagr",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "maximum_drawdown",
            "profit_factor",
            "win_rate",
            "average_trade",
            "expectancy",
            "number_of_trades",
            "avg_holding_time",
            "return_over_drawdown",
        }
        assert required.issubset(set(metrics_df.columns))


# ---------------------------------------------------------------------------
# Report files
# ---------------------------------------------------------------------------


class TestStrategyReports:
    def test_report_files_are_created(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        run_strategy_research(df, output_dir=tmp_path)
        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "top_strategy.json").exists()

    def test_leaderboard_csv_parseable(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        run_strategy_research(df, output_dir=tmp_path)
        loaded = pl.read_csv(tmp_path / "leaderboard.csv")
        assert loaded.height == 6

    def test_no_overwrite_on_second_run(self, tmp_path: Path) -> None:
        df = _make_ohlcv(_crossover_closes())
        run_strategy_research(df, output_dir=tmp_path)
        run_strategy_research(df, output_dir=tmp_path)
        # Both files should coexist — original + one timestamped copy
        csvs = list(tmp_path.glob("leaderboard*.csv"))
        assert len(csvs) == 2


# ---------------------------------------------------------------------------
# Composite score unit tests
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def _make_metrics_df(self, rows: list[dict]) -> pl.DataFrame:
        return _build_metrics_df(rows)

    def test_scores_sum_to_at_most_one(self) -> None:
        rows = [
            {
                "strategy": "A",
                "cagr": 10.0,
                "annual_return": 10.0,
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.0,
                "calmar_ratio": 1.0,
                "maximum_drawdown": 5.0,
                "profit_factor": 2.0,
                "win_rate": 0.6,
                "average_trade": 0.5,
                "expectancy": 100.0,
                "number_of_trades": 20,
                "avg_holding_time": 8.0,
                "return_over_drawdown": 2.0,
            },
        ]
        df = self._make_metrics_df(rows)
        scores = _compute_composite_scores(df)
        # Single row: non-drawdown metrics normalise to 1.0, drawdown inverts to 0.0.
        # composite = 0.25 + 0.25 + 0.20 + 0.0 + 0.15 = 0.85
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_better_strategy_gets_higher_score(self) -> None:
        good = {
            "strategy": "Good",
            "cagr": 20.0,
            "annual_return": 20.0,
            "sharpe_ratio": 2.0,
            "sortino_ratio": 2.5,
            "calmar_ratio": 2.0,
            "maximum_drawdown": 5.0,
            "profit_factor": 3.0,
            "win_rate": 0.65,
            "average_trade": 1.0,
            "expectancy": 200.0,
            "number_of_trades": 30,
            "avg_holding_time": 8.0,
            "return_over_drawdown": 4.0,
        }
        bad = {
            "strategy": "Bad",
            "cagr": 2.0,
            "annual_return": 2.0,
            "sharpe_ratio": 0.3,
            "sortino_ratio": 0.5,
            "calmar_ratio": 0.3,
            "maximum_drawdown": 30.0,
            "profit_factor": 0.8,
            "win_rate": 0.35,
            "average_trade": -0.2,
            "expectancy": -50.0,
            "number_of_trades": 10,
            "avg_holding_time": 12.0,
            "return_over_drawdown": 0.07,
        }
        df = self._make_metrics_df([good, bad])
        scores = _compute_composite_scores(df)
        assert scores[0] > scores[1]
