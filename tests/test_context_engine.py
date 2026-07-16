"""Integration tests for the Alpha Discovery Engine (context research)."""

from __future__ import annotations

import datetime
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from apex_lab.research.context.engine import run_context_research
from apex_lab.research.context.metrics import compute_bucket_metrics
from apex_lab.research.context.registry import get_registry
from apex_lab.research.context.report import (
    build_correlation,
    build_leaderboard,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_nifty_bank_sample(n: int = 2000) -> pl.DataFrame:
    """Build a large synthetic OHLCV dataset with multiple EMA crossovers."""
    base_ts = datetime.datetime(2016, 1, 1, 9, 15, 0)
    closes = [
        40_000.0 + i * 2.0 + 350.0 * math.sin(i / 8.0) + 120.0 * math.sin(i / 37.0)
        for i in range(n)
    ]
    for i in range(300, 420):
        closes[i] -= 500.0
    for i in range(700, 860):
        closes[i] += 650.0
    volumes = [100_000 + int(20_000 * (1.0 + math.sin(i / 15.0))) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)],
            "open": [c - 20.0 for c in closes],
            "high": [c + 35.0 for c in closes],
            "low": [c - 40.0 for c in closes],
            "close": closes,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestContextRegistry:
    def test_all_29_features_registered(self) -> None:
        """All 29 context features must be present in the registry."""
        registry = get_registry()
        assert len(registry) == 29

    def test_every_feature_has_name_and_group(self) -> None:
        registry = get_registry()
        for name, feature in registry.items():
            assert feature.name == name
            assert isinstance(feature.group, str) and feature.group

    def test_expected_feature_names_present(self) -> None:
        registry = get_registry()
        expected = {
            "ema200_side", "dist_ema200", "dist_ema50", "ema50_slope", "ema200_slope",
            "atr_percentile", "atr_state", "realized_vol_20", "bb_width_pct",
            "rsi_bucket", "macd_hist_bucket", "adx_bucket", "roc10_bucket", "roc20_bucket",
            "vwap_side", "dist_vwap", "vwap_slope",
            "higher_high", "higher_low", "lower_high", "lower_low", "swing_distance",
            "gap_pct_bucket", "or_position", "inside_or",
            "hour", "day_of_week", "month", "quarter",
        }
        assert expected == set(registry.keys())


# ---------------------------------------------------------------------------
# Feature execution
# ---------------------------------------------------------------------------


class TestEveryFeatureExecutes:
    """Every registered feature's compute(), label(), and numeric() must run."""

    def setup_method(self) -> None:
        self.df = _make_nifty_bank_sample(1000)

    def test_all_features_compute_without_error(self) -> None:
        from apex_lab.research.context.engine import _base_enrich

        enriched = _base_enrich(self.df)
        registry = get_registry()
        for name, feature in registry.items():
            enriched = feature.compute(enriched)
            assert isinstance(enriched, pl.DataFrame), f"{name}.compute() must return DataFrame"

    def test_all_features_label_returns_utf8_series(self) -> None:
        from apex_lab.research.context.engine import _enrich_ohlcv

        enriched = _enrich_ohlcv(self.df)
        registry = get_registry()
        for name, feature in registry.items():
            labels = feature.label(enriched)
            assert isinstance(labels, pl.Series), f"{name}.label() must return Series"
            assert labels.dtype == pl.Utf8, f"{name}.label() must return Utf8 series"
            assert len(labels) == len(self.df), f"{name}.label() length mismatch"

    def test_all_features_numeric_returns_float64_series(self) -> None:
        from apex_lab.research.context.engine import _enrich_ohlcv

        enriched = _enrich_ohlcv(self.df)
        registry = get_registry()
        for name, feature in registry.items():
            nums = feature.numeric(enriched)
            assert isinstance(nums, pl.Series), f"{name}.numeric() must return Series"
            assert nums.dtype == pl.Float64, f"{name}.numeric() must return Float64 series"
            assert len(nums) == len(self.df), f"{name}.numeric() length mismatch"


# ---------------------------------------------------------------------------
# Engine: run_context_research
# ---------------------------------------------------------------------------


class TestRunContextResearch:
    def test_writes_all_four_output_files(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        run_context_research(df, output_dir=tmp_path)
        assert (tmp_path / "summary.csv").exists()
        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "best_features.json").exists()
        assert (tmp_path / "correlation.csv").exists()

    def test_every_feature_appears_in_summary(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        summary, _, _, _ = run_context_research(df, output_dir=tmp_path)
        registry = get_registry()
        features_in_summary = set(summary["feature"].to_list())
        for fname in registry:
            assert fname in features_in_summary, f"Feature '{fname}' missing from summary.csv"

    def test_summary_has_required_columns(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        summary, _, _, _ = run_context_research(df, output_dir=tmp_path)
        required = {
            "feature", "bucket", "sample_size", "low_sample_size",
            "win_rate", "expectancy", "profit_factor", "sharpe", "maximum_drawdown",
        }
        assert required.issubset(set(summary.columns))

    def test_low_sample_buckets_excluded_from_leaderboard(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        summary, leaderboard, _, _ = run_context_research(df, output_dir=tmp_path)
        # All leaderboard entries must have sample_size >= 30
        lb_features = leaderboard["feature"].to_list()
        lb_buckets = leaderboard["bucket"].to_list()
        for feat, bkt in zip(lb_features, lb_buckets, strict=False):
            row = summary.filter(
                (pl.col("feature") == feat) & (pl.col("bucket") == bkt)
            )
            assert row.height == 1
            assert int(row["sample_size"][0]) >= 30, (
                f"Leaderboard entry {feat}/{bkt} has sample_size < 30"
            )

    def test_leaderboard_sorted_descending_by_score(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        _, leaderboard, _, _ = run_context_research(df, output_dir=tmp_path)
        if leaderboard.height < 2:
            pytest.skip("Not enough leaderboard entries")
        scores = leaderboard["score"].to_list()
        assert scores == sorted(scores, reverse=True), "Leaderboard must be sorted by score descending"

    def test_leaderboard_rank_column_sequential(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        _, leaderboard, _, _ = run_context_research(df, output_dir=tmp_path)
        if leaderboard.is_empty():
            pytest.skip("Empty leaderboard")
        ranks = leaderboard["rank"].to_list()
        assert ranks == list(range(1, len(ranks) + 1)), "Ranks must be sequential starting at 1"

    def test_scores_are_deterministic(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        _, lb1, bf1, _ = run_context_research(df, output_dir=tmp_path / "run1")
        _, lb2, bf2, _ = run_context_research(df, output_dir=tmp_path / "run2")
        if lb1.is_empty() and lb2.is_empty():
            pytest.skip("Empty leaderboard")
        scores1 = lb1["score"].to_list()
        scores2 = lb2["score"].to_list()
        assert scores1 == scores2, "Scores must be deterministic"
        assert bf1 == bf2, "best_features must be deterministic"

    def test_best_features_generated(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        _, _, best_features, _ = run_context_research(df, output_dir=tmp_path)
        assert isinstance(best_features, dict)
        assert len(best_features) > 0
        for key, value in best_features.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_best_features_json_valid(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        run_context_research(df, output_dir=tmp_path)
        content = (tmp_path / "best_features.json").read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_correlation_report_generated(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        _, _, _, correlation = run_context_research(df, output_dir=tmp_path)
        assert correlation.height > 0
        assert "feature" in correlation.columns
        assert "pearson" in correlation.columns
        assert "spearman" in correlation.columns

    def test_correlation_csv_readable(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        run_context_research(df, output_dir=tmp_path)
        corr = pl.read_csv(tmp_path / "correlation.csv")
        assert "feature" in corr.columns
        assert "pearson" in corr.columns
        assert "spearman" in corr.columns

    def test_summary_csv_readable(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        run_context_research(df, output_dir=tmp_path)
        summary = pl.read_csv(tmp_path / "summary.csv")
        assert "feature" in summary.columns
        assert "bucket" in summary.columns
        assert "sample_size" in summary.columns

    def test_empty_trades_produces_empty_reports(self, tmp_path: Path) -> None:
        # Flat price → no crossovers → no trades
        base_ts = datetime.datetime(2024, 1, 1, 9, 15, 0)
        df = pl.DataFrame(
            {
                "timestamp": [
                    base_ts + datetime.timedelta(minutes=30 * i) for i in range(300)
                ],
                "open": [100.0] * 300,
                "high": [101.0] * 300,
                "low": [99.0] * 300,
                "close": [100.0] * 300,
                "volume": [10_000] * 300,
            }
        )
        summary, leaderboard, best_features, correlation = run_context_research(
            df, output_dir=tmp_path
        )
        assert summary.is_empty()
        assert leaderboard.is_empty()
        assert best_features == {}
        assert correlation.is_empty()


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------


class TestComputeBucketMetrics:
    def _make_trades(self, returns: list[float]) -> pl.DataFrame:
        return pl.DataFrame({"return_pct": returns})

    def test_win_rate_correct(self) -> None:
        trades = self._make_trades([1.0, 2.0, -1.0, -0.5])
        m = compute_bucket_metrics(trades)
        assert m["win_rate"] == pytest.approx(0.5)

    def test_sample_size(self) -> None:
        trades = self._make_trades([0.5] * 35)
        m = compute_bucket_metrics(trades)
        assert m["sample_size"] == 35
        assert m["low_sample_size"] is False

    def test_low_sample_size_flag(self) -> None:
        trades = self._make_trades([0.5] * 29)
        m = compute_bucket_metrics(trades)
        assert m["low_sample_size"] is True

    def test_expectancy_positive_only_trades(self) -> None:
        trades = self._make_trades([2.0, 2.0, 2.0])
        m = compute_bucket_metrics(trades)
        assert m["expectancy"] == pytest.approx(2.0)

    def test_max_drawdown_is_non_negative(self) -> None:
        trades = self._make_trades([1.0, -3.0, 1.0, 2.0])
        m = compute_bucket_metrics(trades)
        assert m["maximum_drawdown"] >= 0.0

    def test_profit_factor_none_when_no_losses(self) -> None:
        trades = self._make_trades([1.0, 2.0, 3.0])
        m = compute_bucket_metrics(trades)
        assert m["profit_factor"] is None


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


class TestBuildLeaderboard:
    def _make_summary(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "feature": ["rsi_bucket", "rsi_bucket", "adx_bucket"],
                "bucket": ["50-60", "30-40", "25-35 (Strong)"],
                "sample_size": [50, 10, 60],
                "low_sample_size": [False, True, False],
                "win_rate": [0.6, 0.4, 0.55],
                "average_return": [0.5, 0.1, 0.4],
                "median_return": [0.4, 0.1, 0.35],
                "expectancy": [0.3, 0.05, 0.25],
                "profit_factor": [1.5, 1.1, 1.3],
                "sharpe": [0.8, 0.2, 0.7],
                "maximum_drawdown": [2.0, 0.5, 1.5],
            }
        )

    def test_low_sample_excluded(self) -> None:
        summary = self._make_summary()
        lb = build_leaderboard(summary)
        assert "30-40" not in lb["bucket"].to_list()

    def test_leaderboard_is_sorted_descending(self) -> None:
        summary = self._make_summary()
        lb = build_leaderboard(summary)
        scores = lb["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_score_between_0_and_1(self) -> None:
        summary = self._make_summary()
        lb = build_leaderboard(summary)
        for score in lb["score"].to_list():
            assert 0.0 <= score <= 1.0

    def test_empty_summary_returns_empty_leaderboard(self) -> None:
        empty = pl.DataFrame(
            {
                "feature": pl.Series([], dtype=pl.Utf8),
                "bucket": pl.Series([], dtype=pl.Utf8),
                "sample_size": pl.Series([], dtype=pl.Int64),
                "low_sample_size": pl.Series([], dtype=pl.Boolean),
                "win_rate": pl.Series([], dtype=pl.Float64),
                "average_return": pl.Series([], dtype=pl.Float64),
                "median_return": pl.Series([], dtype=pl.Float64),
                "expectancy": pl.Series([], dtype=pl.Float64),
                "profit_factor": pl.Series([], dtype=pl.Float64),
                "sharpe": pl.Series([], dtype=pl.Float64),
                "maximum_drawdown": pl.Series([], dtype=pl.Float64),
            }
        )
        lb = build_leaderboard(empty)
        assert lb.is_empty()


class TestBuildCorrelation:
    def test_correlation_has_correct_columns(self) -> None:
        registry = get_registry()
        # Build a minimal trade_contexts with ctx_ema200_side and num_ema200_side
        trade_contexts = pl.DataFrame(
            {
                "return_pct": [0.5, -0.3, 1.0, 0.2, -0.1],
                "num_ema200_side": [1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
        small_registry = {k: v for k, v in registry.items() if k == "ema200_side"}
        corr = build_correlation(trade_contexts, small_registry)
        assert "feature" in corr.columns
        assert "pearson" in corr.columns
        assert "spearman" in corr.columns

    def test_pearson_bounded(self) -> None:
        registry = get_registry()
        trade_contexts = pl.DataFrame(
            {
                "return_pct": [1.0, 2.0, 3.0, 4.0, 5.0],
                "num_ema200_side": [0.0, 0.0, 1.0, 1.0, 1.0],
            }
        )
        small_registry = {k: v for k, v in registry.items() if k == "ema200_side"}
        corr = build_correlation(trade_contexts, small_registry)
        for row in corr.iter_rows(named=True):
            if row["pearson"] is not None:
                assert -1.0 <= row["pearson"] <= 1.0
            if row["spearman"] is not None:
                assert -1.0 <= row["spearman"] <= 1.0


# ---------------------------------------------------------------------------
# CLI integration: --mode context
# ---------------------------------------------------------------------------


class TestResearchLabContextMode:
    def test_run_context_writes_reports(self, tmp_path: Path) -> None:
        module = _load_script_module()
        data_path = tmp_path / "input.parquet"
        _make_nifty_bank_sample().write_parquet(data_path)

        out_dir = tmp_path / "context"
        summary, leaderboard, best_features, correlation = module.run_context(
            data_path, output_dir=out_dir
        )
        assert (out_dir / "summary.csv").exists()
        assert (out_dir / "leaderboard.csv").exists()
        assert (out_dir / "best_features.json").exists()
        assert (out_dir / "correlation.csv").exists()
        assert summary.height > 0

    def test_run_context_every_feature_in_summary(self, tmp_path: Path) -> None:
        module = _load_script_module()
        data_path = tmp_path / "input.parquet"
        _make_nifty_bank_sample().write_parquet(data_path)

        out_dir = tmp_path / "context"
        summary, _, _, _ = module.run_context(data_path, output_dir=out_dir)
        registry = get_registry()
        features_in_summary = set(summary["feature"].to_list())
        for fname in registry:
            assert fname in features_in_summary, f"'{fname}' missing from summary"
