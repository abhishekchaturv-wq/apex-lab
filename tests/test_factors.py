"""Tests for the factor combination research engine."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import polars as pl
import pytest

from apex_lab.research.factors.atr_volatility import AtrVolatilityFactor
from apex_lab.research.factors.base import Factor
from apex_lab.research.factors.ema_trend import EmaTrendFactor
from apex_lab.research.factors.factor_engine import (
    COMBINATIONS,
    FACTOR_REGISTRY,
    _build_leaderboard,
    _build_summary,
    _compute_combined_signal,
    _enrich_for_combination,
    run_factor_research,
)
from apex_lab.research.factors.macd import MacdFactor
from apex_lab.research.factors.rsi import RsiFactor
from apex_lab.research.factors.vwap import VwapFactor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300) -> pl.DataFrame:
    """Build a synthetic OHLCV DataFrame with a detectable EMA crossover."""
    base_ts = datetime.datetime(2024, 1, 2, 9, 15, 0)
    # Trend up sharply after 100 bars so EMA20 crosses EMA50
    closes = [100.0 + i * 0.1 for i in range(100)] + [120.0 + i * 0.05 for i in range(n - 100)]
    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(n)],
            "open": [c - 0.3 for c in closes],
            "high": [c + 0.8 for c in closes],
            "low": [c - 0.6 for c in closes],
            "close": closes,
            "volume": [50_000 + i * 100 for i in range(n)],
        }
    )


def _make_nifty_bank_sample(n: int = 1200) -> pl.DataFrame:
    """Build a deterministic, NIFTY BANK-like 30m OHLCV sample for integration checks."""
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
# Factor: EmaTrendFactor
# ---------------------------------------------------------------------------


class TestEmaTrendFactor:
    def test_name(self) -> None:
        assert EmaTrendFactor().name == "EMA"

    def test_compute_adds_required_columns(self) -> None:
        df = _make_ohlcv()
        enriched = EmaTrendFactor().compute(df)
        for col in ("ema_20", "ema_50", "ema_200", "atr_14", "atr_pct", "bullish_crossover", "bearish_crossover"):
            assert col in enriched.columns, f"Missing column: {col}"

    def test_compute_is_idempotent(self) -> None:
        df = _make_ohlcv()
        factor = EmaTrendFactor()
        once = factor.compute(df)
        twice = factor.compute(once)
        assert once.shape == twice.shape
        assert once.columns == twice.columns

    def test_signal_is_boolean_series(self) -> None:
        df = _make_ohlcv()
        enriched = EmaTrendFactor().compute(df)
        sig = EmaTrendFactor().signal(enriched)
        assert sig.dtype == pl.Boolean
        assert len(sig) == len(df)

    def test_signal_has_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = EmaTrendFactor().compute(df)
        sig = EmaTrendFactor().signal(enriched)
        assert sig.null_count() == 0

    def test_metadata_keys(self) -> None:
        meta = EmaTrendFactor().metadata()
        assert "factor" in meta
        assert "fast_period" in meta
        assert "slow_period" in meta


# ---------------------------------------------------------------------------
# Factor: RsiFactor
# ---------------------------------------------------------------------------


class TestRsiFactor:
    def test_name(self) -> None:
        assert RsiFactor().name == "RSI"

    def test_compute_adds_rsi_column(self) -> None:
        df = _make_ohlcv()
        enriched = RsiFactor().compute(df)
        assert "rsi_14" in enriched.columns

    def test_compute_is_idempotent(self) -> None:
        df = _make_ohlcv()
        factor = RsiFactor()
        once = factor.compute(df)
        twice = factor.compute(once)
        assert once.shape == twice.shape

    def test_rsi_bounded(self) -> None:
        df = _make_ohlcv(200)
        enriched = RsiFactor().compute(df)
        rsi = enriched["rsi_14"].drop_nulls()
        assert float(rsi.min()) >= 0.0
        assert float(rsi.max()) <= 100.0

    def test_signal_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = RsiFactor().compute(df)
        sig = RsiFactor().signal(enriched)
        assert sig.null_count() == 0

    def test_metadata_keys(self) -> None:
        meta = RsiFactor().metadata()
        assert "period" in meta
        assert "signal_threshold" in meta


# ---------------------------------------------------------------------------
# Factor: MacdFactor
# ---------------------------------------------------------------------------


class TestMacdFactor:
    def test_name(self) -> None:
        assert MacdFactor().name == "MACD"

    def test_compute_adds_macd_columns(self) -> None:
        df = _make_ohlcv()
        enriched = MacdFactor().compute(df)
        for col in ("macd_line", "macd_signal", "macd_hist"):
            assert col in enriched.columns

    def test_compute_is_idempotent(self) -> None:
        df = _make_ohlcv()
        factor = MacdFactor()
        once = factor.compute(df)
        twice = factor.compute(once)
        assert once.shape == twice.shape

    def test_signal_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = MacdFactor().compute(df)
        sig = MacdFactor().signal(enriched)
        assert sig.null_count() == 0

    def test_macd_hist_equals_line_minus_signal(self) -> None:
        df = _make_ohlcv(200)
        enriched = MacdFactor().compute(df)
        diff = (enriched["macd_line"] - enriched["macd_signal"] - enriched["macd_hist"]).abs()
        assert float(diff.max()) < 1e-9

    def test_metadata_keys(self) -> None:
        meta = MacdFactor().metadata()
        assert "fast_period" in meta
        assert "signal_period" in meta


# ---------------------------------------------------------------------------
# Factor: VwapFactor
# ---------------------------------------------------------------------------


class TestVwapFactor:
    def test_name(self) -> None:
        assert VwapFactor().name == "VWAP"

    def test_compute_adds_vwap_column(self) -> None:
        df = _make_ohlcv()
        enriched = VwapFactor().compute(df)
        assert "vwap" in enriched.columns

    def test_compute_is_idempotent(self) -> None:
        df = _make_ohlcv()
        factor = VwapFactor()
        once = factor.compute(df)
        twice = factor.compute(once)
        assert once.shape == twice.shape

    def test_vwap_positive(self) -> None:
        df = _make_ohlcv(200)
        enriched = VwapFactor().compute(df)
        vwap = enriched["vwap"].drop_nulls()
        assert float(vwap.min()) > 0.0

    def test_vwap_with_zero_volume_has_no_nan(self) -> None:
        df = _make_ohlcv().with_columns(pl.lit(0).alias("volume"))
        enriched = VwapFactor().compute(df)
        assert int(enriched["vwap"].is_nan().sum()) == 0

    def test_signal_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = VwapFactor().compute(df)
        sig = VwapFactor().signal(enriched)
        assert sig.null_count() == 0

    def test_metadata_keys(self) -> None:
        meta = VwapFactor().metadata()
        assert "mode" in meta


# ---------------------------------------------------------------------------
# Factor: AtrVolatilityFactor
# ---------------------------------------------------------------------------


class TestAtrVolatilityFactor:
    def test_name(self) -> None:
        assert AtrVolatilityFactor().name == "ATR"

    def test_compute_adds_atr_columns(self) -> None:
        df = _make_ohlcv()
        enriched = AtrVolatilityFactor().compute(df)
        assert "atr_14" in enriched.columns
        assert "atr_pct" in enriched.columns

    def test_compute_is_idempotent(self) -> None:
        df = _make_ohlcv()
        factor = AtrVolatilityFactor()
        once = factor.compute(df)
        twice = factor.compute(once)
        assert once.shape == twice.shape

    def test_atr_pct_bounded(self) -> None:
        df = _make_ohlcv(200)
        enriched = AtrVolatilityFactor().compute(df)
        pct = enriched["atr_pct"].drop_nulls()
        assert float(pct.min()) >= 0.0
        assert float(pct.max()) <= 100.0

    def test_signal_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = AtrVolatilityFactor().compute(df)
        sig = AtrVolatilityFactor().signal(enriched)
        assert sig.null_count() == 0

    def test_metadata_keys(self) -> None:
        meta = AtrVolatilityFactor().metadata()
        assert "min_pct" in meta
        assert "max_pct" in meta


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------


class TestFactorRegistry:
    def test_all_expected_keys_present(self) -> None:
        for key in ("EMA", "RSI", "MACD", "VWAP", "ATR"):
            assert key in FACTOR_REGISTRY

    def test_all_values_are_factor_instances(self) -> None:
        for key, factor in FACTOR_REGISTRY.items():
            assert isinstance(factor, Factor), f"{key} is not a Factor instance"


# ---------------------------------------------------------------------------
# Factor engine internals
# ---------------------------------------------------------------------------


class TestEnrichForCombination:
    def test_accumulates_columns_from_all_factors(self) -> None:
        df = _make_ohlcv()
        enriched = _enrich_for_combination(df, ("EMA", "RSI"), FACTOR_REGISTRY)
        assert "bullish_crossover" in enriched.columns
        assert "rsi_14" in enriched.columns

    def test_ema_and_macd(self) -> None:
        df = _make_ohlcv()
        enriched = _enrich_for_combination(df, ("EMA", "MACD"), FACTOR_REGISTRY)
        assert "macd_line" in enriched.columns

    def test_ema_and_vwap(self) -> None:
        df = _make_ohlcv()
        enriched = _enrich_for_combination(df, ("EMA", "VWAP"), FACTOR_REGISTRY)
        assert "vwap" in enriched.columns


class TestComputeCombinedSignal:
    def test_combined_signal_is_subset_of_ema_signal(self) -> None:
        df = _make_ohlcv()
        enriched = _enrich_for_combination(df, ("EMA", "RSI"), FACTOR_REGISTRY)
        ema_sig = FACTOR_REGISTRY["EMA"].signal(enriched)
        combined = _compute_combined_signal(enriched, ("EMA", "RSI"), FACTOR_REGISTRY)
        # Combined (AND) must fire on a subset of EMA-only bars
        ema_count = int(ema_sig.sum())
        combined_count = int(combined.sum())
        assert combined_count <= ema_count

    def test_combined_signal_no_nulls(self) -> None:
        df = _make_ohlcv()
        enriched = _enrich_for_combination(df, ("EMA", "RSI", "MACD"), FACTOR_REGISTRY)
        sig = _compute_combined_signal(enriched, ("EMA", "RSI", "MACD"), FACTOR_REGISTRY)
        assert sig.null_count() == 0


class TestBuildLeaderboard:
    def test_empty_input_returns_correct_schema(self) -> None:
        lb = _build_leaderboard([])
        assert lb.is_empty()
        assert "factor_combination" in lb.columns
        assert "trade_reduction_pct" in lb.columns
        assert "win_rate" in lb.columns

    def test_row_count_matches_input(self) -> None:
        rows = [
            {
                "factor_combination": "EMA",
                "number_of_trades": 10,
                "win_rate": 0.5,
                "expectancy": 0.2,
                "profit_factor": 1.2,
                "maximum_drawdown": -4.0,
            },
            {
                "factor_combination": "EMA AND RSI",
                "number_of_trades": 5,
                "win_rate": 0.6,
                "expectancy": 0.3,
                "profit_factor": 1.5,
                "maximum_drawdown": -3.0,
            }
        ]
        lb = _build_leaderboard(rows)
        assert lb.height == 2
        ema_reduction = lb.filter(pl.col("factor_combination") == "EMA")["trade_reduction_pct"][0]
        ema_rsi_reduction = lb.filter(pl.col("factor_combination") == "EMA AND RSI")[
            "trade_reduction_pct"
        ][0]
        assert float(ema_reduction) == 0.0
        assert float(ema_rsi_reduction) == -50.0


class TestBuildSummary:
    def test_empty_input_returns_correct_schema(self) -> None:
        s = _build_summary([])
        assert s.is_empty()
        assert "factor_combination" in s.columns

    def test_factor_combination_is_first_column(self) -> None:
        s = _build_summary([])
        assert s.columns[0] == "factor_combination"


# ---------------------------------------------------------------------------
# End-to-end: run_factor_research
# ---------------------------------------------------------------------------


class TestRunFactorResearch:
    def test_produces_leaderboard_and_summary(self, tmp_path: Path) -> None:
        df = _make_ohlcv(300)
        leaderboard, summary = run_factor_research(df, output_dir=tmp_path)
        assert leaderboard.height == len(COMBINATIONS)
        assert "factor_combination" in leaderboard.columns
        assert "number_of_trades" in leaderboard.columns
        assert "trade_reduction_pct" in leaderboard.columns
        assert "win_rate" in leaderboard.columns
        assert "expectancy" in leaderboard.columns
        assert "profit_factor" in leaderboard.columns
        assert "maximum_drawdown" in leaderboard.columns

    def test_writes_csv_files(self, tmp_path: Path) -> None:
        df = _make_ohlcv(300)
        run_factor_research(df, output_dir=tmp_path)
        assert (tmp_path / "leaderboard.csv").exists()
        assert (tmp_path / "summary.csv").exists()

    def test_leaderboard_csv_readable(self, tmp_path: Path) -> None:
        df = _make_ohlcv(300)
        run_factor_research(df, output_dir=tmp_path)
        lb = pl.read_csv(tmp_path / "leaderboard.csv")
        assert lb.height == len(COMBINATIONS)

    def test_summary_csv_contains_factor_combination_column(self, tmp_path: Path) -> None:
        df = _make_ohlcv(300)
        _, summary = run_factor_research(df, output_dir=tmp_path)
        if summary.height > 0:
            assert "factor_combination" in summary.columns

    def test_all_combination_labels_present_in_leaderboard(self, tmp_path: Path) -> None:
        df = _make_ohlcv(300)
        leaderboard, _ = run_factor_research(df, output_dir=tmp_path)
        expected_labels = {" AND ".join(c) for c in COMBINATIONS}
        actual_labels = set(leaderboard["factor_combination"].to_list())
        assert actual_labels == expected_labels
        assert "EMA" in actual_labels

    def test_custom_registry_is_used(self, tmp_path: Path) -> None:
        """Verify the engine accepts a custom registry (for testing isolation)."""

        class AlwaysTrueEma(EmaTrendFactor):
            def signal(self, df: pl.DataFrame) -> pl.Series:
                return pl.Series([True] * len(df), dtype=pl.Boolean)

        class AlwaysTrueRsi(RsiFactor):
            def signal(self, df: pl.DataFrame) -> pl.Series:
                return pl.Series([True] * len(df), dtype=pl.Boolean)

        registry = {
            "EMA": AlwaysTrueEma(),
            "RSI": AlwaysTrueRsi(),
            "MACD": MacdFactor(),
            "VWAP": VwapFactor(),
            "ATR": AtrVolatilityFactor(),
        }
        df = _make_ohlcv(300)
        lb, _ = run_factor_research(
            df,
            output_dir=tmp_path,
            combinations=(("EMA", "RSI"),),
            registry=registry,
        )
        assert lb.height == 1

    def test_fixed_bars_parameter_respected(self, tmp_path: Path) -> None:
        """Engine should pass fixed_bars through to the backtester."""
        df = _make_ohlcv(300)
        _, summary = run_factor_research(df, output_dir=tmp_path, fixed_bars=5)
        if summary.height > 0:
            assert int(summary["bars_held"].max()) <= 5

    def test_allows_explicit_expected_zero_trade_combination(self, tmp_path: Path) -> None:
        """Expected zero-trade combinations should not raise."""
        df = _make_ohlcv(300)
        run_factor_research(
            df,
            output_dir=tmp_path,
            combinations=(("EMA", "VWAP"),),
            expected_zero_trade_combinations=("EMA AND VWAP",),
        )

    def test_unexpected_zero_trade_combination_raises(self, tmp_path: Path) -> None:
        """Unexpected zero-trade combinations should fail loudly."""

        class AlwaysFalseVwap(VwapFactor):
            def signal(self, df: pl.DataFrame) -> pl.Series:
                return pl.Series([False] * len(df), dtype=pl.Boolean)

        registry = {
            "EMA": EmaTrendFactor(),
            "RSI": RsiFactor(),
            "MACD": MacdFactor(),
            "VWAP": AlwaysFalseVwap(),
            "ATR": AtrVolatilityFactor(),
        }
        df = _make_ohlcv(300)
        with pytest.raises(ValueError, match="Unexpected zero-trade result"):
            run_factor_research(
                df,
                output_dir=tmp_path,
                combinations=(("EMA", "VWAP"),),
                registry=registry,
                fail_on_unexpected_zero_trades=True,
            )


class TestFactorIntegrationOnNiftyBankSample:
    """Integration checks on the bundled NIFTY BANK-like sample dataset."""

    def test_every_registered_factor_has_at_least_one_signal(self) -> None:
        df = _make_nifty_bank_sample()
        for key, factor in FACTOR_REGISTRY.items():
            enriched = factor.compute(df)
            signal_count = int(factor.signal(enriched).fill_null(False).sum())
            assert signal_count > 0, f"{key} produced zero bullish signals"

    def test_all_combinations_execute_and_have_non_zero_trades(self, tmp_path: Path) -> None:
        df = _make_nifty_bank_sample()
        leaderboard, _ = run_factor_research(
            df, output_dir=tmp_path, fail_on_unexpected_zero_trades=True
        )
        assert leaderboard.height == len(COMBINATIONS)
        assert "EMA" in set(leaderboard["factor_combination"].to_list())
        assert leaderboard.filter(pl.col("number_of_trades") == 0).height == 0


# ---------------------------------------------------------------------------
# CLI integration (scripts/research_lab.py)
# ---------------------------------------------------------------------------


class TestResearchLabFactorsMode:
    """Verify that --mode factors works via the run_factors helper."""

    def test_run_factors_writes_reports(self, tmp_path: Path) -> None:
        import importlib.util
        from types import ModuleType

        script_path = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"
        spec = importlib.util.spec_from_file_location("research_lab_script", script_path)
        assert spec and spec.loader
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        data_path = tmp_path / "input.parquet"
        _make_ohlcv(300).write_parquet(data_path)

        out_dir = tmp_path / "factors"
        leaderboard, summary = module.run_factors(data_path, output_dir=out_dir)
        assert (out_dir / "leaderboard.csv").exists()
        assert (out_dir / "summary.csv").exists()
        assert leaderboard.height == len(COMBINATIONS)
