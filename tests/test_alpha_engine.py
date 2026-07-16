"""Integration tests for the Alpha Scoring Engine."""

from __future__ import annotations

import datetime
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import polars as pl

from apex_lab.research.alpha.engine import run_alpha_scoring
from apex_lab.research.context.engine import run_context_research

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_nifty_bank_sample(n: int = 2000) -> pl.DataFrame:
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


def test_alpha_engine_generates_all_reports_and_constraints(tmp_path: Path) -> None:
    df = _make_nifty_bank_sample()
    context_dir = tmp_path / "context"
    run_context_research(df, output_dir=context_dir)

    alpha_dir = tmp_path / "alpha"
    result = run_alpha_scoring(
        df,
        output_dir=alpha_dir,
        context_leaderboard_path=context_dir / "leaderboard.csv",
    )

    expected_files = {
        "weights.json",
        "scores.csv",
        "score_analysis.csv",
        "score_validation.json",
        "top_trades.csv",
        "bottom_trades.csv",
        "alpha_summary.json",
    }
    assert expected_files.issubset({path.name for path in alpha_dir.iterdir()})

    assert result.scores.height > 0
    assert result.scores.height == result.scores.select(["alpha_score"]).height

    alpha_scores = result.scores.get_column("alpha_score")
    assert alpha_scores.min() >= 0.0
    assert alpha_scores.max() <= 100.0

    weights_payload = json.loads((alpha_dir / "weights.json").read_text(encoding="utf-8"))
    weight_sum = sum(float(row["weight"]) for row in weights_payload["weights"])
    assert weight_sum == 100.0

    required_columns = [
        "entry_time",
        "exit_time",
        "alpha_score",
        "trend_score",
        "momentum_score",
        "volatility_score",
        "vwap_score",
        "market_structure_score",
        "opening_range_score",
        "time_score",
        "return_pct",
    ]
    for column in required_columns:
        assert result.scores.get_column(column).null_count() == 0

    assert result.score_analysis.height == 5
    assert set(result.score_analysis.columns) == {
        "score_bucket",
        "number_of_trades",
        "win_rate",
        "average_return",
        "median_return",
        "expectancy",
        "profit_factor",
        "sharpe",
        "maximum_drawdown",
    }

    assert set(result.validation.keys()) == {
        "pearson",
        "spearman",
        "highest_bucket_expectancy",
        "lowest_bucket_expectancy",
        "monotonicity",
    }
    assert isinstance(result.validation["monotonicity"], bool)

    if result.top_trades.height > 1:
        top_scores = result.top_trades.get_column("alpha_score").to_list()
        assert top_scores == sorted(top_scores, reverse=True)

    if result.bottom_trades.height > 1:
        bottom_scores = result.bottom_trades.get_column("alpha_score").to_list()
        assert bottom_scores == sorted(bottom_scores)


def test_research_lab_alpha_mode_wrapper_runs(tmp_path: Path) -> None:
    module = _load_script_module()
    data_path = tmp_path / "input.parquet"
    _make_nifty_bank_sample().write_parquet(data_path)

    output_dir = tmp_path / "alpha"
    scores, score_analysis, score_validation, top_trades, bottom_trades = module.run_alpha(
        data_path,
        output_dir=output_dir,
    )

    assert scores.height > 0
    assert score_analysis.height == 5
    assert isinstance(score_validation, dict)
    assert "monotonicity" in score_validation
    assert top_trades.height <= 20
    assert bottom_trades.height <= 20
