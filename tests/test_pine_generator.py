"""Tests for the Pine Script generator (apex_lab.export)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apex_lab.export.pine_generator import PineGenerator, generate_pine_strategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BEST_PARAMS: dict[str, Any] = {
    "fast_ema": 50,
    "slow_ema": 200,
    "mean_profit_factor": 1.42,
    "mean_expectancy": 0.03,
    "mean_win_rate": 0.53,
    "mean_drawdown": -8.7,
}

SAMPLE_BEST_FEATURES: dict[str, Any] = {
    "ema200_side": "Below EMA200",
    "vwap_side": "Above VWAP",
    "atr_state": "Neutral",
    "gap_pct_bucket": "Flat (-0.3% to 0.3%)",
    "hour": "01",
    "day_of_week": "Friday",
    "ema200_slope": "Falling (<-0.05%)",
    "or_position": "Above OR High",
    "inside_or": "Outside Opening Range",
}

SAMPLE_WEIGHTS_DATA: dict[str, Any] = {
    "normalization": {"method": "sum_to_100_with_last_row_adjustment"},
    "weights": [
        {"feature": "dist_ema200", "bucket": "-2% to 0%", "category": "trend", "weight": 10.0, "source_score": 0.5},
        {"feature": "dist_ema50", "bucket": "0% to 1%", "category": "trend", "weight": 10.0, "source_score": 0.5},
        {"feature": "rsi_bucket", "bucket": "70-100", "category": "momentum", "weight": 15.0, "source_score": 0.6},
        {"feature": "adx_bucket", "bucket": "35+", "category": "momentum", "weight": 5.0, "source_score": 0.4},
        {"feature": "atr_state", "bucket": "Neutral", "category": "volatility", "weight": 10.0, "source_score": 0.5},
        {"feature": "vwap_side", "bucket": "Above VWAP", "category": "vwap", "weight": 15.0, "source_score": 0.6},
        {"feature": "higher_high", "bucket": "Higher High", "category": "market_structure", "weight": 10.0, "source_score": 0.5},
        {"feature": "or_position", "bucket": "Above OR High", "category": "opening_range", "weight": 15.0, "source_score": 0.6},
        {"feature": "hour", "bucket": "01", "category": "time", "weight": 10.0, "source_score": 0.5},
    ],
}


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    """Create a temporary directory tree mimicking reports/lab."""
    walkforward_dir = tmp_path / "reports" / "lab" / "walkforward"
    context_dir = tmp_path / "reports" / "lab" / "context"
    alpha_dir = tmp_path / "reports" / "lab" / "alpha"

    for d in (walkforward_dir, context_dir, alpha_dir):
        d.mkdir(parents=True)

    (walkforward_dir / "best_parameters.json").write_text(
        json.dumps(SAMPLE_BEST_PARAMS), encoding="utf-8"
    )
    (context_dir / "best_features.json").write_text(
        json.dumps(SAMPLE_BEST_FEATURES), encoding="utf-8"
    )
    (alpha_dir / "weights.json").write_text(
        json.dumps(SAMPLE_WEIGHTS_DATA), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Return a fresh output directory."""
    return tmp_path / "generated"


@pytest.fixture()
def generated_outputs(report_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Run the generator and return (pine_path, summary_path)."""
    pine_path, summary_path = generate_pine_strategy(
        walkforward_params_path=report_dir / "reports/lab/walkforward/best_parameters.json",
        context_features_path=report_dir / "reports/lab/context/best_features.json",
        alpha_weights_path=report_dir / "reports/lab/alpha/weights.json",
        output_dir=output_dir,
    )
    return pine_path, summary_path


# ---------------------------------------------------------------------------
# File-existence tests
# ---------------------------------------------------------------------------


def test_strategy_pine_is_created(generated_outputs: tuple[Path, Path]) -> None:
    """strategy.pine must exist after generation."""
    pine_path, _ = generated_outputs
    assert pine_path.exists()
    assert pine_path.name == "strategy.pine"


def test_strategy_summary_json_is_created(generated_outputs: tuple[Path, Path]) -> None:
    """strategy_summary.json must exist after generation."""
    _, summary_path = generated_outputs
    assert summary_path.exists()
    assert summary_path.name == "strategy_summary.json"


# ---------------------------------------------------------------------------
# Pine Script content tests
# ---------------------------------------------------------------------------


def test_pine_version_header_exists(generated_outputs: tuple[Path, Path]) -> None:
    """The script must start with //@version=5."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "//@version=5" in content


def test_strategy_declaration_exists(generated_outputs: tuple[Path, Path]) -> None:
    """strategy() declaration must be present."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert 'strategy(' in content
    assert '"Apex Research Strategy"' in content


def test_ta_ema_exists(generated_outputs: tuple[Path, Path]) -> None:
    """ta.ema() must appear in the generated script."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "ta.ema(" in content


def test_ta_vwap_exists(generated_outputs: tuple[Path, Path]) -> None:
    """ta.vwap() must appear in the generated script."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "ta.vwap(" in content


def test_ta_atr_exists(generated_outputs: tuple[Path, Path]) -> None:
    """ta.atr() must appear in the generated script."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "ta.atr(" in content


def test_ta_rsi_exists(generated_outputs: tuple[Path, Path]) -> None:
    """ta.rsi() must appear in the generated script."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "ta.rsi(" in content


def test_ta_macd_exists(generated_outputs: tuple[Path, Path]) -> None:
    """ta.macd() must appear in the generated script."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "ta.macd(" in content


def test_plot_statements_exist(generated_outputs: tuple[Path, Path]) -> None:
    """plot() statements must be present."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "plot(" in content


def test_alertcondition_exists(generated_outputs: tuple[Path, Path]) -> None:
    """alertcondition() must be present."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    assert "alertcondition(" in content


def test_ema_defaults_from_walkforward(generated_outputs: tuple[Path, Path]) -> None:
    """EMA defaults should come from walk-forward best_parameters.json."""
    pine_path, _ = generated_outputs
    content = pine_path.read_text(encoding="utf-8")
    # fast_ema=50, slow_ema=200 from SAMPLE_BEST_PARAMS
    assert "input.int(50," in content or "input.int(50 " in content or "input.int(50," in content
    assert "input.int(200," in content or "input.int(200 " in content


# ---------------------------------------------------------------------------
# Summary JSON validation
# ---------------------------------------------------------------------------


def test_summary_json_validates(generated_outputs: tuple[Path, Path]) -> None:
    """strategy_summary.json must be valid JSON with required top-level keys."""
    _, summary_path = generated_outputs
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    required_keys = {
        "generation_timestamp",
        "generator_version",
        "research_version",
        "pine_version",
        "ema_parameters",
        "context_filters",
        "alpha_weights",
        "input_files_used",
    }
    for key in required_keys:
        assert key in summary, f"Missing key in summary: {key}"

    assert summary["pine_version"] == "5"
    assert isinstance(summary["ema_parameters"], dict)
    assert summary["ema_parameters"]["fast"] == 50
    assert summary["ema_parameters"]["slow"] == 200
    assert isinstance(summary["context_filters"], list)
    assert isinstance(summary["alpha_weights"], dict)
    assert isinstance(summary["input_files_used"], list)
    assert len(summary["input_files_used"]) == 3


# ---------------------------------------------------------------------------
# Missing input file tests
# ---------------------------------------------------------------------------


def test_missing_walkforward_params_raises_runtime_error(tmp_path: Path) -> None:
    """RuntimeError with clear message when best_parameters.json is absent."""
    context_dir = tmp_path / "context"
    alpha_dir = tmp_path / "alpha"
    context_dir.mkdir()
    alpha_dir.mkdir()
    (context_dir / "best_features.json").write_text(json.dumps(SAMPLE_BEST_FEATURES), encoding="utf-8")
    (alpha_dir / "weights.json").write_text(json.dumps(SAMPLE_WEIGHTS_DATA), encoding="utf-8")

    with pytest.raises(RuntimeError, match="best_parameters.json"):
        generate_pine_strategy(
            walkforward_params_path=tmp_path / "missing" / "best_parameters.json",
            context_features_path=context_dir / "best_features.json",
            alpha_weights_path=alpha_dir / "weights.json",
            output_dir=tmp_path / "generated",
        )


def test_missing_context_features_raises_runtime_error(tmp_path: Path) -> None:
    """RuntimeError with clear message when best_features.json is absent."""
    walkforward_dir = tmp_path / "walkforward"
    alpha_dir = tmp_path / "alpha"
    walkforward_dir.mkdir()
    alpha_dir.mkdir()
    (walkforward_dir / "best_parameters.json").write_text(json.dumps(SAMPLE_BEST_PARAMS), encoding="utf-8")
    (alpha_dir / "weights.json").write_text(json.dumps(SAMPLE_WEIGHTS_DATA), encoding="utf-8")

    with pytest.raises(RuntimeError, match="best_features.json"):
        generate_pine_strategy(
            walkforward_params_path=walkforward_dir / "best_parameters.json",
            context_features_path=tmp_path / "missing" / "best_features.json",
            alpha_weights_path=alpha_dir / "weights.json",
            output_dir=tmp_path / "generated",
        )


def test_missing_alpha_weights_raises_runtime_error(tmp_path: Path) -> None:
    """RuntimeError with clear message when weights.json is absent."""
    walkforward_dir = tmp_path / "walkforward"
    context_dir = tmp_path / "context"
    walkforward_dir.mkdir()
    context_dir.mkdir()
    (walkforward_dir / "best_parameters.json").write_text(json.dumps(SAMPLE_BEST_PARAMS), encoding="utf-8")
    (context_dir / "best_features.json").write_text(json.dumps(SAMPLE_BEST_FEATURES), encoding="utf-8")

    with pytest.raises(RuntimeError, match="weights.json"):
        generate_pine_strategy(
            walkforward_params_path=walkforward_dir / "best_parameters.json",
            context_features_path=context_dir / "best_features.json",
            alpha_weights_path=tmp_path / "missing" / "weights.json",
            output_dir=tmp_path / "generated",
        )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


def test_generator_is_deterministic(report_dir: Path, tmp_path: Path) -> None:
    """Running the generator twice on identical inputs produces identical output."""
    params = report_dir / "reports/lab/walkforward/best_parameters.json"
    features = report_dir / "reports/lab/context/best_features.json"
    weights = report_dir / "reports/lab/alpha/weights.json"

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    pine1, _ = generate_pine_strategy(params, features, weights, out1)
    pine2, _ = generate_pine_strategy(params, features, weights, out2)

    assert pine1.read_text(encoding="utf-8") == pine2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# PineGenerator class API
# ---------------------------------------------------------------------------


def test_pine_generator_class_returns_paths(report_dir: Path, output_dir: Path) -> None:
    """PineGenerator.generate() returns valid (pine_path, summary_path) tuple."""
    gen = PineGenerator(
        walkforward_params_path=report_dir / "reports/lab/walkforward/best_parameters.json",
        context_features_path=report_dir / "reports/lab/context/best_features.json",
        alpha_weights_path=report_dir / "reports/lab/alpha/weights.json",
        output_dir=output_dir,
    )
    pine_path, summary_path = gen.generate()
    assert pine_path.exists()
    assert summary_path.exists()
