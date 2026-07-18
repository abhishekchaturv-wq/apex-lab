"""Tests for signal-pattern driven Pine export."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from apex_lab.export.pine_generator import generate_pine_strategy
from apex_lab.export.rule_translator import FeatureTranslatorRegistry, RuleTranslator
from apex_lab.export.signal_pattern_loader import SignalPatternLoader

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_signal_pattern_fixture(tmp_path: Path) -> tuple[Path, Path]:
    signal_patterns_path = tmp_path / "top_signals.json"
    quantiles_path = tmp_path / "quantiles.json"

    signal_patterns_path.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "rule_label": "swing_high == q1 AND rsi == q2 AND ema_9 == q1 AND macd == q3",
                    "features": ["swing_high", "rsi", "ema_9", "macd"],
                    "conditions": [
                        "swing_high == q1",
                        "rsi == q2",
                        "ema_9 == q1",
                        "macd == q3",
                    ],
                    "combination_size": 4,
                    "signal_frequency": 284,
                    "win_rate": 1.0,
                    "average_return": 1.512,
                    "expectancy": 1.512,
                    "average_mfe": 2.597,
                    "average_mae": -0.0038,
                    "is_robust": True,
                    "diversity_score": 0.81,
                    "composite_score": 0.671691,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    quantiles_path.write_text(
        json.dumps(
            {
                "_meta": {"bins": 4, "label_scheme": "zero_based"},
                "swing_high": {
                    "q0": [90.0, 100.0],
                    "q1": [100.0, 110.0],
                    "q2": [110.0, 120.0],
                    "q3": [120.0, 130.0],
                },
                "rsi": {
                    "q0": [0.0, 30.0],
                    "q1": [30.0, 45.0],
                    "q2": [45.0, 60.0],
                    "q3": [60.0, 100.0],
                },
                "ema_9": {
                    "q0": [39500.0, 40000.0],
                    "q1": [40000.0, 40500.0],
                    "q2": [40500.0, 41000.0],
                    "q3": [41000.0, 41500.0],
                },
                "macd": {
                    "q0": [-100.0, -10.0],
                    "q1": [-10.0, 0.0],
                    "q2": [0.0, 10.0],
                    "q3": [10.0, 100.0],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return signal_patterns_path, quantiles_path


def test_signal_pattern_loader_reads_json_records(tmp_path: Path) -> None:
    signal_patterns_path, _ = _write_signal_pattern_fixture(tmp_path)

    pattern = SignalPatternLoader().load_top_signal(signal_patterns_path)

    assert pattern.rule_label.startswith("swing_high == q1")
    assert pattern.features == ("swing_high", "rsi", "ema_9", "macd")
    assert pattern.conditions[1] == "rsi == q2"
    assert pattern.signal_frequency == 284
    assert pattern.composite_score == pytest.approx(0.671691)


def test_rule_translator_decodes_quantile_conditions(tmp_path: Path) -> None:
    signal_patterns_path, quantiles_path = _write_signal_pattern_fixture(tmp_path)
    loader = SignalPatternLoader()
    translator = RuleTranslator()

    translated = translator.translate(
        loader.load_top_signal(signal_patterns_path),
        translator.load_quantiles(quantiles_path),
    )

    assert "(swingHighVal >= 100 and swingHighVal < 110)" in translated.entry_conditions
    assert "(rsiVal >= 45 and rsiVal < 60)" in translated.entry_conditions
    assert "ema9Val        = ta.ema(close, 9)" in translated.indicator_lines
    assert "[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)" in translated.indicator_lines


def test_feature_translator_registry_resolves_dynamic_ema() -> None:
    translator = FeatureTranslatorRegistry().resolve("ema_9")
    assert translator.pine_expression("ema_9") == "ema9Val"
    assert translator.indicator_lines("ema_9") == ("ema9Val        = ta.ema(close, 9)",)


def test_generate_pine_strategy_from_signal_patterns(tmp_path: Path) -> None:
    signal_patterns_path, quantiles_path = _write_signal_pattern_fixture(tmp_path)

    pine_path, summary_path = generate_pine_strategy(
        signal_patterns_path=signal_patterns_path,
        quantiles_path=quantiles_path,
        output_dir=tmp_path / "generated",
    )

    pine_script = pine_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert pine_path.exists()
    assert "// Generated by Apex Lab" in pine_script
    assert "strategy(" in pine_script
    assert 'title="Stop Loss %"' in pine_script
    assert 'title="Maximum Holding Bars"' in pine_script
    assert "label.new(bar_index, low" in pine_script
    assert "(rsiVal >= 45 and rsiVal < 60)" in pine_script
    assert summary["signal_pattern"]["rule_label"] == "swing_high == q1 AND rsi == q2 AND ema_9 == q1 AND macd == q3"


def test_research_lab_cli_pine_mode_accepts_signal_pattern_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    signal_patterns_path, quantiles_path = _write_signal_pattern_fixture(tmp_path)
    output_dir = tmp_path / "cli_output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research_lab.py",
            "--mode",
            "pine",
            "--signal-patterns",
            str(signal_patterns_path),
            "--quantiles",
            str(quantiles_path),
            "--output",
            str(output_dir),
        ],
    )

    module.main()

    assert (output_dir / "strategy.pine").exists()
    assert (output_dir / "strategy_summary.json").exists()


def test_missing_quantile_metadata_raises_descriptive_error(tmp_path: Path) -> None:
    signal_patterns_path, quantiles_path = _write_signal_pattern_fixture(tmp_path)
    quantiles = json.loads(quantiles_path.read_text(encoding="utf-8"))
    quantiles.pop("rsi")
    quantiles_path.write_text(json.dumps(quantiles), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing quantile metadata for feature 'rsi'"):
        generate_pine_strategy(
            signal_patterns_path=signal_patterns_path,
            quantiles_path=quantiles_path,
            output_dir=tmp_path / "generated",
        )


def test_unsupported_feature_raises_descriptive_error(tmp_path: Path) -> None:
    signal_patterns_path = tmp_path / "top_signals.json"
    quantiles_path = tmp_path / "quantiles.json"
    signal_patterns_path.write_text(
        json.dumps(
            [
                {
                    "rule_label": "adx == q1",
                    "conditions": ["adx == q1"],
                    "features": ["adx"],
                }
            ]
        ),
        encoding="utf-8",
    )
    quantiles_path.write_text(json.dumps({"adx": {"q1": [10.0, 20.0]}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported feature 'adx'"):
        generate_pine_strategy(
            signal_patterns_path=signal_patterns_path,
            quantiles_path=quantiles_path,
            output_dir=tmp_path / "generated",
        )


def test_invalid_operator_raises_descriptive_error(tmp_path: Path) -> None:
    signal_patterns_path, quantiles_path = _write_signal_pattern_fixture(tmp_path)
    payload = json.loads(signal_patterns_path.read_text(encoding="utf-8"))
    payload[0]["conditions"] = ["rsi > q2"]
    payload[0]["rule_label"] = "rsi > q2"
    signal_patterns_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported operator '>'"):
        generate_pine_strategy(
            signal_patterns_path=signal_patterns_path,
            quantiles_path=quantiles_path,
            output_dir=tmp_path / "generated",
        )


def test_invalid_rule_raises_descriptive_error(tmp_path: Path) -> None:
    signal_patterns_path = tmp_path / "top_signals.json"
    quantiles_path = tmp_path / "quantiles.json"
    signal_patterns_path.write_text(
        json.dumps([{"rule_label": "invalid rule", "conditions": ["invalid rule"]}]),
        encoding="utf-8",
    )
    quantiles_path.write_text(json.dumps({"rsi": {"q1": [10.0, 20.0]}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid rule condition"):
        generate_pine_strategy(
            signal_patterns_path=signal_patterns_path,
            quantiles_path=quantiles_path,
            output_dir=tmp_path / "generated",
        )
