"""Tests for objective reversal labeling engine."""

from __future__ import annotations

import polars as pl
import pytest

from apex_lab.labels import LabelEngine, LabelingRules, LabelType, evaluate_labels


@pytest.fixture()
def base_rules() -> LabelingRules:
    """Return compact test rules for deterministic fixtures."""
    return LabelingRules(atr_column="atr", atr_multiplier=1.0, reward_multiplier=2.0, risk_multiplier=1.0, lookahead_window=4)


def test_bottom_detection(base_rules: LabelingRules) -> None:
    """Label is BOTTOM when reward is hit before risk."""
    df = pl.DataFrame(
        {
            "high": [101.0, 102.5, 101.8, 101.7, 101.6],
            "low": [100.0, 100.2, 100.1, 100.0, 99.9],
            "close": [100.5, 101.0, 101.0, 100.8, 100.7],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    out = LabelEngine(base_rules).label(df)

    assert out["label"][0] == LabelType.BOTTOM.value
    assert out["bars_to_target"][0] == 1


def test_top_detection(base_rules: LabelingRules) -> None:
    """Label is TOP when downside reward is hit before upside risk."""
    df = pl.DataFrame(
        {
            "high": [110.0, 110.2, 109.8, 109.7, 109.6],
            "low": [109.0, 107.5, 108.2, 108.1, 108.0],
            "close": [109.5, 108.2, 108.5, 108.6, 108.7],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    out = LabelEngine(base_rules).label(df)

    assert out["label"][0] == LabelType.TOP.value
    assert out["bars_to_target"][0] == 1


def test_no_future_leakage_beyond_lookahead() -> None:
    """Bars outside lookahead window must not affect earlier labels."""
    rules = LabelingRules(atr_column="atr", lookahead_window=12)
    base = pl.DataFrame(
        {
            "high": [101.0] + [101.1] * 18,
            "low": [100.0] + [99.8] * 18,
            "close": [100.5] + [100.0] * 18,
            "atr": [1.0] * 19,
        }
    )
    changed = base.with_columns(pl.when(pl.arange(0, pl.len()) == 15).then(120.0).otherwise(pl.col("high")).alias("high"))

    base_label = LabelEngine(rules).label(base)["label"][0]
    changed_label = LabelEngine(rules).label(changed)["label"][0]

    assert base_label == LabelType.NONE.value
    assert changed_label == LabelType.NONE.value


def test_edge_cases_near_dataset_end(base_rules: LabelingRules) -> None:
    """Rows near dataset end should remain deterministic with null future metrics."""
    df = pl.DataFrame(
        {
            "high": [101.0, 101.2, 101.1, 101.0, 100.9],
            "low": [100.0, 100.1, 100.0, 99.9, 99.8],
            "close": [100.5, 100.7, 100.6, 100.5, 100.4],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    out = LabelEngine(base_rules).label(df)

    assert out["label"][-1] == LabelType.NONE.value
    assert out["bars_to_target"][-1] is None
    assert out["future_return"][-1] is None


def test_reward_multiplier_parameter_changes_labels() -> None:
    """Changing reward multiplier should change label assignments."""
    df = pl.DataFrame(
        {
            "high": [101.0, 102.6, 102.4, 102.2, 102.0],
            "low": [100.0, 100.1, 100.0, 99.9, 99.8],
            "close": [100.5, 101.9, 101.8, 101.7, 101.6],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    low_threshold = LabelEngine(LabelingRules(atr_column="atr", reward_multiplier=2.0, lookahead_window=4)).label(df)
    high_threshold = LabelEngine(LabelingRules(atr_column="atr", reward_multiplier=3.0, lookahead_window=4)).label(df)

    assert low_threshold["label"][0] == LabelType.BOTTOM.value
    assert high_threshold["label"][0] == LabelType.NONE.value


def test_evaluator_statistics() -> None:
    """Evaluator should return expected aggregate stats."""
    df = pl.DataFrame(
        {
            "label": [LabelType.BOTTOM.value, LabelType.TOP.value, LabelType.NONE.value],
            "future_return": [0.04, -0.03, None],
        }
    )

    stats = evaluate_labels(df)

    assert stats.total_rows == 3
    assert stats.total_labels == 2
    assert stats.class_balance[LabelType.BOTTOM.value] == 1
    assert stats.class_balance[LabelType.TOP.value] == 1
    assert stats.average_move == pytest.approx(0.035, rel=1e-6)
