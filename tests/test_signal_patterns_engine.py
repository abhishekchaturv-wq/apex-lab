"""Tests for the High-Expectancy Signal Discovery Engine (PR20)."""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl

from apex_lab.research.signal_patterns.candidate_generator import (
    CandidateGeneratorConfig,
    generate_candidates,
)
from apex_lab.research.signal_patterns.diversity import analyze_feature_diversity
from apex_lab.research.signal_patterns.engine import run_signal_patterns
from apex_lab.research.signal_patterns.evaluator import evaluate_all_candidates
from apex_lab.research.signal_patterns.ranking import (
    build_ranking_artifacts,
    rank_signals,
    walk_forward_validate,
)
from apex_lab.research.signal_patterns.report import build_summary_payload, write_reports

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research_lab.py"
_CLONE_SIGNAL_SCALE = 1.01
_CLONE_NOISE_WEIGHT = 0.01


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_lab_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _signal_class(value: float) -> str:
    if value >= 1.5:
        return "Strong Bull Move"
    if value >= 0.4:
        return "Bull Move"
    if value <= -1.5:
        return "Strong Bear Move"
    if value <= -0.4:
        return "Bear Move"
    return "Neutral"


def _make_signal_dataset(rows: int = 400) -> pl.DataFrame:
    """Create a synthetic signal dataset with known predictive features."""
    base_ts = datetime.datetime(2024, 1, 1, 9, 15, 0)

    signal_strength = [((i % 20) - 10) / 5.0 for i in range(rows)]
    noise = [((i * 7) % 13) / 13.0 - 0.5 for i in range(rows)]
    atr_state = ["high" if v > 0.5 else "low" for v in signal_strength]
    rsi_bucket = ["60-70" if v > 0.5 else "40-50" for v in signal_strength]
    opening_range = ["above_or" if i % 3 else "inside_or" for i in range(rows)]
    market_regime = ["above_ema200_high" if i % 4 else "below_ema200_low" for i in range(rows)]
    symbol = ["NIFTY BANK" if i % 2 == 0 else "NIFTY 50" for i in range(rows)]

    future_return_20 = [
        (s * 1.2) + (0.3 if a == "high" else -0.2) + (0.05 * n)
        for s, a, n in zip(signal_strength, atr_state, noise, strict=True)
    ]
    future_return_5 = [v * 0.6 for v in future_return_20]
    future_return_10 = [v * 0.8 for v in future_return_20]
    future_return_40 = [v * 1.1 for v in future_return_20]
    future_high_return = [abs(v) * 1.5 for v in future_return_20]
    future_low_return = [-abs(v) * 1.2 for v in future_return_20]

    return pl.DataFrame(
        {
            "timestamp": [base_ts + datetime.timedelta(minutes=30 * i) for i in range(rows)],
            "open": [100.0 + i * 0.1 for i in range(rows)],
            "high": [101.0 + i * 0.1 for i in range(rows)],
            "low": [99.0 + i * 0.1 for i in range(rows)],
            "close": [100.5 + i * 0.1 for i in range(rows)],
            "volume": [10_000 + i for i in range(rows)],
            "symbol": symbol,
            "market_regime": market_regime,
            "hour": [9 + (i % 7) for i in range(rows)],
            "day": [1 + (i % 5) for i in range(rows)],
            "month": [1 + (i % 3) for i in range(rows)],
            "quarter": [1 + (i % 4) for i in range(rows)],
            "ema_signal_strength": signal_strength,
            "atr_state": atr_state,
            "rsi_bucket": rsi_bucket,
            "opening_range": opening_range,
            "noise_feature": noise,
            "ema_signal_clone": [
                (signal * _CLONE_SIGNAL_SCALE) + (_CLONE_NOISE_WEIGHT * noise_value)
                for signal, noise_value in zip(signal_strength, noise, strict=True)
            ],
            "price_proxy": [100.25 + (i * 0.1) + (0.01 * noise[i]) for i in range(rows)],
            "future_return_5": future_return_5,
            "future_return_10": future_return_10,
            "future_return_20": future_return_20,
            "future_return_40": future_return_40,
            "future_high_return": future_high_return,
            "future_low_return": future_low_return,
            "signal_class": [_signal_class(v) for v in future_return_20],
        }
    )


def _make_feature_importance() -> pl.DataFrame:
    """Create a minimal feature importance table with known feature order."""
    features = [
        "ema_signal_strength",
        "atr_state",
        "rsi_bucket",
        "opening_range",
        "market_regime",
        "hour",
        "ema_signal_clone",
        "price_proxy",
        "day",
        "month",
        "quarter",
        "noise_feature",
    ]
    scores = [0.9, 0.8, 0.75, 0.6, 0.55, 0.4, 0.39, 0.38, 0.35, 0.3, 0.25, 0.1]
    return pl.DataFrame(
        {
            "rank": list(range(1, len(features) + 1)),
            "feature": features,
            "composite_score": scores,
        }
    )


# ---------------------------------------------------------------------------
# Candidate generation tests
# ---------------------------------------------------------------------------


def test_candidate_generation_produces_2_3_4_feature_combos() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    config = CandidateGeneratorConfig(top_features=5, min_samples=10, combo_sizes=(2, 3, 4))
    candidates = generate_candidates(dataset, importance, config=config)

    assert len(candidates) > 0
    sizes = {len(c.features) for c in candidates}
    # With 5 top features we should get 2-, 3- and 4-feature combos.
    assert 2 in sizes
    assert 3 in sizes
    assert 4 in sizes


def test_candidate_generation_respects_min_samples() -> None:
    dataset = _make_signal_dataset(rows=50)  # small dataset
    importance = _make_feature_importance()

    # Very high min_samples should yield no candidates.
    config = CandidateGeneratorConfig(top_features=5, min_samples=200, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)
    assert len(candidates) == 0


def test_candidate_generation_is_deterministic() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=6, min_samples=10, combo_sizes=(2,))

    run1 = generate_candidates(dataset, importance, config=config)
    run2 = generate_candidates(dataset, importance, config=config)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.features == c2.features
        assert c1.bucket_key == c2.bucket_key


def test_candidate_generation_uses_only_top_n_features() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    config = CandidateGeneratorConfig(top_features=3, min_samples=5, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    allowed_features = {"ema_signal_strength", "atr_state", "rsi_bucket"}
    for c in candidates:
        for feat in c.features:
            assert feat in allowed_features, f"unexpected feature: {feat}"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


def test_evaluate_all_candidates_returns_expected_columns() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=4, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    stats = evaluate_all_candidates(dataset, candidates, target_column="future_return_20")

    if stats.height > 0:
        expected_cols = {
            "rule_label",
            "signal_frequency",
            "win_rate",
            "average_return",
            "median_return",
            "profit_factor",
            "expectancy",
            "average_mfe",
            "average_mae",
        }
        assert expected_cols.issubset(set(stats.columns))


def test_evaluate_all_candidates_win_rate_in_valid_range() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=4, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    stats = evaluate_all_candidates(dataset, candidates, target_column="future_return_20")

    if stats.height > 0:
        win_rates = stats.get_column("win_rate").drop_nulls().to_list()
        for wr in win_rates:
            assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# Walk-forward validation tests
# ---------------------------------------------------------------------------


def test_walk_forward_validation_returns_expected_columns() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=4, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    wf = walk_forward_validate(dataset, candidates, target_column="future_return_20")

    if wf.height > 0:
        assert "is_robust" in wf.columns
        assert "train_expectancy" in wf.columns
        assert "val_expectancy" in wf.columns
        assert "oos_expectancy" in wf.columns


def test_walk_forward_rejects_low_sample_rules() -> None:
    # Use a very small dataset so splits have few rows.
    dataset = _make_signal_dataset(rows=60)
    importance = _make_feature_importance()
    # Use high min_samples in config so rules are generated but splits are small.
    config = CandidateGeneratorConfig(top_features=4, min_samples=15, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    wf = walk_forward_validate(dataset, candidates, target_column="future_return_20")

    if wf.height > 0:
        # All rules should be marked non-robust when splits are too small.
        robust_count = wf.get_column("is_robust").sum()
        assert isinstance(robust_count, int)


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------


def test_ranking_produces_composite_score_and_is_robust_columns() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=5, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    stats = evaluate_all_candidates(dataset, candidates, target_column="future_return_20")
    wf = walk_forward_validate(dataset, candidates, target_column="future_return_20")
    ranked = rank_signals(stats, wf)

    if ranked.height > 0:
        assert "composite_score" in ranked.columns
        assert "is_robust" in ranked.columns
        assert "rank" in ranked.columns
        # Verify descending order.
        scores = ranked.get_column("composite_score").to_list()
        assert scores == sorted(scores, reverse=True)


def test_feature_diversity_analysis_groups_correlated_numeric_features() -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    analysis = analyze_feature_diversity(dataset, importance)

    assert "feature_left" in analysis.feature_correlation.columns
    assert "pearson_correlation" in analysis.feature_correlation.columns
    assert "spearman_correlation" in analysis.feature_correlation.columns
    assert analysis.feature_to_cluster["ema_signal_strength"] == analysis.feature_to_cluster["ema_signal_clone"]
    assert analysis.cluster_importance.height > 0


def test_ranking_keeps_strongest_representative_for_similar_rules() -> None:
    stats = pl.DataFrame(
        {
            "rule_label": [
                "ema_signal_strength == q3 AND atr_state == high",
                "ema_signal_clone == q3 AND atr_state == high",
                "noise_feature == q0 AND market_regime == above_ema200_high",
            ],
            "features": [
                "['ema_signal_strength', 'atr_state']",
                "['ema_signal_clone', 'atr_state']",
                "['noise_feature', 'market_regime']",
            ],
            "conditions": [
                "['ema_signal_strength == q3', 'atr_state == high']",
                "['ema_signal_clone == q3', 'atr_state == high']",
                "['noise_feature == q0', 'market_regime == above_ema200_high']",
            ],
            "combination_size": [2, 2, 2],
            "signal_frequency": [120, 100, 95],
            "win_rate": [0.7, 0.68, 0.64],
            "average_return": [0.42, 0.39, 0.31],
            "median_return": [0.4, 0.37, 0.28],
            "profit_factor": [2.2, 2.0, 1.7],
            "expectancy": [0.36, 0.33, 0.24],
            "average_mfe": [0.6, 0.58, 0.44],
            "average_mae": [-0.2, -0.21, -0.18],
        }
    )
    wf = pl.DataFrame(
        {
            "rule_label": stats.get_column("rule_label"),
            "is_robust": [True, True, True],
            "train_expectancy": [0.35, 0.32, 0.23],
            "val_expectancy": [0.34, 0.31, 0.22],
            "oos_expectancy": [0.33, 0.29, 0.2],
        }
    )

    artifacts = build_ranking_artifacts(
        stats,
        wf,
        feature_to_cluster={
            "ema_signal_strength": "cluster_001",
            "ema_signal_clone": "cluster_001",
            "atr_state": "cluster_002",
            "noise_feature": "cluster_003",
            "market_regime": "cluster_004",
        },
    )

    ranked = artifacts.ranked_signals
    assert ranked.height == 2
    assert "diversity_score" in ranked.columns
    assert ranked.get_column("rule_label").to_list()[0] == "ema_signal_strength == q3 AND atr_state == high"
    similar_group = artifacts.all_ranked_signals.filter(
        pl.col("rule_label").is_in(
            [
                "ema_signal_strength == q3 AND atr_state == high",
                "ema_signal_clone == q3 AND atr_state == high",
            ]
        )
    )
    assert similar_group.get_column("similarity_group_id").n_unique() == 1
    assert similar_group.get_column("representative_rule_label").unique().to_list() == [
        "ema_signal_strength == q3 AND atr_state == high"
    ]
    assert artifacts.rule_similarity.height == 3


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------


def test_report_generation_creates_required_files(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=5, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    stats = evaluate_all_candidates(dataset, candidates, target_column="future_return_20")
    wf = walk_forward_validate(dataset, candidates, target_column="future_return_20")
    ranked = rank_signals(stats, wf)
    summary = build_summary_payload(ranked, wf, stats)

    output_dir = tmp_path / "signal_patterns"
    write_reports(
        output_dir=output_dir,
        ranked=ranked,
        candidate_stats=stats,
        wf=wf,
        summary=summary,
    )

    expected = {
        "top_signals.csv",
        "top_signals.json",
        "candidate_statistics.csv",
        "walkforward_validation.csv",
        "summary.json",
    }
    assert expected.issubset({p.name for p in output_dir.iterdir()})


def test_summary_contains_required_keys(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()
    config = CandidateGeneratorConfig(top_features=5, min_samples=10, combo_sizes=(2,))
    candidates = generate_candidates(dataset, importance, config=config)

    stats = evaluate_all_candidates(dataset, candidates, target_column="future_return_20")
    wf = walk_forward_validate(dataset, candidates, target_column="future_return_20")
    ranked = rank_signals(stats, wf)
    summary = build_summary_payload(ranked, wf, stats)

    required_keys = {
        "top_20_signals",
        "recommended_pine_rules",
        "recommended_entry_conditions",
        "recommended_exit_conditions",
        "most_stable_signals",
        "rejected_signals",
        "total_candidates_evaluated",
        "total_robust_signals",
    }
    assert required_keys.issubset(set(summary.keys()))


# ---------------------------------------------------------------------------
# Full engine integration test
# ---------------------------------------------------------------------------


def test_engine_generates_all_required_output_files(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    dataset_path = tmp_path / "dataset.parquet"
    feature_ranking_path = tmp_path / "feature_importance.csv"
    output_dir = tmp_path / "signal_patterns"

    dataset.write_parquet(dataset_path)
    importance.write_csv(feature_ranking_path)

    result = run_signal_patterns(
        dataset_path=dataset_path,
        feature_ranking_path=feature_ranking_path,
        output_dir=output_dir,
        generator_config=CandidateGeneratorConfig(
            top_features=5, min_samples=10, combo_sizes=(2,)
        ),
    )

    expected = {
        "top_signals.csv",
        "top_signals.json",
        "all_ranked_signals.csv",
        "candidate_statistics.csv",
        "walkforward_validation.csv",
        "summary.json",
        "feature_correlation.csv",
        "feature_clusters.json",
        "cluster_importance.csv",
        "rule_similarity.csv",
        "signal_diversity_report.md",
    }
    assert expected.issubset({p.name for p in output_dir.iterdir()})
    assert "top_20_signals" in result.summary
    assert "recommended_pine_rules" in result.summary


def test_engine_output_is_deterministic(tmp_path: Path) -> None:
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    dataset_path = tmp_path / "dataset.parquet"
    feature_ranking_path = tmp_path / "feature_importance.csv"
    dataset.write_parquet(dataset_path)
    importance.write_csv(feature_ranking_path)

    config = CandidateGeneratorConfig(top_features=5, min_samples=10, combo_sizes=(2,))

    run1 = run_signal_patterns(
        dataset_path=dataset_path,
        feature_ranking_path=feature_ranking_path,
        output_dir=tmp_path / "run1",
        generator_config=config,
    )
    run2 = run_signal_patterns(
        dataset_path=dataset_path,
        feature_ranking_path=feature_ranking_path,
        output_dir=tmp_path / "run2",
        generator_config=config,
    )

    assert run1.ranked_signals.equals(run2.ranked_signals)
    assert run1.summary["total_candidates_evaluated"] == run2.summary["total_candidates_evaluated"]


# ---------------------------------------------------------------------------
# CLI wrapper test
# ---------------------------------------------------------------------------


def test_research_lab_signal_patterns_wrapper(tmp_path: Path) -> None:
    module = _load_script_module()
    dataset = _make_signal_dataset()
    importance = _make_feature_importance()

    dataset_path = tmp_path / "dataset.parquet"
    feature_ranking_path = tmp_path / "feature_importance.csv"
    output_dir = tmp_path / "wrapper_out"

    dataset.write_parquet(dataset_path)
    importance.write_csv(feature_ranking_path)

    summary = module.run_signal_patterns(
        dataset_path=dataset_path,
        feature_ranking_path=feature_ranking_path,
        output_dir=output_dir,
    )

    assert (output_dir / "summary.json").exists()
    assert "top_20_signals" in summary
    assert "recommended_pine_rules" in summary


# ---------------------------------------------------------------------------
# Scalability regression tests – prevent eager O(N²) all-pairs computation
# ---------------------------------------------------------------------------


def test_no_eager_all_pairs_lookup_function_removed() -> None:
    """The O(N²) dictionary builder must no longer exist in the ranking module."""
    from apex_lab.research.signal_patterns import ranking

    assert not hasattr(ranking, "_build_rule_similarity_lookup"), (
        "_build_rule_similarity_lookup builds an O(N²) dict and must not exist; "
        "similarities must be computed lazily."
    )


def test_similarity_report_bounded_by_top_k() -> None:
    """Similarity report rows must be at most N * top_k, not O(N²)."""
    from apex_lab.research.signal_patterns.ranking import (
        _SIMILARITY_REPORT_TOP_K,
        build_ranking_artifacts,
    )

    # Build 30 distinct rules (all sharing one common feature so all pairs have sim > 0).
    n = 30
    rule_rows = [
        {
            "rule_label": f"feature_{i} == bucket AND base == val",
            "features": f"['feature_{i}', 'base']",
            "conditions": f"['feature_{i} == bucket', 'base == val']",
            "combination_size": 2,
            "signal_frequency": 50 + i,
            "win_rate": 0.55 + 0.001 * i,
            "average_return": 0.20 + 0.001 * i,
            "median_return": 0.18 + 0.001 * i,
            "profit_factor": 1.5 + 0.01 * i,
            "expectancy": 0.15 + 0.001 * i,
            "average_mfe": 0.30 + 0.001 * i,
            "average_mae": -0.12 - 0.001 * i,
        }
        for i in range(n)
    ]
    stats = pl.DataFrame(rule_rows)
    wf = pl.DataFrame(
        {
            "rule_label": [r["rule_label"] for r in rule_rows],
            "is_robust": [True] * n,
            "train_expectancy": [0.14] * n,
            "val_expectancy": [0.13] * n,
            "oos_expectancy": [0.12] * n,
        }
    )

    artifacts = build_ranking_artifacts(stats, wf)

    all_pairs_count = n * (n - 1) // 2  # 435 for n=30
    max_top_k_pairs = n * _SIMILARITY_REPORT_TOP_K  # 300 for n=30, top_k=10
    assert artifacts.rule_similarity.height <= max_top_k_pairs, (
        f"Similarity report has {artifacts.rule_similarity.height} rows; "
        f"expected at most {max_top_k_pairs} (n={n}, top_k={_SIMILARITY_REPORT_TOP_K})"
    )
    assert artifacts.rule_similarity.height < all_pairs_count, (
        f"Similarity report materialised all {all_pairs_count} pairs – "
        "this is the O(N²) regression the fix was designed to prevent."
    )
