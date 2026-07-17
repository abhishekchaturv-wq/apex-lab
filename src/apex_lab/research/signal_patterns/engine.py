"""High-Expectancy Signal Discovery Engine orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.signal_patterns.candidate_generator import (
    CandidateGeneratorConfig,
    generate_candidates,
)
from apex_lab.research.signal_patterns.diversity import (
    FeatureDiversityAnalysis,
    analyze_feature_diversity,
)
from apex_lab.research.signal_patterns.evaluator import (
    DEFAULT_TARGET,
    evaluate_all_candidates,
)
from apex_lab.research.signal_patterns.ranking import (
    build_ranking_artifacts,
    walk_forward_validate,
)
from apex_lab.research.signal_patterns.report import build_summary_payload, write_reports

DEFAULT_OUTPUT_DIR = Path("reports/lab/signal_patterns")

# Column name for feature importance ranking (from PR19 output).
_FEATURE_COL = "feature"
_SCORE_COL = "composite_score"


@dataclass(frozen=True)
class SignalPatternsResult:
    """Result bundle produced by the signal patterns engine."""

    ranked_signals: pl.DataFrame
    all_ranked_signals: pl.DataFrame
    candidate_statistics: pl.DataFrame
    walkforward_validation: pl.DataFrame
    rule_similarity: pl.DataFrame
    feature_diversity: FeatureDiversityAnalysis
    summary: dict[str, Any]


def _load_feature_importance(path: Path) -> pl.DataFrame:
    """Load feature importance CSV and return sorted by composite_score desc."""
    df = pl.read_csv(path)
    if _SCORE_COL in df.columns:
        df = df.sort(_SCORE_COL, descending=True)
    return df


def run_signal_patterns(
    dataset_path: Path,
    feature_ranking_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    generator_config: CandidateGeneratorConfig | None = None,
    target_column: str = DEFAULT_TARGET,
) -> SignalPatternsResult:
    """Run the High-Expectancy Signal Discovery Engine.

    Loads *dataset_path* (parquet) and *feature_ranking_path* (CSV from PR19),
    generates candidate rules from the top features, evaluates each rule,
    performs walk-forward validation to reject unstable patterns, ranks
    survivors by a composite score, and writes all reports to *output_dir*.

    Args:
        dataset_path: Path to the signal dataset parquet (PR18 output).
        feature_ranking_path: Path to feature_importance.csv (PR19 output).
        output_dir: Directory for output reports.
        generator_config: Optional candidate generator configuration.
        target_column: Forward-return column to use as evaluation target.

    Returns:
        A :class:`SignalPatternsResult` with all computed artefacts.
    """
    dataset = pl.read_parquet(dataset_path)
    feature_importance = _load_feature_importance(feature_ranking_path)
    feature_diversity = analyze_feature_diversity(dataset, feature_importance)

    # 1. Generate candidate rules from top features.
    candidates = generate_candidates(dataset, feature_importance, config=generator_config)

    # 2. Evaluate every candidate on the full dataset.
    candidate_stats = evaluate_all_candidates(
        dataset, candidates, target_column=target_column
    )

    # 3. Walk-forward validation — chronological splits.
    wf = walk_forward_validate(
        dataset, candidates, target_column=target_column
    )

    # 4. Rank signals with diversity-aware composite score.
    ranking_artifacts = build_ranking_artifacts(
        candidate_stats,
        wf,
        feature_to_cluster=feature_diversity.feature_to_cluster,
    )
    ranked = ranking_artifacts.ranked_signals

    # 5. Build summary payload and write reports.
    summary = build_summary_payload(
        ranked,
        wf,
        candidate_stats,
        all_ranked=ranking_artifacts.all_ranked_signals,
        feature_diversity=feature_diversity,
        rule_similarity=ranking_artifacts.rule_similarity,
    )
    write_reports(
        output_dir=output_dir,
        ranked=ranked,
        all_ranked=ranking_artifacts.all_ranked_signals,
        candidate_stats=candidate_stats,
        wf=wf,
        summary=summary,
        feature_diversity=feature_diversity,
        rule_similarity=ranking_artifacts.rule_similarity,
    )

    return SignalPatternsResult(
        ranked_signals=ranked,
        all_ranked_signals=ranking_artifacts.all_ranked_signals,
        candidate_statistics=candidate_stats,
        walkforward_validation=wf,
        rule_similarity=ranking_artifacts.rule_similarity,
        feature_diversity=feature_diversity,
        summary=summary,
    )
