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
from apex_lab.research.signal_patterns.evaluator import (
    DEFAULT_TARGET,
    evaluate_all_candidates,
)
from apex_lab.research.signal_patterns.ranking import rank_signals, walk_forward_validate
from apex_lab.research.signal_patterns.report import build_summary_payload, write_reports

DEFAULT_OUTPUT_DIR = Path("reports/lab/signal_patterns")

# Column name for feature importance ranking (from PR19 output).
_FEATURE_COL = "feature"
_SCORE_COL = "composite_score"


@dataclass(frozen=True)
class SignalPatternsResult:
    """Result bundle produced by the signal patterns engine."""

    ranked_signals: pl.DataFrame
    candidate_statistics: pl.DataFrame
    walkforward_validation: pl.DataFrame
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

    # 4. Rank signals with composite score.
    ranked = rank_signals(candidate_stats, wf)

    # 5. Build summary payload and write reports.
    summary = build_summary_payload(ranked, wf, candidate_stats)
    write_reports(
        output_dir=output_dir,
        ranked=ranked,
        candidate_stats=candidate_stats,
        wf=wf,
        summary=summary,
    )

    return SignalPatternsResult(
        ranked_signals=ranked,
        candidate_statistics=candidate_stats,
        walkforward_validation=wf,
        summary=summary,
    )
