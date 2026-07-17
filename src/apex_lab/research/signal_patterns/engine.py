"""High-Expectancy Signal Discovery Engine orchestrator."""

from __future__ import annotations

import logging
import time
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

_logger = logging.getLogger(__name__)


def _rss_mb() -> float | None:
    """Return peak resident set size in MiB, or None if unavailable."""
    try:
        import resource  # type: ignore[import-not-found]

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports in KiB; macOS in bytes.
        import sys

        divisor = 1024 if sys.platform != "darwin" else (1024 * 1024)
        return raw / divisor
    except Exception:
        return None


def _log_stage(stage: str, elapsed: float, n_rules: int = 0) -> None:
    """Emit a single DEBUG line per pipeline stage (zero cost when DEBUG is off)."""
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    rss = _rss_mb()
    rss_str = f"  peak_rss={rss:.1f}MiB" if rss is not None else ""
    _logger.debug("[profiling] %-34s  elapsed=%.3fs%s  rules=%d", stage, elapsed, rss_str, n_rules)


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

    t0 = time.perf_counter()
    feature_diversity = analyze_feature_diversity(dataset, feature_importance)
    _log_stage("analyze_feature_diversity", time.perf_counter() - t0)

    # 1. Generate candidate rules from top features.
    t0 = time.perf_counter()
    candidates = generate_candidates(dataset, feature_importance, config=generator_config)
    _log_stage("generate_candidates", time.perf_counter() - t0, n_rules=len(candidates))

    # 2. Evaluate every candidate on the full dataset.
    t0 = time.perf_counter()
    candidate_stats = evaluate_all_candidates(
        dataset, candidates, target_column=target_column
    )
    _log_stage("evaluate_all_candidates", time.perf_counter() - t0, n_rules=candidate_stats.height)

    # 3. Walk-forward validation — chronological splits.
    t0 = time.perf_counter()
    wf = walk_forward_validate(
        dataset, candidates, target_column=target_column
    )
    _log_stage("walk_forward_validate", time.perf_counter() - t0, n_rules=wf.height)

    # 4. Rank signals with diversity-aware composite score.
    t0 = time.perf_counter()
    ranking_artifacts = build_ranking_artifacts(
        candidate_stats,
        wf,
        feature_to_cluster=feature_diversity.feature_to_cluster,
    )
    _log_stage(
        "build_ranking_artifacts",
        time.perf_counter() - t0,
        n_rules=ranking_artifacts.all_ranked_signals.height,
    )
    ranked = ranking_artifacts.ranked_signals

    # 5. Build summary payload and write reports.
    t0 = time.perf_counter()
    summary = build_summary_payload(
        ranked,
        wf,
        candidate_stats,
        all_ranked=ranking_artifacts.all_ranked_signals,
        feature_diversity=feature_diversity,
        rule_similarity=ranking_artifacts.rule_similarity,
    )
    _log_stage("build_summary_payload", time.perf_counter() - t0)

    t0 = time.perf_counter()
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
    _log_stage("write_reports", time.perf_counter() - t0)

    return SignalPatternsResult(
        ranked_signals=ranked,
        all_ranked_signals=ranking_artifacts.all_ranked_signals,
        candidate_statistics=candidate_stats,
        walkforward_validation=wf,
        rule_similarity=ranking_artifacts.rule_similarity,
        feature_diversity=feature_diversity,
        summary=summary,
    )
