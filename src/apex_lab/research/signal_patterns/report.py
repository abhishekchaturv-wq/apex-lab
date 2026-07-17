"""Report writing helpers for signal patterns outputs."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.signal_patterns.diversity import FeatureDiversityAnalysis


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {dict_key: _sanitize(dict_value) for dict_key, dict_value in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _records(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [_sanitize(record) for record in df.to_dicts()]


def _parse_features(features_value: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(features_value)
    except (ValueError, SyntaxError):
        return (features_value,)
    if not isinstance(parsed, list):
        return (features_value,)
    return tuple(str(feature) for feature in parsed)


def _concept_counts(
    ranked: pl.DataFrame,
    feature_diversity: FeatureDiversityAnalysis,
    limit: int = 20,
) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    if ranked.is_empty() or "features" not in ranked.columns:
        return []
    for row in ranked.head(limit).to_dicts():
        concepts = {
            feature_diversity.feature_to_representative.get(feature, feature)
            for feature in _parse_features(str(row["features"]))
        }
        for concept in sorted(concepts):
            counts[concept] = counts.get(concept, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_summary_payload(
    ranked: pl.DataFrame,
    wf: pl.DataFrame,
    candidate_stats: pl.DataFrame,
    *,
    all_ranked: pl.DataFrame | None = None,
    feature_diversity: FeatureDiversityAnalysis | None = None,
    rule_similarity: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Build summary.json payload for signal patterns."""
    top_20 = _records(ranked.head(20))
    robust_signals = ranked.filter(pl.col("is_robust").cast(pl.Boolean).fill_null(False))
    recommended_pine_rules = [row["rule_label"] for row in _records(robust_signals.head(10))]
    recommended_entry = recommended_pine_rules[:5]
    recommended_exit = [f"NOT ({rule})" for rule in recommended_pine_rules[:5]]
    most_stable = [row["rule_label"] for row in top_20 if row.get("is_robust")]

    all_labels = set(ranked.get_column("rule_label").to_list()) if "rule_label" in ranked.columns else set()
    robust_labels = (
        set(robust_signals.get_column("rule_label").to_list()) if not robust_signals.is_empty() else set()
    )
    rejected = list(all_labels - robust_labels)[:20]

    summary: dict[str, Any] = {
        "top_20_signals": top_20,
        "recommended_pine_rules": recommended_pine_rules,
        "recommended_entry_conditions": recommended_entry,
        "recommended_exit_conditions": recommended_exit,
        "most_stable_signals": most_stable,
        "rejected_signals": rejected,
        "total_candidates_evaluated": candidate_stats.height,
        "total_robust_signals": len(robust_labels),
    }

    if all_ranked is not None and not all_ranked.is_empty():
        summary["diversity_statistics"] = {
            "ranked_rules_before_diversity": all_ranked.height,
            "representative_rules_after_diversity": ranked.height,
            "redundancy_removed": max(0, all_ranked.height - ranked.height),
        }

    if feature_diversity is not None:
        summary["feature_cluster_count"] = len(feature_diversity.feature_clusters)
        summary["top_20_market_concepts"] = [
            {"concept": concept, "count": count}
            for concept, count in _concept_counts(ranked, feature_diversity, limit=20)
        ]

    if rule_similarity is not None:
        summary["robust_rule_similarity_pairs"] = rule_similarity.height

    return _sanitize(summary)


def build_diversity_report(
    ranked: pl.DataFrame,
    all_ranked: pl.DataFrame,
    feature_diversity: FeatureDiversityAnalysis,
    rule_similarity: pl.DataFrame,
) -> str:
    """Build the markdown diversity report."""
    top_before = (
        all_ranked.head(10)
        if not all_ranked.is_empty()
        else pl.DataFrame({"rank_before_diversity": [], "rule_label": [], "base_composite_score": []})
    )
    top_after = (
        ranked.head(10)
        if not ranked.is_empty()
        else pl.DataFrame({"rank": [], "rule_label": [], "composite_score": [], "diversity_score": []})
    )
    concept_counts = _concept_counts(ranked, feature_diversity, limit=20)
    redundancy_removed = max(0, all_ranked.height - ranked.height)
    robust_pairs = rule_similarity.height

    lines = [
        "# Signal Diversity Report",
        "",
        "## Diversity Statistics",
        f"- Ranked rules before diversity filtering: {all_ranked.height}",
        f"- Representative rules after diversity filtering: {ranked.height}",
        f"- Redundant rules removed: {redundancy_removed}",
        f"- Robust rule similarity pairs: {robust_pairs}",
        f"- Feature clusters discovered: {len(feature_diversity.feature_clusters)}",
        "",
        "## Discovered Feature Clusters",
    ]

    for cluster in feature_diversity.feature_clusters:
        lines.append(
            f"- **{cluster.cluster_id}** — representative `{cluster.representative_feature}`, "
            f"strongest `{cluster.strongest_feature}`, members: {', '.join(cluster.member_features)}"
        )

    lines.extend(
        [
            "",
            "## Representative Features",
            "| Cluster | Representative | Strongest | Avg Importance | Count |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in feature_diversity.cluster_importance.to_dicts():
        lines.append(
            f"| {row['cluster_id']} | {row['representative_feature']} | {row['strongest_feature']} | "
            f"{row['average_importance']:.6f} | {row['feature_count']} |"
        )

    lines.extend(
        [
            "",
            "## Before vs After Ranking Comparison",
            "| Before Rank | Before Rule | Base Score | After Rank | After Rule | Final Score | Diversity |",
            "|---:|---|---:|---:|---|---:|---:|",
        ]
    )
    before_rows = top_before.to_dicts()
    after_rows = top_after.to_dicts()
    row_count = max(len(before_rows), len(after_rows))
    for index in range(row_count):
        before = before_rows[index] if index < len(before_rows) else {}
        after = after_rows[index] if index < len(after_rows) else {}
        lines.append(
            "| "
            f"{before.get('rank_before_diversity', '')} | {before.get('rule_label', '')} | "
            f"{before.get('base_composite_score', '')} | {after.get('rank', '')} | "
            f"{after.get('rule_label', '')} | {after.get('composite_score', '')} | "
            f"{after.get('diversity_score', '')} |"
        )

    lines.extend(
        [
            "",
            "## Representative Rules",
            "| Rank | Rule | Group | Final Score | Diversity |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in ranked.head(20).to_dicts():
        lines.append(
            f"| {row['rank']} | {row['rule_label']} | {row['similarity_group_id']} | "
            f"{row['composite_score']:.6f} | {row['diversity_score']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Cluster Coverage of Top 20",
            "| Market Concept | Count |",
            "|---|---:|",
        ]
    )
    for concept, count in concept_counts:
        lines.append(f"| {concept} | {count} |")

    lines.extend(
        [
            "",
            "## Distribution of Market Concepts Represented",
            f"- Unique concepts in Top 20: {len(concept_counts)}",
        ]
    )
    for concept, count in concept_counts:
        lines.append(f"- {concept}: {count}")

    return "\n".join(lines) + "\n"


def write_reports(
    output_dir: Path,
    ranked: pl.DataFrame,
    candidate_stats: pl.DataFrame,
    wf: pl.DataFrame,
    summary: dict[str, Any],
    *,
    all_ranked: pl.DataFrame | None = None,
    feature_diversity: FeatureDiversityAnalysis | None = None,
    rule_similarity: pl.DataFrame | None = None,
) -> None:
    """Write all signal patterns reports to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top = ranked.head(50) if ranked.height > 50 else ranked
    top.write_csv(output_dir / "top_signals.csv")
    (output_dir / "top_signals.json").write_text(
        json.dumps(_records(top), indent=2),
        encoding="utf-8",
    )

    candidate_stats.write_csv(output_dir / "candidate_statistics.csv")
    wf.write_csv(output_dir / "walkforward_validation.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if all_ranked is not None:
        all_ranked.write_csv(output_dir / "all_ranked_signals.csv")

    if feature_diversity is not None:
        feature_diversity.feature_correlation.write_csv(output_dir / "feature_correlation.csv")
        feature_diversity.cluster_importance.write_csv(output_dir / "cluster_importance.csv")
        (output_dir / "feature_clusters.json").write_text(
            json.dumps([cluster.to_dict() for cluster in feature_diversity.feature_clusters], indent=2),
            encoding="utf-8",
        )
        diversity_report = build_diversity_report(
            ranked=ranked,
            all_ranked=all_ranked if all_ranked is not None else ranked,
            feature_diversity=feature_diversity,
            rule_similarity=rule_similarity if rule_similarity is not None else pl.DataFrame(),
        )
        (output_dir / "signal_diversity_report.md").write_text(
            diversity_report,
            encoding="utf-8",
        )

    if rule_similarity is not None:
        rule_similarity.write_csv(output_dir / "rule_similarity.csv")
