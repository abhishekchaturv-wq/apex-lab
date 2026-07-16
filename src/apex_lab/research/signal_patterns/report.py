"""Report writing helpers for signal patterns outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl


def _sanitize(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _records(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [_sanitize(record) for record in df.to_dicts()]


def build_summary_payload(
    ranked: pl.DataFrame,
    wf: pl.DataFrame,
    candidate_stats: pl.DataFrame,
) -> dict[str, Any]:
    """Build summary.json payload for signal patterns."""
    top_20 = _records(ranked.head(20))

    # Recommend Pine rules from robust, high-scoring signals.
    robust_signals = ranked.filter(pl.col("is_robust").cast(pl.Boolean).fill_null(False))
    recommended_pine_rules = [
        row["rule_label"] for row in _records(robust_signals.head(10))
    ]

    # Entry / exit conditions from top robust signals.
    recommended_entry = recommended_pine_rules[:5]
    recommended_exit = [
        f"NOT ({rule})" for rule in recommended_pine_rules[:5]
    ]

    # Most stable signals (robust + high composite score).
    most_stable = [row["rule_label"] for row in top_20 if row.get("is_robust")]

    # Rejected signals: non-robust entries.
    all_labels = set(ranked.get_column("rule_label").to_list())
    robust_labels = set(robust_signals.get_column("rule_label").to_list()) if not robust_signals.is_empty() else set()
    rejected = list(all_labels - robust_labels)[:20]

    return _sanitize(
        {
            "top_20_signals": top_20,
            "recommended_pine_rules": recommended_pine_rules,
            "recommended_entry_conditions": recommended_entry,
            "recommended_exit_conditions": recommended_exit,
            "most_stable_signals": most_stable,
            "rejected_signals": rejected,
            "total_candidates_evaluated": candidate_stats.height,
            "total_robust_signals": len(robust_labels),
        }
    )


def write_reports(
    output_dir: Path,
    ranked: pl.DataFrame,
    candidate_stats: pl.DataFrame,
    wf: pl.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write all signal patterns reports to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # top_signals.csv / .json — top-ranked signals
    top = ranked.head(50) if ranked.height > 50 else ranked
    top.write_csv(output_dir / "top_signals.csv")
    (output_dir / "top_signals.json").write_text(
        json.dumps(_records(top), indent=2),
        encoding="utf-8",
    )

    # candidate_statistics.csv — full statistics table
    candidate_stats.write_csv(output_dir / "candidate_statistics.csv")

    # walkforward_validation.csv
    wf.write_csv(output_dir / "walkforward_validation.csv")

    # summary.json
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
