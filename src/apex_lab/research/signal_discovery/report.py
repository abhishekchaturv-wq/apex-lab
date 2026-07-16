"""Report writing helpers for signal discovery outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def build_summary_payload(
    feature_importance: pl.DataFrame,
    combinations: pl.DataFrame,
    stability_report: dict[str, Any],
    category_report: pl.DataFrame,
) -> dict[str, Any]:
    """Build summary.json payload."""
    top_features = feature_importance.head(20).to_dicts()
    top_combinations = combinations.head(20).to_dicts()
    most_stable = [
        row
        for row in stability_report.get("features", [])
        if row.get("stability") == "Stable"
    ][:20]
    predictive_categories = category_report.head(20).to_dicts()

    recommended = [
        row["feature"]
        for row in top_features
        if row.get("composite_score") is not None and row.get("composite_score", 0.0) >= 0.6
    ][:20]
    ignored = [
        row["feature"]
        for row in feature_importance.sort("composite_score", descending=False).head(20).to_dicts()
    ]

    return {
        "top_features": top_features,
        "top_feature_combinations": top_combinations,
        "most_stable_features": most_stable,
        "most_predictive_categories": predictive_categories,
        "recommended_pine_features": recommended,
        "features_to_ignore": ignored,
    }


def write_reports(
    output_dir: Path,
    feature_importance: pl.DataFrame,
    combinations: pl.DataFrame,
    stability_report: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Write all signal discovery reports to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_importance.write_csv(output_dir / "feature_importance.csv")
    combinations.write_csv(output_dir / "top_combinations.csv")

    (output_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance.to_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "top_combinations.json").write_text(
        json.dumps(combinations.to_dicts(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "stability_report.json").write_text(
        json.dumps(stability_report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
