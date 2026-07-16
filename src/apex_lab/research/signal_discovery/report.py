"""Report writing helpers for signal discovery outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _records(df: pl.DataFrame) -> list[dict[str, Any]]:
    return [_sanitize_value(record) for record in df.to_dicts()]


def build_summary_payload(
    feature_importance: pl.DataFrame,
    combinations: pl.DataFrame,
    stability_report: dict[str, Any],
    category_report: pl.DataFrame,
) -> dict[str, Any]:
    """Build summary.json payload."""
    top_features = _records(feature_importance.head(20))
    top_combinations = _records(combinations.head(20))
    most_stable = [
        row
        for row in stability_report.get("features", [])
        if row.get("stability") == "Stable"
    ][:20]
    predictive_categories = _records(category_report.head(20))

    recommended = [
        row["feature"]
        for row in top_features
        if row.get("composite_score") is not None and row.get("composite_score", 0.0) >= 0.6
    ][:20]
    ignored = [
        row["feature"]
        for row in _records(feature_importance.sort("composite_score", descending=False).head(20))
    ]

    return {
        "top_features": top_features,
        "top_feature_combinations": top_combinations,
        "most_stable_features": _sanitize_value(most_stable),
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
        json.dumps(_records(feature_importance), indent=2),
        encoding="utf-8",
    )
    (output_dir / "top_combinations.json").write_text(
        json.dumps(_records(combinations), indent=2),
        encoding="utf-8",
    )
    (output_dir / "stability_report.json").write_text(
        json.dumps(_sanitize_value(stability_report), indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_sanitize_value(summary), indent=2),
        encoding="utf-8",
    )
