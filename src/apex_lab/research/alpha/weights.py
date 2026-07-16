"""Weight loading and generation for the Alpha Scoring Engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from apex_lab.research.alpha.registry import get_alpha_feature_registry

DEFAULT_WEIGHTS_FILENAME = "weights.json"
DEFAULT_CONTEXT_LEADERBOARD_PATH = Path("reports/lab/context/leaderboard.csv")


@dataclass(frozen=True)
class FeatureWeight:
    """A bucket-specific feature weight entry loaded from weights.json."""

    feature: str
    bucket: str
    category: str
    weight: float
    source_score: float | None = None


def ensure_weights_file(
    weights_path: Path,
    context_leaderboard_path: Path = DEFAULT_CONTEXT_LEADERBOARD_PATH,
) -> list[FeatureWeight]:
    """Load weights from disk, auto-generating them from context leaderboard when absent."""
    if not weights_path.exists():
        generated = generate_weights_from_leaderboard(context_leaderboard_path)
        write_weights_file(weights_path, generated)
    return load_weights_file(weights_path)


def generate_weights_from_leaderboard(context_leaderboard_path: Path) -> list[FeatureWeight]:
    """Generate one scored bucket per feature from PR12 leaderboard and normalize to 100."""
    registry = get_alpha_feature_registry()
    leaderboard = pl.read_csv(context_leaderboard_path)

    candidates = leaderboard.filter(
        pl.col("feature").is_in(list(registry.keys()))
        & pl.col("bucket").is_not_null()
        & pl.col("score").is_not_null()
    )

    top_by_feature = (
        candidates.sort(["feature", "score"], descending=[False, True])
        .group_by("feature", maintain_order=True)
        .first()
    )

    raw_rows: list[FeatureWeight] = []
    for row in top_by_feature.iter_rows(named=True):
        feature = str(row["feature"])
        spec = registry.get(feature)
        if spec is None:
            continue
        raw_rows.append(
            FeatureWeight(
                feature=feature,
                bucket=str(row["bucket"]),
                category=spec.category,
                weight=float(row["score"]),
                source_score=float(row["score"]),
            )
        )

    return _normalize_weights(raw_rows)


def load_weights_file(weights_path: Path) -> list[FeatureWeight]:
    """Load weights.json entries and normalize loaded weights to an exact sum of 100."""
    payload = json.loads(weights_path.read_text(encoding="utf-8"))
    rows = payload.get("weights", []) if isinstance(payload, dict) else []
    weights = [
        FeatureWeight(
            feature=str(row["feature"]),
            bucket=str(row["bucket"]),
            category=str(row["category"]),
            weight=float(row["weight"]),
            source_score=(float(row["source_score"]) if row.get("source_score") is not None else None),
        )
        for row in rows
    ]
    return _normalize_weights(weights)


def write_weights_file(weights_path: Path, weights: list[FeatureWeight]) -> None:
    """Write weights with normalization metadata for future manual editing."""
    normalized = _normalize_weights(weights)
    payload = {
        "normalization": {
            "method": "sum_to_100_with_last_row_adjustment",
            "description": (
                "Raw weights are scaled by (raw_weight / total_raw) * 100, "
                "rounded to 6 decimals, and final-row difference is applied so the sum is exactly 100."
            ),
            "weight_sum": round(sum(weight.weight for weight in normalized), 6),
        },
        "weights": [asdict(weight) for weight in normalized],
    }
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_weights(weights: list[FeatureWeight]) -> list[FeatureWeight]:
    if not weights:
        return []

    raw_total = sum(max(weight.weight, 0.0) for weight in weights)
    if raw_total <= 0:
        equal = 100.0 / len(weights)
        normalized = [
            FeatureWeight(
                feature=weight.feature,
                bucket=weight.bucket,
                category=weight.category,
                weight=round(equal, 6),
                source_score=weight.source_score,
            )
            for weight in weights
        ]
    else:
        normalized = [
            FeatureWeight(
                feature=weight.feature,
                bucket=weight.bucket,
                category=weight.category,
                weight=round((max(weight.weight, 0.0) / raw_total) * 100.0, 6),
                source_score=weight.source_score,
            )
            for weight in weights
        ]

    running_sum = sum(weight.weight for weight in normalized)
    adjustment = round(100.0 - running_sum, 6)
    last = normalized[-1]
    normalized[-1] = FeatureWeight(
        feature=last.feature,
        bucket=last.bucket,
        category=last.category,
        weight=round(max(last.weight + adjustment, 0.0), 6),
        source_score=last.source_score,
    )
    return normalized
