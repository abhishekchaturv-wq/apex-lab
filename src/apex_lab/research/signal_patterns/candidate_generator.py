"""Candidate signal rule generator for signal patterns engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import polars as pl

from apex_lab.research.signal_discovery.statistics import discretize_series

# Maximum number of top features to draw candidates from.
DEFAULT_TOP_FEATURES = 25
# Combination sizes to generate.
COMBO_SIZES: tuple[int, ...] = (2, 3, 4)
# Minimum sample count for a candidate to be considered.
MIN_SAMPLES = 30


@dataclass(frozen=True)
class CandidateRule:
    """A candidate signal rule combining one or more feature conditions."""

    features: tuple[str, ...]
    conditions: tuple[str, ...]  # human-readable condition per feature
    bucket_key: str  # internal join key for filtering
    bucket_columns: tuple[str, ...]  # column names used in bucket key

    def label(self) -> str:
        return " AND ".join(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "conditions": list(self.conditions),
            "label": self.label(),
        }


@dataclass
class CandidateGeneratorConfig:
    """Configuration for candidate rule generation."""

    top_features: int = DEFAULT_TOP_FEATURES
    combo_sizes: tuple[int, ...] = field(default_factory=lambda: COMBO_SIZES)
    min_samples: int = MIN_SAMPLES
    bins: int = 4


def _build_bucket_column(df: pl.DataFrame, feature: str, bins: int) -> pl.Series:
    """Discretize one feature into stable string buckets."""
    series = df.get_column(feature)
    return discretize_series(series, bins=bins)


def _condition_label(feature: str, bucket: str) -> str:
    """Build a human-readable condition string from feature and bucket value."""
    return f"{feature} == {bucket}"


def generate_candidates(
    df: pl.DataFrame,
    feature_importance: pl.DataFrame,
    config: CandidateGeneratorConfig | None = None,
) -> list[CandidateRule]:
    """Generate candidate signal rules using top-ranked features.

    Combinations of 2-, 3- and 4-feature conditions are created by
    discretizing numeric features and using native categories for
    categorical/string features.  Only combinations backed by at least
    ``config.min_samples`` rows are retained.

    Args:
        df: Signal dataset with feature columns and target columns.
        feature_importance: Feature importance table produced by PR19; must
            have a ``feature`` column ordered by descending importance.
        config: Generator configuration; defaults are used when ``None``.

    Returns:
        List of :class:`CandidateRule` objects ready for evaluation.
    """
    cfg = config or CandidateGeneratorConfig()

    top_features: list[str] = (
        feature_importance.select("feature")
        .head(cfg.top_features)
        .get_column("feature")
        .to_list()
    )
    available = [f for f in top_features if f in df.columns]

    # Pre-discretize all available features once.
    bucket_map: dict[str, pl.Series] = {}
    for feature in available:
        bucket_map[feature] = _build_bucket_column(df, feature, cfg.bins)

    candidates: list[CandidateRule] = []

    for size in cfg.combo_sizes:
        if len(available) < size:
            continue
        for combo in combinations(available, size):
            bucket_cols = [f"_b_{f}" for f in combo]
            # Build a temporary frame with bucket columns only.
            scoped = pl.DataFrame(
                {
                    col: bucket_map[f].to_list()
                    for f, col in zip(combo, bucket_cols, strict=True)
                }
            )

            # Find bucket combinations sorted by count (desc) then by bucket values
            # (asc) for a fully deterministic ordering.
            grouped = (
                scoped.group_by(bucket_cols)
                .agg(pl.len().alias("_count"))
                .sort(["_count", *bucket_cols], descending=[True, *[False] * len(bucket_cols)])
            )

            for row in grouped.to_dicts():
                count = row["_count"]
                if count < cfg.min_samples:
                    continue

                bucket_values = [str(row[col]) for col in bucket_cols]
                conditions = tuple(
                    _condition_label(feature, bucket)
                    for feature, bucket in zip(combo, bucket_values, strict=True)
                )
                bucket_key = "|".join(bucket_values)

                candidates.append(
                    CandidateRule(
                        features=tuple(combo),
                        conditions=conditions,
                        bucket_key=bucket_key,
                        bucket_columns=tuple(bucket_cols),
                    )
                )

    return candidates
