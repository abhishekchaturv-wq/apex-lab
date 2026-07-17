"""Feature correlation, clustering, and diversity analysis for signal patterns."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import polars as pl

from apex_lab.research.signal_discovery.statistics import correlation

_FEATURE_COL = "feature"
_IMPORTANCE_COL = "composite_score"
_CORRELATION_THRESHOLD = 0.85


@dataclass(frozen=True)
class FeatureCluster:
    """A deterministic cluster of related features."""

    cluster_id: str
    representative_feature: str
    member_features: tuple[str, ...]
    average_importance: float
    strongest_feature: str
    strongest_importance: float
    feature_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "representative_feature": self.representative_feature,
            "member_features": list(self.member_features),
            "average_importance": round(self.average_importance, 6),
            "strongest_feature": self.strongest_feature,
            "strongest_importance": round(self.strongest_importance, 6),
            "feature_count": self.feature_count,
        }


@dataclass(frozen=True)
class FeatureDiversityAnalysis:
    """Computed diversity artifacts for engineered features."""

    feature_correlation: pl.DataFrame
    feature_clusters: tuple[FeatureCluster, ...]
    cluster_importance: pl.DataFrame
    feature_to_cluster: dict[str, str]
    feature_to_representative: dict[str, str]


def _importance_lookup(feature_importance: pl.DataFrame) -> dict[str, float]:
    if _FEATURE_COL not in feature_importance.columns:
        return {}
    score_column = _IMPORTANCE_COL if _IMPORTANCE_COL in feature_importance.columns else None
    lookup: dict[str, float] = {}
    for row in feature_importance.select(
        [_FEATURE_COL, score_column] if score_column is not None else [_FEATURE_COL]
    ).to_dicts():
        score = float(row.get(score_column, 0.0)) if score_column is not None else 0.0
        lookup[str(row[_FEATURE_COL])] = score
    return lookup


def _connected_components(
    features: list[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    remaining = sorted(features)
    seen: set[str] = set()
    components: list[list[str]] = []

    for feature in remaining:
        if feature in seen:
            continue
        stack = [feature]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - seen, reverse=True))
        components.append(sorted(component))
    return components


def _correlation_strength(
    left: str,
    right: str,
    correlation_lookup: dict[tuple[str, str], float],
) -> float:
    key = (left, right) if left <= right else (right, left)
    return correlation_lookup.get(key, 0.0)


def _representative_feature(
    members: list[str],
    importance_lookup: dict[str, float],
    correlation_lookup: dict[tuple[str, str], float],
) -> str:
    if len(members) == 1:
        return members[0]

    scores: list[tuple[str, float, float]] = []
    for feature in members:
        related = [
            _correlation_strength(feature, other, correlation_lookup)
            for other in members
            if other != feature
        ]
        scores.append(
            (
                feature,
                mean(related) if related else 0.0,
                importance_lookup.get(feature, 0.0),
            )
        )
    scores.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return scores[0][0]


def _strongest_feature(members: list[str], importance_lookup: dict[str, float]) -> tuple[str, float]:
    ranked = sorted(
        ((feature, importance_lookup.get(feature, 0.0)) for feature in members),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0]


def analyze_feature_diversity(
    dataset: pl.DataFrame,
    feature_importance: pl.DataFrame,
) -> FeatureDiversityAnalysis:
    """Compute correlation, clustering, and importance summaries for ranked features."""
    if _FEATURE_COL not in feature_importance.columns:
        empty = pl.DataFrame()
        return FeatureDiversityAnalysis(empty, (), empty, {}, {})

    ranked_features = [
        str(feature)
        for feature in feature_importance.get_column(_FEATURE_COL).to_list()
        if str(feature) in dataset.columns
    ]
    if not ranked_features:
        empty = pl.DataFrame()
        return FeatureDiversityAnalysis(empty, (), empty, {}, {})

    importance_lookup = _importance_lookup(feature_importance)
    numeric_features = [
        feature for feature in ranked_features if dataset.schema[feature].is_numeric()
    ]
    non_numeric_features = [
        feature for feature in ranked_features if not dataset.schema[feature].is_numeric()
    ]

    adjacency: dict[str, set[str]] = {feature: set() for feature in numeric_features}
    correlation_rows: list[dict[str, object]] = []
    correlation_lookup: dict[tuple[str, str], float] = {}

    for index, left in enumerate(numeric_features):
        left_series = dataset.get_column(left)
        for right in numeric_features[index + 1 :]:
            right_series = dataset.get_column(right)
            pearson = correlation(left_series, right_series, method="pearson")
            spearman = correlation(left_series, right_series, method="spearman")
            abs_pearson = abs(pearson) if pearson is not None else 0.0
            abs_spearman = abs(spearman) if spearman is not None else 0.0
            max_abs = max(abs_pearson, abs_spearman)
            linked = max_abs >= _CORRELATION_THRESHOLD
            correlation_rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                    "abs_pearson_correlation": abs_pearson,
                    "abs_spearman_correlation": abs_spearman,
                    "max_abs_correlation": max_abs,
                    "is_cluster_link": linked,
                }
            )
            correlation_lookup[(left, right)] = max_abs
            if linked:
                adjacency[left].add(right)
                adjacency[right].add(left)

    feature_correlation = (
        pl.DataFrame(correlation_rows)
        if correlation_rows
        else pl.DataFrame(
            {
                "feature_left": pl.Series([], dtype=pl.Utf8),
                "feature_right": pl.Series([], dtype=pl.Utf8),
                "pearson_correlation": pl.Series([], dtype=pl.Float64),
                "spearman_correlation": pl.Series([], dtype=pl.Float64),
                "abs_pearson_correlation": pl.Series([], dtype=pl.Float64),
                "abs_spearman_correlation": pl.Series([], dtype=pl.Float64),
                "max_abs_correlation": pl.Series([], dtype=pl.Float64),
                "is_cluster_link": pl.Series([], dtype=pl.Boolean),
            }
        )
    )
    if not feature_correlation.is_empty():
        feature_correlation = feature_correlation.sort(
            ["max_abs_correlation", "feature_left", "feature_right"],
            descending=[True, False, False],
        )

    numeric_components = _connected_components(numeric_features, adjacency)
    all_components = numeric_components + [[feature] for feature in sorted(non_numeric_features)]

    provisional_clusters: list[dict[str, object]] = []
    for members in all_components:
        representative = _representative_feature(members, importance_lookup, correlation_lookup)
        strongest_feature, strongest_importance = _strongest_feature(members, importance_lookup)
        provisional_clusters.append(
            {
                "representative_feature": representative,
                "member_features": tuple(sorted(members)),
                "average_importance": mean(importance_lookup.get(feature, 0.0) for feature in members),
                "strongest_feature": strongest_feature,
                "strongest_importance": strongest_importance,
                "feature_count": len(members),
            }
        )

    provisional_clusters.sort(
        key=lambda item: (
            -float(item["average_importance"]),
            -int(item["feature_count"]),
            str(item["representative_feature"]),
        )
    )

    feature_to_cluster: dict[str, str] = {}
    feature_to_representative: dict[str, str] = {}
    cluster_rows: list[dict[str, object]] = []
    clusters: list[FeatureCluster] = []

    for index, cluster in enumerate(provisional_clusters, start=1):
        cluster_id = f"cluster_{index:03d}"
        members = tuple(cluster["member_features"])
        record = FeatureCluster(
            cluster_id=cluster_id,
            representative_feature=str(cluster["representative_feature"]),
            member_features=members,
            average_importance=float(cluster["average_importance"]),
            strongest_feature=str(cluster["strongest_feature"]),
            strongest_importance=float(cluster["strongest_importance"]),
            feature_count=int(cluster["feature_count"]),
        )
        clusters.append(record)
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "representative_feature": record.representative_feature,
                "member_features": "|".join(record.member_features),
                "average_importance": round(record.average_importance, 6),
                "strongest_feature": record.strongest_feature,
                "strongest_importance": round(record.strongest_importance, 6),
                "feature_count": record.feature_count,
            }
        )
        for feature in record.member_features:
            feature_to_cluster[feature] = cluster_id
            feature_to_representative[feature] = record.representative_feature

    cluster_importance = pl.DataFrame(cluster_rows) if cluster_rows else pl.DataFrame()
    return FeatureDiversityAnalysis(
        feature_correlation=feature_correlation,
        feature_clusters=tuple(clusters),
        cluster_importance=cluster_importance,
        feature_to_cluster=feature_to_cluster,
        feature_to_representative=feature_to_representative,
    )
