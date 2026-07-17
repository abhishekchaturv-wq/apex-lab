"""Walk-forward validation and composite ranking for signal patterns."""

from __future__ import annotations

import ast
import heapq
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

import polars as pl

from apex_lab.research.signal_discovery.statistics import normalize_series
from apex_lab.research.signal_patterns.candidate_generator import CandidateRule
from apex_lab.research.signal_patterns.evaluator import (
    DEFAULT_TARGET,
    _group_stats,
    precompute_bucket_columns,
)

# Composite ranking weights. They sum to 1.0 and are applied to normalized
# metrics before complexity and suspicious-rule adjustments.
_EXPECTANCY_WEIGHT = 0.25
_OOS_EXPECTANCY_WEIGHT = 0.15
_WF_STABILITY_WEIGHT = 0.15
_WIN_RATE_WEIGHT = 0.10
_PROFIT_FACTOR_WEIGHT = 0.15
_FREQUENCY_WEIGHT = 0.10
_ROBUSTNESS_WEIGHT = 0.10

_TRAIN_RATIO = 0.60
_VALIDATION_RATIO = 0.20
_MIN_SPLIT_SAMPLES = 10
_MAX_WIN_RATE_STD = 0.20
_MAX_EXPECTANCY_CV = 1.5

_SUSPICIOUS_MAX_SAMPLES = 45
_SUSPICIOUS_WIN_RATE_THRESHOLD = 0.98
_SUSPICIOUS_PENALTY_ROBUST = 0.10
_SUSPICIOUS_PENALTY_NOT_ROBUST = 0.30

_COMPLEXITY_PENALTY_PER_EXTRA_FEATURE = 0.02
_REPRESENTATIVE_SIMILARITY_THRESHOLD = 0.85
_DIVERSITY_ZERO_THRESHOLD = 0.85
_DIVERSITY_MODERATE_THRESHOLD = 0.50
_SIMILARITY_REPORT_TOP_K = 10
_DEFAULT_SIMILARITY_METRICS = {
    "jaccard_similarity": 0.0,
    "cluster_overlap": 0.0,
    "shared_feature_ratio": 0.0,
    "similarity_score": 0.0,
}

EPSILON = 1e-12


@dataclass(frozen=True)
class RankingArtifacts:
    """Diversity-aware ranking outputs."""

    ranked_signals: pl.DataFrame
    all_ranked_signals: pl.DataFrame
    rule_similarity: pl.DataFrame


def _parse_feature_list(features_str: str) -> tuple[str, ...]:
    try:
        parsed = ast.literal_eval(features_str)
    except (ValueError, SyntaxError):
        return (features_str,)
    if not isinstance(parsed, list):
        return (features_str,)
    return tuple(str(feature) for feature in parsed)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _cluster_for_feature(feature: str, feature_to_cluster: dict[str, str]) -> str:
    return feature_to_cluster.get(feature, f"singleton::{feature}")


def _precompute_rule_sets(
    features: tuple[str, ...],
    feature_to_cluster: dict[str, str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Return (feature_set, cluster_set) for a rule, ready to reuse across comparisons."""
    feature_set = frozenset(features)
    cluster_set = frozenset(_cluster_for_feature(f, feature_to_cluster) for f in feature_set)
    return feature_set, cluster_set


def _build_inverted_indexes(
    rule_sets: list[tuple[frozenset[str], frozenset[str]]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Build feature→rules and cluster→rules inverted indexes in O(N) time.

    Returns two dicts: feature_to_rules and cluster_to_rules. Each maps a token
    to the list of rule indexes (in ascending order) that contain that token.
    These indexes allow O(actual-neighbors) candidate enumeration instead of O(N²).
    """
    feature_to_rules: dict[str, list[int]] = defaultdict(list)
    cluster_to_rules: dict[str, list[int]] = defaultdict(list)
    for i, (feat_set, clust_set) in enumerate(rule_sets):
        for f in feat_set:
            feature_to_rules[f].append(i)
        for c in clust_set:
            cluster_to_rules[c].append(i)
    return dict(feature_to_rules), dict(cluster_to_rules)


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) with path compression
# ---------------------------------------------------------------------------


def _uf_find(parent: list[int], x: int) -> int:
    """Iterative find with two-pass path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Second pass: path compression.
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


def _uf_union(parent: list[int], rank: list[int], x: int, y: int) -> None:
    """Union by rank to keep trees shallow."""
    px, py = _uf_find(parent, x), _uf_find(parent, y)
    if px == py:
        return
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1


# ---------------------------------------------------------------------------
# Frozenset deduplication: the key scalability insight
#
# The similarity_score between any two rules depends exclusively on their
# feature frozensets and the cluster mapping — bucket values are irrelevant.
# Therefore, two rules with identical feature frozensets have *identical*
# pairwise similarity with every third rule, and must always land in the
# same connected component.
#
# With top_features=25 and combo_sizes=(2,3,4) the number of distinct
# feature frozensets is at most C(25,2)+C(25,3)+C(25,4) = 15,250, which is
# orders of magnitude smaller than the ~692 K individual rules.
#
# Algorithm (exact, not approximate):
#   1. Group the N rules by feature frozenset → F ≤ 15,250 unique groups.
#   2. Build the similarity graph on those F unique frozensets.
#   3. Find connected components via Union-Find on F nodes.
#   4. Map every rule to its component through its frozenset group.
#
# Complexity: O(N + F² / index-pruning + N log N)
# vs previous: O(N × avg_candidates) ≈ O(N²) when avg_candidates ≈ N
#
# Correctness proof: The connected components of the N-rule graph are
# exactly the unions of rule-index sets for each frozenset-level component,
# because similarity is a function of (feature_frozenset, cluster_map) only.
# ---------------------------------------------------------------------------


def _build_frozenset_index(
    feature_lists: list[tuple[str, ...]],
) -> tuple[list[frozenset[str]], list[int], dict[frozenset[str], int]]:
    """Deduplicate feature lists by frozenset.

    Returns:
        unique_fs:    List of unique feature frozensets (length F ≤ 15,250).
        rule_to_uid:  rule_index → frozenset uid (length N).
        fs_to_uid:    frozenset → uid lookup.
    """
    fs_to_uid: dict[frozenset[str], int] = {}
    unique_fs: list[frozenset[str]] = []
    rule_to_uid: list[int] = []

    for feats in feature_lists:
        fs = frozenset(feats)
        if fs not in fs_to_uid:
            fs_to_uid[fs] = len(unique_fs)
            unique_fs.append(fs)
        rule_to_uid.append(fs_to_uid[fs])

    return unique_fs, rule_to_uid, fs_to_uid


def _candidate_neighbor_indexes(
    index: int,
    rule_sets: list[tuple[frozenset[str], frozenset[str]]],
    feature_to_rules: dict[str, list[int]],
    cluster_to_rules: dict[str, list[int]],
) -> set[int]:
    """Return the set of rule indexes sharing at least one feature or cluster with rule *index*.

    Two rules with no shared features AND no shared clusters always have
    similarity_score == 0 (jaccard == cluster_overlap == shared_ratio == 0), so
    they can never influence any threshold-based decision.  By restricting
    comparison to this candidate set we skip all structurally zero-similarity
    pairs while preserving exact correctness.
    """
    feat_set, clust_set = rule_sets[index]
    candidates: set[int] = set()
    for f in feat_set:
        if f in feature_to_rules:
            candidates.update(feature_to_rules[f])
    for c in clust_set:
        if c in cluster_to_rules:
            candidates.update(cluster_to_rules[c])
    candidates.discard(index)
    return candidates


def _rule_similarity_metrics(
    left_exact: frozenset[str],
    left_clusters: frozenset[str],
    right_exact: frozenset[str],
    right_clusters: frozenset[str],
) -> dict[str, float]:
    exact_union = left_exact | right_exact
    cluster_union = left_clusters | right_clusters
    exact_shared = left_exact & right_exact
    cluster_shared = left_clusters & right_clusters

    jaccard = len(exact_shared) / len(exact_union) if exact_union else 0.0
    cluster_overlap = len(cluster_shared) / len(cluster_union) if cluster_union else 0.0
    shared_ratio = len(exact_shared) / min(len(left_exact), len(right_exact)) if left_exact and right_exact else 0.0
    similarity = max(jaccard, cluster_overlap, shared_ratio)
    return {
        "jaccard_similarity": jaccard,
        "cluster_overlap": cluster_overlap,
        "shared_feature_ratio": shared_ratio,
        "similarity_score": similarity,
    }


def _diversity_score(metrics: dict[str, float]) -> float:
    cluster_overlap = metrics["cluster_overlap"]
    shared_ratio = metrics["shared_feature_ratio"]
    similarity = metrics["similarity_score"]

    if cluster_overlap <= EPSILON and shared_ratio <= EPSILON:
        return 1.0
    if similarity >= _DIVERSITY_ZERO_THRESHOLD:
        return 0.0
    if similarity >= _DIVERSITY_MODERATE_THRESHOLD or cluster_overlap >= _DIVERSITY_MODERATE_THRESHOLD:
        return 0.40
    return 0.75


def _detect_suspicious_rules(
    win_rates: list[float],
    frequencies: list[int],
    profit_factors: list[float | None],
) -> list[bool]:
    flags: list[bool] = []
    for wr, freq, pf in zip(win_rates, frequencies, profit_factors, strict=True):
        suspicious = False
        if wr >= _SUSPICIOUS_WIN_RATE_THRESHOLD and freq < _SUSPICIOUS_MAX_SAMPLES:
            suspicious = True
        elif pf is None and freq < _SUSPICIOUS_MAX_SAMPLES:
            suspicious = True
        flags.append(suspicious)
    return flags


def _representative_sort_key(row: dict[str, Any]) -> tuple[int, float, int, float, int, float, str]:
    """Return the representative-selection sort key.

    Components are ordered by robustness, walk-forward stability, signal
    frequency, expectancy, rule complexity, base composite score, and
    rule label.
    """
    return (
        -int(bool(row.get("is_robust", False))),
        -float(row.get("_wf_stability") or 0.0),
        -int(row.get("signal_frequency", 0)),
        -float(row.get("expectancy", 0.0)),
        int(row.get("combination_size", 0)),
        -float(row.get("base_composite_score", 0.0)),
        str(row.get("rule_label", "")),
    )


def walk_forward_validate(
    df: pl.DataFrame,
    candidates: list[CandidateRule],
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> pl.DataFrame:
    """Validate candidates chronologically across train / val / OOS splits."""
    n = df.height
    train_end = int(n * _TRAIN_RATIO)
    val_end = int(n * (_TRAIN_RATIO + _VALIDATION_RATIO))

    splits = {
        "train": df.slice(0, train_end),
        "val": df.slice(train_end, val_end - train_end),
        "oos": df.slice(val_end, n - val_end),
    }

    all_features: list[str] = sorted(
        {feature for rule in candidates for feature in rule.features if feature in df.columns}
    )
    splits_b: dict[str, pl.DataFrame] = {
        name: precompute_bucket_columns(split_df, all_features, bins=bins)
        for name, split_df in splits.items()
    }

    combo_to_candidates: dict[tuple[str, ...], list[CandidateRule]] = {}
    for rule in candidates:
        combo_to_candidates.setdefault(rule.features, []).append(rule)

    split_lookups: dict[str, dict[tuple[tuple[str, ...], str], dict[str, Any]]] = {
        name: {} for name in splits
    }

    for features_tuple in combo_to_candidates:
        bucket_columns = [f"_b_{feature}" for feature in features_tuple]
        for split_name, split_with_buckets in splits_b.items():
            if not all(column in split_with_buckets.columns for column in bucket_columns):
                continue
            if target_column not in split_with_buckets.columns:
                continue
            grouped = _group_stats(split_with_buckets, bucket_columns, target_column)
            if grouped.is_empty():
                continue
            for row in grouped.to_dicts():
                bucket_key = "|".join(str(row[column]) for column in bucket_columns)
                split_lookups[split_name][(features_tuple, bucket_key)] = row

    rows: list[dict[str, Any]] = []
    for rule in candidates:
        rule_row: dict[str, Any] = {
            "rule_label": rule.label(),
            "features": str(list(rule.features)),
            "combination_size": len(rule.features),
        }
        is_robust = True
        split_win_rates: list[float] = []
        split_expectancies: list[float] = []

        for split_name in ("train", "val", "oos"):
            stats = split_lookups[split_name].get((rule.features, rule.bucket_key))
            n_split = int(stats["_n"]) if stats is not None else 0

            if stats is None or n_split < _MIN_SPLIT_SAMPLES:
                is_robust = False
                rule_row[f"{split_name}_frequency"] = 0
                rule_row[f"{split_name}_win_rate"] = None
                rule_row[f"{split_name}_expectancy"] = None
                rule_row[f"{split_name}_profit_factor"] = None
            else:
                expectancy = float(stats["_expectancy"])
                win_rate = float(stats["_win_rate"])
                rule_row[f"{split_name}_frequency"] = n_split
                rule_row[f"{split_name}_win_rate"] = round(win_rate, 6)
                rule_row[f"{split_name}_expectancy"] = round(expectancy, 6)
                rule_row[f"{split_name}_profit_factor"] = _safe_float(stats["_profit_factor"])
                if expectancy <= 0:
                    is_robust = False
                split_win_rates.append(win_rate)
                split_expectancies.append(expectancy)

        if is_robust and len(split_win_rates) == 3:
            wr_mean = sum(split_win_rates) / 3.0
            wr_var = sum((value - wr_mean) ** 2 for value in split_win_rates) / 3.0
            wr_std = math.sqrt(wr_var)
            if wr_std > _MAX_WIN_RATE_STD:
                is_robust = False

        if is_robust and len(split_expectancies) == 3:
            exp_mean = sum(split_expectancies) / 3.0
            if abs(exp_mean) > EPSILON:
                exp_var = sum((value - exp_mean) ** 2 for value in split_expectancies) / 3.0
                exp_cv = math.sqrt(exp_var) / abs(exp_mean)
                if exp_cv > _MAX_EXPECTANCY_CV:
                    is_robust = False

        rule_row["is_robust"] = is_robust
        rows.append(rule_row)

    if not rows:
        return _empty_wf_frame()
    return pl.DataFrame(rows)


def _build_base_ranking(stats: pl.DataFrame, wf: pl.DataFrame) -> pl.DataFrame:
    if stats.is_empty():
        return _empty_ranked_frame()

    wf_join_cols = ["rule_label", "is_robust"]
    for column in ("train_expectancy", "val_expectancy", "oos_expectancy"):
        if not wf.is_empty() and column in wf.columns:
            wf_join_cols.append(column)

    if not wf.is_empty() and {"rule_label", "is_robust"}.issubset(set(wf.columns)):
        merged = stats.join(wf.select(wf_join_cols), on="rule_label", how="left")
        merged = merged.with_columns(pl.col("is_robust").fill_null(False))
    else:
        merged = stats.with_columns(pl.lit(False).alias("is_robust"))

    train_col = "train_expectancy" if "train_expectancy" in merged.columns else None
    val_col = "val_expectancy" if "val_expectancy" in merged.columns else None
    oos_col = "oos_expectancy" if "oos_expectancy" in merged.columns else None

    wf_stability_scores: list[float] = []
    oos_expectancy_list: list[float | None] = []

    for row in merged.to_dicts():
        expectancies: list[float] = []
        for column in (train_col, val_col, oos_col):
            if column is None:
                continue
            value = _safe_float(row.get(column))
            if value is not None:
                expectancies.append(value)

        oos_expectancy = _safe_float(row.get(oos_col)) if oos_col else None
        oos_expectancy_list.append(oos_expectancy)

        if len(expectancies) >= 2:
            mean_expectancy = sum(expectancies) / len(expectancies)
            if abs(mean_expectancy) > EPSILON:
                variance = sum((value - mean_expectancy) ** 2 for value in expectancies) / len(expectancies)
                cv = math.sqrt(variance) / abs(mean_expectancy)
                stability = max(0.0, 1.0 - min(1.0, cv))
            else:
                stability = 0.0
        else:
            stability = 0.0
        wf_stability_scores.append(stability)

    merged = merged.with_columns(
        [
            pl.Series("_oos_exp_calc", oos_expectancy_list, dtype=pl.Float64),
            pl.Series("_wf_stability", wf_stability_scores, dtype=pl.Float64),
        ]
    )

    expectancy_norm = normalize_series(merged.get_column("expectancy").to_list())
    oos_exp_norm = normalize_series(merged.get_column("_oos_exp_calc").to_list())
    win_rate_norm = normalize_series(merged.get_column("win_rate").to_list())
    profit_factor_norm = normalize_series(
        [_safe_float(value) for value in merged.get_column("profit_factor").to_list()]
    )
    frequency_norm = normalize_series(
        merged.get_column("signal_frequency").cast(pl.Float64).to_list()
    )
    wf_stability_norm = normalize_series(merged.get_column("_wf_stability").to_list())
    robustness_norm = normalize_series(
        [1.0 if value else 0.0 for value in merged.get_column("is_robust").to_list()]
    )

    win_rates = [float(value) for value in merged.get_column("win_rate").to_list()]
    frequencies = [int(value) for value in merged.get_column("signal_frequency").to_list()]
    profit_factors = [_safe_float(value) for value in merged.get_column("profit_factor").to_list()]
    suspicious_flags = _detect_suspicious_rules(win_rates, frequencies, profit_factors)
    robust_flags = merged.get_column("is_robust").to_list()
    combination_sizes = [int(value) for value in merged.get_column("combination_size").to_list()]

    composite_scores: list[float] = []
    for index, (expectancy, oos_expectancy, win_rate, profit_factor, frequency, stability, robustness) in enumerate(
        zip(
            expectancy_norm,
            oos_exp_norm,
            win_rate_norm,
            profit_factor_norm,
            frequency_norm,
            wf_stability_norm,
            robustness_norm,
            strict=True,
        )
    ):
        score = (
            (_EXPECTANCY_WEIGHT * expectancy)
            + (_OOS_EXPECTANCY_WEIGHT * oos_expectancy)
            + (_WIN_RATE_WEIGHT * win_rate)
            + (_PROFIT_FACTOR_WEIGHT * profit_factor)
            + (_FREQUENCY_WEIGHT * frequency)
            + (_WF_STABILITY_WEIGHT * stability)
            + (_ROBUSTNESS_WEIGHT * robustness)
        )

        if combination_sizes[index] > 2:
            score -= _COMPLEXITY_PENALTY_PER_EXTRA_FEATURE * (combination_sizes[index] - 2)

        if suspicious_flags[index]:
            penalty = (
                _SUSPICIOUS_PENALTY_ROBUST if robust_flags[index] else _SUSPICIOUS_PENALTY_NOT_ROBUST
            )
            score *= 1.0 - penalty

        composite_scores.append(max(0.0, score))

    ranked = merged.with_columns(
        [
            pl.Series("base_composite_score", composite_scores, dtype=pl.Float64),
            pl.Series("composite_score", composite_scores, dtype=pl.Float64),
            pl.lit(1.0).cast(pl.Float64).alias("diversity_score"),
            pl.lit(0.0).cast(pl.Float64).alias("max_similarity"),
        ]
    )

    return (
        ranked.sort(["base_composite_score", "rule_label"], descending=[True, False])
        .with_row_index("rank_before_diversity", offset=1)
        .with_columns(
            [
                pl.col("base_composite_score").round(6),
                pl.col("composite_score").round(6),
                pl.col("win_rate").round(6),
                pl.col("expectancy").round(6),
                pl.col("average_return").round(6),
                pl.col("median_return").round(6),
                pl.col("_wf_stability").round(6),
            ]
        )
    )


def _select_representatives(
    ranked: pl.DataFrame,
    feature_lists: list[tuple[str, ...]],
    feature_to_cluster: dict[str, str],
) -> tuple[dict[int, str], dict[int, str], set[int]]:
    """Build similarity groups and pick representatives using frozenset deduplication.

    Algorithmic improvement over the previous O(N²) / O(N × avg_candidates) scan
    -------------------------------------------------------------------------------
    The similarity_score between any two rules depends *only* on their feature
    frozensets and the cluster mapping — the specific bucket values in each rule
    are irrelevant.  Therefore, all rules that share the same feature frozenset
    have identical pairwise similarity with every other rule, and are guaranteed
    to be in the same connected component.

    This allows us to reduce the graph construction from N individual rules to
    F unique feature frozensets (F ≤ C(top_features,2) + C(top_features,3) +
    C(top_features,4) = 15,250 for top_features=25), and build the component
    graph on that much smaller set.

    Complexity:
        Step 1 – frozenset deduplication:  O(N)
        Step 2 – inverted index on F items: O(F × avg_features)
        Step 3 – pairwise on F items:       O(F × avg_candidates_among_F)
        Step 4 – Union-Find components:     O(F × α(F)) ≈ O(F)
        Step 5 – map N rules to components: O(N)
        Step 6 – representative selection:  O(N log N)
        Total:  O(N log N + F²) vs previous O(N × avg_candidates) ≈ O(N²)

    Correctness guarantee: the connected components of the N-rule similarity
    graph are *exactly* the unions of rule-index sets for each frozenset-level
    component.  No false negatives, no approximations.
    """
    row_dicts = ranked.to_dicts()
    N = len(feature_lists)

    # ------------------------------------------------------------------
    # Step 1: Deduplicate feature lists into F unique frozensets.
    # ------------------------------------------------------------------
    unique_fs, rule_to_uid, _ = _build_frozenset_index(feature_lists)
    F = len(unique_fs)

    # ------------------------------------------------------------------
    # Step 2: Build (feature_set, cluster_set) for each unique frozenset
    # and construct inverted indexes on F items (not N).
    # ------------------------------------------------------------------
    uid_rule_sets: list[tuple[frozenset[str], frozenset[str]]] = [
        _precompute_rule_sets(tuple(fs), feature_to_cluster) for fs in unique_fs
    ]
    feature_to_uids, cluster_to_uids = _build_inverted_indexes(uid_rule_sets)

    # ------------------------------------------------------------------
    # Step 3: Build adjacency graph among the F unique frozensets.
    # With F ≤ 15,250 and efficient inverted-index pruning, this is fast
    # even when the original N-rule indexes were degenerate.
    # ------------------------------------------------------------------
    uid_parent = list(range(F))
    uid_rank = [0] * F
    sim_computations = 0
    total_uid_candidates = 0

    for left_uid in range(F):
        neighbors_uid = _candidate_neighbor_indexes(
            left_uid, uid_rule_sets, feature_to_uids, cluster_to_uids
        )
        total_uid_candidates += len(neighbors_uid)
        for right_uid in neighbors_uid:
            if right_uid <= left_uid:
                continue
            left_exact, left_clusters = uid_rule_sets[left_uid]
            right_exact, right_clusters = uid_rule_sets[right_uid]
            metrics = _rule_similarity_metrics(
                left_exact, left_clusters, right_exact, right_clusters
            )
            sim_computations += 1
            if metrics["similarity_score"] >= _REPRESENTATIVE_SIMILARITY_THRESHOLD:
                _uf_union(uid_parent, uid_rank, left_uid, right_uid)

    # ------------------------------------------------------------------
    # Step 4: Build connected components — map rule indexes to component
    # roots via their frozenset uid.
    # ------------------------------------------------------------------
    root_to_rules: dict[int, list[int]] = defaultdict(list)
    for rule_idx in range(N):
        root = _uf_find(uid_parent, rule_to_uid[rule_idx])
        root_to_rules[root].append(rule_idx)

    components: list[list[int]] = [sorted(idxs) for idxs in root_to_rules.values()]

    if _logger.isEnabledFor(logging.DEBUG):
        avg_uid_cands = total_uid_candidates / F if F else 0.0
        _logger.debug(
            "_select_representatives: rules=%d  unique_feature_sets=%d  "
            "similarity_computations=%d  avg_uid_candidates=%.1f  components=%d",
            N,
            F,
            sim_computations,
            avg_uid_cands,
            len(components),
        )

    # ------------------------------------------------------------------
    # Step 5: Sort components by minimum rank_before_diversity (same
    # ordering as the previous implementation).
    # ------------------------------------------------------------------
    components.sort(
        key=lambda comp: min(int(row_dicts[idx]["rank_before_diversity"]) for idx in comp)
    )

    # ------------------------------------------------------------------
    # Step 6: Assign group IDs and pick the best representative per
    # component using the same sort key as before.
    # ------------------------------------------------------------------
    group_ids: dict[int, str] = {}
    representatives: dict[int, str] = {}
    representative_indexes: set[int] = set()

    for position, component in enumerate(components, start=1):
        group_id = f"group_{position:03d}"
        best_index = sorted(component, key=lambda idx: _representative_sort_key(row_dicts[idx]))[0]
        representative_indexes.add(best_index)
        representative_label = str(row_dicts[best_index]["rule_label"])
        for idx in component:
            group_ids[idx] = group_id
            representatives[idx] = representative_label

    return group_ids, representatives, representative_indexes


def _rerank_with_diversity(
    ranked: pl.DataFrame,
    feature_lists: list[tuple[str, ...]],
    feature_to_cluster: dict[str, str],
) -> pl.DataFrame:
    """Greedy diversity re-ranking with a lazy-deletion min-heap.

    Compared to the previous O(K² log K) implementation that rescanned all
    remaining rules every iteration, this version maintains an incrementally
    updated priority queue:

    - Initialisation:           O(K log K)
    - Selection per iteration:  O(log K)  amortised (lazy-deletion pops)
    - Neighbor updates:         O(avg_neighbors × log K)
    - Total:                    O(K × avg_neighbors × log K)

    The heap stores tuples of the form:
        (−adjusted_score, −diversity, −base_score, rule_label, index, seq)
    where ``seq`` is a monotonically increasing per-index counter.  When
    strongest_metrics for a rule is updated, a new entry with an incremented
    ``seq`` is pushed, and the old entry becomes stale.  On pop, a stale entry
    is detected by checking whether the stored ``seq`` matches the current
    sequence number for that index; stale entries are discarded.

    Because strongest_metrics is monotonically non-decreasing (similarity only
    ever rises), adjusted_score is monotonically non-increasing.  Stale entries
    therefore always have *better* (more negative) neg_adjusted_score values
    and will be popped before the valid entry.  Version tracking ensures they
    are correctly discarded.

    All scores, diversity values and ranking order are bit-for-bit identical
    to the previous implementation.
    """
    if ranked.is_empty():
        return ranked

    row_dicts = ranked.to_dicts()
    K = len(row_dicts)
    remaining: set[int] = set(range(K))
    selected: list[int] = []
    adjusted_scores: dict[int, float] = {}
    diversity_scores: dict[int, float] = {}
    max_similarities: dict[int, float] = {}
    strongest_metrics: dict[int, dict[str, float]] = {
        idx: dict(_DEFAULT_SIMILARITY_METRICS) for idx in range(K)
    }

    rule_sets = [_precompute_rule_sets(f, feature_to_cluster) for f in feature_lists]
    feature_to_rules, cluster_to_rules = _build_inverted_indexes(rule_sets)

    # ------------------------------------------------------------------
    # Build the initial heap.  At the start every rule has diversity 1.0
    # so adjusted_score == base_composite_score.
    # Heap entry: (neg_adj, neg_div, neg_base, label, index, seq)
    # ------------------------------------------------------------------
    base_scores: list[float] = [float(row_dicts[i]["base_composite_score"]) for i in range(K)]
    seq: list[int] = [0] * K  # current sequence number per rule
    heap: list[tuple] = []
    for idx in range(K):
        entry = (
            -base_scores[idx],
            -1.0,
            -base_scores[idx],
            str(row_dicts[idx]["rule_label"]),
            idx,
            0,
        )
        heapq.heappush(heap, entry)

    sim_computations = 0
    total_candidates = 0

    while remaining:
        # ------------------------------------------------------------------
        # Pop until we find a valid (non-stale, still remaining) entry.
        # ------------------------------------------------------------------
        while heap:
            neg_adj, neg_div, neg_base, _label, idx, s = heap[0]
            if idx in remaining and s == seq[idx]:
                break
            heapq.heappop(heap)

        if not heap:
            break  # defensive: should not happen if implementation is correct

        neg_adj, neg_div, neg_base, _label, chosen, _ = heapq.heappop(heap)

        adjusted_scores[chosen] = -neg_adj
        diversity_scores[chosen] = -neg_div
        max_similarities[chosen] = strongest_metrics[chosen]["similarity_score"]

        selected.append(chosen)
        remaining.discard(chosen)

        # ------------------------------------------------------------------
        # Update strongest_metrics for each remaining neighbor of `chosen`.
        # Only neighbors can have their similarity change; all other remaining
        # rules have similarity 0 with `chosen` and are unaffected.
        # ------------------------------------------------------------------
        chosen_exact, chosen_clusters = rule_sets[chosen]
        neighbors = _candidate_neighbor_indexes(chosen, rule_sets, feature_to_rules, cluster_to_rules)
        total_candidates += len(neighbors & remaining)
        for idx in neighbors:
            if idx not in remaining:
                continue
            idx_exact, idx_clusters = rule_sets[idx]
            pair_metrics = _rule_similarity_metrics(
                idx_exact, idx_clusters, chosen_exact, chosen_clusters
            )
            sim_computations += 1
            current_metrics = strongest_metrics[idx]
            if (
                pair_metrics["similarity_score"],
                pair_metrics["cluster_overlap"],
                pair_metrics["shared_feature_ratio"],
            ) > (
                current_metrics["similarity_score"],
                current_metrics["cluster_overlap"],
                current_metrics["shared_feature_ratio"],
            ):
                strongest_metrics[idx] = pair_metrics
                new_div = _diversity_score(pair_metrics)
                new_adj = base_scores[idx] * new_div
                seq[idx] += 1
                heapq.heappush(
                    heap,
                    (
                        -new_adj,
                        -new_div,
                        -base_scores[idx],
                        str(row_dicts[idx]["rule_label"]),
                        idx,
                        seq[idx],
                    ),
                )

    if _logger.isEnabledFor(logging.DEBUG):
        avg_cands = total_candidates / K if K else 0.0
        _logger.debug(
            "_rerank_with_diversity:    rules=%d  similarity_computations=%d  avg_candidates=%.1f",
            K,
            sim_computations,
            avg_cands,
        )

    ordered = ranked[selected].with_columns(
        [
            pl.Series(
                "diversity_score",
                [round(diversity_scores[index], 6) for index in selected],
                dtype=pl.Float64,
            ),
            pl.Series(
                "max_similarity",
                [round(max_similarities[index], 6) for index in selected],
                dtype=pl.Float64,
            ),
            pl.Series(
                "composite_score",
                [round(adjusted_scores[index], 6) for index in selected],
                dtype=pl.Float64,
            ),
        ]
    )
    return ordered.with_row_index("rank", offset=1)


def _build_rule_similarity_report(
    ranked: pl.DataFrame,
    feature_to_cluster: dict[str, str],
    top_k: int = _SIMILARITY_REPORT_TOP_K,
) -> pl.DataFrame:
    """Report the top-K most similar neighbours per rule (O(N×K) rows, not O(N²))."""
    if ranked.is_empty() or "is_robust" not in ranked.columns:
        return _empty_rule_similarity_frame()

    robust = ranked.filter(pl.col("is_robust").cast(pl.Boolean).fill_null(False))
    if robust.height < 2:
        return _empty_rule_similarity_frame()

    feature_lists = [_parse_feature_list(str(v)) for v in robust.get_column("features").to_list()]
    labels = [str(v) for v in robust.get_column("rule_label").to_list()]
    n = len(feature_lists)
    k = min(top_k, n - 1)

    rule_sets = [_precompute_rule_sets(f, feature_to_cluster) for f in feature_lists]

    # Per-rule min-heaps of size k, keyed by (score, neighbor_label, ...).
    # heap[0] is the LOWEST-score entry – the one we would evict.
    # We update both sides of each pair in a single O(N²/2) scan.
    _SimilarityTuple = tuple[float, float, float, float]  # (jaccard, cluster_overlap, shared_ratio, sim_score)
    _HeapEntry = tuple[float, str, int, _SimilarityTuple]
    heaps: list[list[_HeapEntry]] = [[] for _ in range(n)]

    def _push(heap: list[_HeapEntry], score: float, neighbor_label: str, neighbor_index: int, m_tuple: _SimilarityTuple) -> None:
        entry: _HeapEntry = (score, neighbor_label, neighbor_index, m_tuple)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    for left_index in range(n):
        left_exact, left_clusters = rule_sets[left_index]
        for right_index in range(left_index + 1, n):
            right_exact, right_clusters = rule_sets[right_index]
            metrics = _rule_similarity_metrics(
                left_exact, left_clusters, right_exact, right_clusters
            )
            score = metrics["similarity_score"]
            m_tuple: _SimilarityTuple = (
                metrics["jaccard_similarity"],
                metrics["cluster_overlap"],
                metrics["shared_feature_ratio"],
                score,
            )
            _push(heaps[left_index], score, labels[right_index], right_index, m_tuple)
            _push(heaps[right_index], score, labels[left_index], left_index, m_tuple)

    seen_pairs: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []

    for rule_index in range(n):
        for score, neighbor_label, _, m_tuple in sorted(
            heaps[rule_index], key=lambda x: (-x[0], x[1])
        ):
            left_label, right_label = sorted([labels[rule_index], neighbor_label])
            pair_key = (left_label, right_label)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            jaccard, cluster_overlap, shared_ratio, sim_score = m_tuple
            rows.append(
                {
                    "rule_label_left": left_label,
                    "rule_label_right": right_label,
                    "jaccard_similarity": round(jaccard, 6),
                    "cluster_overlap": round(cluster_overlap, 6),
                    "shared_feature_ratio": round(shared_ratio, 6),
                    "similarity_score": round(sim_score, 6),
                }
            )

    if not rows:
        return _empty_rule_similarity_frame()

    return pl.DataFrame(rows).sort(
        ["similarity_score", "rule_label_left", "rule_label_right"],
        descending=[True, False, False],
    )


def build_ranking_artifacts(
    stats: pl.DataFrame,
    wf: pl.DataFrame,
    feature_to_cluster: dict[str, str] | None = None,
) -> RankingArtifacts:
    """Compute diversity-aware ranking artifacts."""
    cluster_map = feature_to_cluster or {}
    base_ranked = _build_base_ranking(stats, wf)
    if base_ranked.is_empty():
        empty = _empty_rule_similarity_frame()
        return RankingArtifacts(
            ranked_signals=base_ranked,
            all_ranked_signals=base_ranked,
            rule_similarity=empty,
        )

    feature_lists = [
        _parse_feature_list(str(v)) for v in base_ranked.get_column("features").to_list()
    ]

    group_ids, representatives, representative_indexes = _select_representatives(
        base_ranked,
        feature_lists,
        cluster_map,
    )

    all_ranked = base_ranked.with_columns(
        [
            pl.Series(
                "similarity_group_id",
                [group_ids[index] for index in range(base_ranked.height)],
                dtype=pl.Utf8,
            ),
            pl.Series(
                "representative_rule_label",
                [representatives[index] for index in range(base_ranked.height)],
                dtype=pl.Utf8,
            ),
            pl.Series(
                "is_representative_rule",
                [index in representative_indexes for index in range(base_ranked.height)],
                dtype=pl.Boolean,
            ),
        ]
    )

    # Extract feature lists for representative rules, preserving their order in base_ranked.
    rep_feature_lists = [
        feature_lists[i] for i in range(base_ranked.height) if i in representative_indexes
    ]

    representative_ranked = _rerank_with_diversity(
        all_ranked.filter(pl.col("is_representative_rule")),
        rep_feature_lists,
        cluster_map,
    )
    representative_ranked = representative_ranked.select(_ranking_output_columns())

    all_ranked = all_ranked.select(_all_ranking_output_columns())
    rule_similarity = _build_rule_similarity_report(all_ranked, cluster_map)

    return RankingArtifacts(
        ranked_signals=representative_ranked,
        all_ranked_signals=all_ranked,
        rule_similarity=rule_similarity,
    )


def rank_signals(
    stats: pl.DataFrame,
    wf: pl.DataFrame,
    feature_to_cluster: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Compute the final diversity-aware ranked signal table."""
    return build_ranking_artifacts(stats, wf, feature_to_cluster=feature_to_cluster).ranked_signals


def _ranking_output_columns() -> list[str]:
    return [
        "rank",
        "rule_label",
        "features",
        "conditions",
        "combination_size",
        "signal_frequency",
        "win_rate",
        "average_return",
        "median_return",
        "profit_factor",
        "expectancy",
        "average_mfe",
        "average_mae",
        "is_robust",
        "similarity_group_id",
        "representative_rule_label",
        "diversity_score",
        "max_similarity",
        "base_composite_score",
        "composite_score",
    ]


def _all_ranking_output_columns() -> list[str]:
    return [
        "rank_before_diversity",
        "rule_label",
        "features",
        "conditions",
        "combination_size",
        "signal_frequency",
        "win_rate",
        "average_return",
        "median_return",
        "profit_factor",
        "expectancy",
        "average_mfe",
        "average_mae",
        "is_robust",
        "similarity_group_id",
        "representative_rule_label",
        "is_representative_rule",
        "diversity_score",
        "max_similarity",
        "base_composite_score",
        "composite_score",
    ]


def _empty_wf_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rule_label": pl.Series([], dtype=pl.Utf8),
            "features": pl.Series([], dtype=pl.Utf8),
            "combination_size": pl.Series([], dtype=pl.Int64),
            "train_frequency": pl.Series([], dtype=pl.Int64),
            "train_win_rate": pl.Series([], dtype=pl.Float64),
            "train_expectancy": pl.Series([], dtype=pl.Float64),
            "train_profit_factor": pl.Series([], dtype=pl.Float64),
            "val_frequency": pl.Series([], dtype=pl.Int64),
            "val_win_rate": pl.Series([], dtype=pl.Float64),
            "val_expectancy": pl.Series([], dtype=pl.Float64),
            "val_profit_factor": pl.Series([], dtype=pl.Float64),
            "oos_frequency": pl.Series([], dtype=pl.Int64),
            "oos_win_rate": pl.Series([], dtype=pl.Float64),
            "oos_expectancy": pl.Series([], dtype=pl.Float64),
            "oos_profit_factor": pl.Series([], dtype=pl.Float64),
            "is_robust": pl.Series([], dtype=pl.Boolean),
        }
    )


def _empty_ranked_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rank": pl.Series([], dtype=pl.UInt32),
            "rule_label": pl.Series([], dtype=pl.Utf8),
            "features": pl.Series([], dtype=pl.Utf8),
            "conditions": pl.Series([], dtype=pl.Utf8),
            "combination_size": pl.Series([], dtype=pl.Int64),
            "signal_frequency": pl.Series([], dtype=pl.Int64),
            "win_rate": pl.Series([], dtype=pl.Float64),
            "average_return": pl.Series([], dtype=pl.Float64),
            "median_return": pl.Series([], dtype=pl.Float64),
            "profit_factor": pl.Series([], dtype=pl.Float64),
            "expectancy": pl.Series([], dtype=pl.Float64),
            "average_mfe": pl.Series([], dtype=pl.Float64),
            "average_mae": pl.Series([], dtype=pl.Float64),
            "is_robust": pl.Series([], dtype=pl.Boolean),
            "similarity_group_id": pl.Series([], dtype=pl.Utf8),
            "representative_rule_label": pl.Series([], dtype=pl.Utf8),
            "diversity_score": pl.Series([], dtype=pl.Float64),
            "max_similarity": pl.Series([], dtype=pl.Float64),
            "base_composite_score": pl.Series([], dtype=pl.Float64),
            "composite_score": pl.Series([], dtype=pl.Float64),
        }
    )


def _empty_rule_similarity_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "rule_label_left": pl.Series([], dtype=pl.Utf8),
            "rule_label_right": pl.Series([], dtype=pl.Utf8),
            "jaccard_similarity": pl.Series([], dtype=pl.Float64),
            "cluster_overlap": pl.Series([], dtype=pl.Float64),
            "shared_feature_ratio": pl.Series([], dtype=pl.Float64),
            "similarity_score": pl.Series([], dtype=pl.Float64),
        }
    )
