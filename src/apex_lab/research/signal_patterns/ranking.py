"""Walk-forward validation and composite ranking for signal patterns."""

from __future__ import annotations

import ast
import math
from typing import Any

import polars as pl

from apex_lab.research.signal_discovery.statistics import normalize_series
from apex_lab.research.signal_patterns.candidate_generator import CandidateRule
from apex_lab.research.signal_patterns.evaluator import (
    DEFAULT_TARGET,
    _group_stats,
    precompute_bucket_columns,
)

# Composite ranking weights (must sum to 1.0).
# Walk-forward robustness, OOS expectancy, and stability are emphasised;
# raw in-sample win rate is de-emphasised.
_EXPECTANCY_WEIGHT = 0.25
_OOS_EXPECTANCY_WEIGHT = 0.15
_WF_STABILITY_WEIGHT = 0.15
_WIN_RATE_WEIGHT = 0.10
_PROFIT_FACTOR_WEIGHT = 0.15
_FREQUENCY_WEIGHT = 0.10
_ROBUSTNESS_WEIGHT = 0.10

# Walk-forward split ratios.
_TRAIN_RATIO = 0.60
_VALIDATION_RATIO = 0.20
# Out-of-sample is the remainder (0.20).

# Minimum signal count per split for a rule to be considered stable.
# Raised from 5 → 10 to demand meaningful frequency in every fold.
_MIN_SPLIT_SAMPLES = 10

# Maximum allowed standard deviation of win-rates across folds.
# Rules whose win rate swings wildly between periods are rejected.
_MAX_WIN_RATE_STD = 0.20

# Maximum coefficient of variation of expectancy across folds.
# Guards against rules that look great in one fold and terrible in another.
_MAX_EXPECTANCY_CV = 1.5

# Suspicious-rule thresholds.  Rules with near-perfect win rates on very few
# samples are likely overfit; they receive a confidence penalty unless they
# also pass walk-forward validation.
_SUSPICIOUS_MAX_SAMPLES = 45
_SUSPICIOUS_WIN_RATE_THRESHOLD = 0.98
_SUSPICIOUS_PENALTY_ROBUST = 0.10      # 10 % score reduction if rule is also WF-robust
_SUSPICIOUS_PENALTY_NOT_ROBUST = 0.30  # 30 % score reduction otherwise

# Complexity penalty applied per extra feature beyond a 2-feature baseline.
# Two-feature rules are preferred over three-feature rules of equal quality.
_COMPLEXITY_PENALTY_PER_EXTRA_FEATURE = 0.02

EPSILON = 1e-12

# ---------------------------------------------------------------------------
# Correlated-feature synonym map for deduplication
# ---------------------------------------------------------------------------

# Features whose names appear below are "canonicalised" to the same group
# representative before deduplication.  Rules that differ only by substituting
# one group member for another are treated as near-duplicates; only the
# highest-scoring representative is retained in the ranked output.

_FEATURE_SYNONYM_MAP: dict[str, str] = {
    # Price aggregates — all measure the same underlying concept.
    "median_price": "price_aggregate",
    "typical_price": "price_aggregate",
    "weighted_price": "price_aggregate",
    "vwap": "price_aggregate",
    "hlc3": "price_aggregate",
    "ohlc4": "price_aggregate",
    # Opening-range levels — both express the same opening range.
    "or_high": "opening_range",
    "or_low": "opening_range",
    "or_range": "opening_range",
    "or_midpoint": "opening_range",
    "or_size": "opening_range",
    # Swing-point variants — high and low swing points are symmetric.
    "swing_high": "swing_point",
    "swing_low": "swing_point",
}

# Substring-based canonicalisation handles naming variants like "_swing_high_10".
_FEATURE_CANONICAL_SUBSTRINGS: list[tuple[str, str]] = [
    ("swing_high", "swing_point"),
    ("swing_low", "swing_point"),
    ("or_high", "opening_range"),
    ("or_low", "opening_range"),
    ("median_price", "price_aggregate"),
    ("typical_price", "price_aggregate"),
    ("weighted_price", "price_aggregate"),
    ("vwap", "price_aggregate"),
]


def _canonical_feature_name(feature: str) -> str:
    """Return a canonical group name for a feature (used for deduplication)."""
    if feature in _FEATURE_SYNONYM_MAP:
        return _FEATURE_SYNONYM_MAP[feature]
    fl = feature.lower()
    for substring, canonical in _FEATURE_CANONICAL_SUBSTRINGS:
        if substring in fl:
            return canonical
    return feature


def _rule_canonical_signature(features_str: str) -> frozenset[str]:
    """Parse a features string and return a canonical frozenset for similarity checks."""
    try:
        feature_list: list[str] = ast.literal_eval(features_str)
    except (ValueError, SyntaxError):
        return frozenset({features_str})
    return frozenset(_canonical_feature_name(f) for f in feature_list)


def _deduplicate_ranked(ranked: pl.DataFrame) -> pl.DataFrame:
    """Remove near-duplicate rules by canonical feature signature.

    The DataFrame is assumed to be pre-sorted descending by composite_score so
    the first occurrence of each canonical group is the highest-scoring one.
    Rank numbers are re-assigned after deduplication.
    """
    if ranked.is_empty() or "features" not in ranked.columns:
        return ranked

    seen: set[frozenset[str]] = set()
    keep: list[int] = []
    features_list = ranked.get_column("features").to_list()

    for i, features_val in enumerate(features_list):
        sig = _rule_canonical_signature(str(features_val))
        if sig not in seen:
            seen.add(sig)
            keep.append(i)

    if not keep:
        return ranked.head(0)

    deduped = ranked[keep]
    # Re-assign contiguous rank numbers after removing duplicates.
    if "rank" in deduped.columns:
        deduped = deduped.drop("rank")
    return deduped.with_row_index("rank", offset=1)


def _detect_suspicious_rules(
    win_rates: list[float],
    frequencies: list[int],
    profit_factors: list[float | None],
) -> list[bool]:
    """Return True for each rule that exhibits suspiciously perfect in-sample stats.

    A rule is flagged when:
    - Its win rate is at or above ``_SUSPICIOUS_WIN_RATE_THRESHOLD`` AND its
      sample count is below ``_SUSPICIOUS_MAX_SAMPLES`` (100 % win rate on
      very few trades is almost certainly overfit).
    - Its profit factor is ``None`` (zero losses recorded) AND its sample count
      is below ``_SUSPICIOUS_MAX_SAMPLES``.
    """
    flags: list[bool] = []
    for wr, freq, pf in zip(win_rates, frequencies, profit_factors, strict=True):
        suspicious = False
        if wr >= _SUSPICIOUS_WIN_RATE_THRESHOLD and freq < _SUSPICIOUS_MAX_SAMPLES:
            suspicious = True
        elif pf is None and freq < _SUSPICIOUS_MAX_SAMPLES:
            suspicious = True
        flags.append(suspicious)
    return flags


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def walk_forward_validate(
    df: pl.DataFrame,
    candidates: list[CandidateRule],
    target_column: str = DEFAULT_TARGET,
    bins: int = 4,
) -> pl.DataFrame:
    """Validate candidates chronologically across train / val / OOS splits.

    The dataset is split in chronological order:

    - **train**  — first 60 % of rows
    - **val**    — next  20 % of rows
    - **oos**    — last  20 % of rows (out-of-sample)

    A rule is marked **robust** only when ALL of the following hold:

    1. Each split has at least ``_MIN_SPLIT_SAMPLES`` matching rows.
    2. Expectancy is **positive** in every split (train, val, and OOS).
    3. The standard deviation of win rates across all three folds does not
       exceed ``_MAX_WIN_RATE_STD`` (consistent win rate across periods).
    4. The coefficient of variation of expectancy across all three folds does
       not exceed ``_MAX_EXPECTANCY_CV`` (stable expectancy, not just lucky
       on one fold).

    Bucket columns are precomputed once per split; each unique feature
    combination is aggregated in a single ``group_by`` pass.
    """
    n = df.height
    train_end = int(n * _TRAIN_RATIO)
    val_end = int(n * (_TRAIN_RATIO + _VALIDATION_RATIO))

    splits = {
        "train": df.slice(0, train_end),
        "val": df.slice(train_end, val_end - train_end),
        "oos": df.slice(val_end, n - val_end),
    }

    # Collect all features needed across all candidates.
    all_features: list[str] = sorted(
        {f for rule in candidates for f in rule.features if f in df.columns}
    )

    # Precompute bucket columns for each split exactly once.
    splits_b: dict[str, pl.DataFrame] = {
        name: precompute_bucket_columns(split_df, all_features, bins=bins)
        for name, split_df in splits.items()
    }

    # Group candidates by feature combination.
    combo_to_candidates: dict[tuple[str, ...], list[CandidateRule]] = {}
    for rule in candidates:
        combo_to_candidates.setdefault(rule.features, []).append(rule)

    # Precompute group_by stats for every (feature_combination, split) pair.
    # Outer key: split name.  Inner key: (features_tuple, bucket_key) → stats row dict.
    split_lookups: dict[str, dict[tuple[tuple[str, ...], str], dict[str, Any]]] = {
        name: {} for name in splits
    }

    for features_tuple in combo_to_candidates:
        bucket_cols = [f"_b_{f}" for f in features_tuple]
        for split_name, split_b in splits_b.items():
            if not all(c in split_b.columns for c in bucket_cols):
                continue
            if target_column not in split_b.columns:
                continue
            grouped = _group_stats(split_b, bucket_cols, target_column)
            if grouped.is_empty():
                continue
            for row in grouped.to_dicts():
                bucket_key = "|".join(str(row[c]) for c in bucket_cols)
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
                exp = float(stats["_expectancy"])
                wr = float(stats["_win_rate"])
                rule_row[f"{split_name}_frequency"] = n_split
                rule_row[f"{split_name}_win_rate"] = round(wr, 6)
                rule_row[f"{split_name}_expectancy"] = round(exp, 6)
                rule_row[f"{split_name}_profit_factor"] = _safe_float(stats["_profit_factor"])
                # All three splits must show positive expectancy.
                if exp <= 0:
                    is_robust = False
                split_win_rates.append(wr)
                split_expectancies.append(exp)

        # Consistency gate — only evaluated when all three splits were seen.
        if is_robust and len(split_win_rates) == 3:
            wr_mean = sum(split_win_rates) / 3.0
            wr_var = sum((w - wr_mean) ** 2 for w in split_win_rates) / 3.0
            wr_std = math.sqrt(wr_var)
            if wr_std > _MAX_WIN_RATE_STD:
                is_robust = False

        if is_robust and len(split_expectancies) == 3:
            exp_mean = sum(split_expectancies) / 3.0
            if abs(exp_mean) > EPSILON:
                exp_var = sum((e - exp_mean) ** 2 for e in split_expectancies) / 3.0
                exp_cv = math.sqrt(exp_var) / abs(exp_mean)
                if exp_cv > _MAX_EXPECTANCY_CV:
                    is_robust = False

        rule_row["is_robust"] = is_robust
        rows.append(rule_row)

    if not rows:
        return _empty_wf_frame()

    return pl.DataFrame(rows)


def rank_signals(
    stats: pl.DataFrame,
    wf: pl.DataFrame,
) -> pl.DataFrame:
    """Compute composite scores and produce the final ranked signal table.

    Improvements over the earlier implementation:

    - OOS expectancy and walk-forward stability are explicit scoring dimensions,
      reducing in-sample bias.
    - Raw win-rate weight is reduced; walk-forward consistency is rewarded.
    - Rules with more features receive a small complexity penalty so that
      simpler rules are preferred when predictive quality is similar.
    - Rules with suspiciously perfect in-sample statistics (100 % win rate
      on very few samples) receive a confidence penalty unless they also pass
      walk-forward validation.
    - Near-duplicate rules (same canonical feature signature) are deduplicated
      so the output contains genuinely diverse trading ideas.
    - Sorting uses a deterministic tiebreaker (rule_label ascending) so
      repeated runs on the same data always produce identical output.
    """
    if stats.is_empty():
        return _empty_ranked_frame()

    # ------------------------------------------------------------------
    # Join walk-forward data.
    # ------------------------------------------------------------------
    wf_join_cols = ["rule_label", "is_robust"]
    for col in ("train_expectancy", "val_expectancy", "oos_expectancy"):
        if not wf.is_empty() and col in wf.columns:
            wf_join_cols.append(col)

    if not wf.is_empty() and "rule_label" in wf.columns and "is_robust" in wf.columns:
        wf_slim = wf.select(wf_join_cols)
        merged = stats.join(wf_slim, on="rule_label", how="left")
        merged = merged.with_columns(pl.col("is_robust").fill_null(False))
    else:
        merged = stats.with_columns(pl.lit(False).alias("is_robust"))

    # ------------------------------------------------------------------
    # Compute walk-forward stability and extract OOS expectancy.
    # Walk-forward stability = 1 − CV(expectancy across folds), clipped to
    # [0, 1].  High stability means the expectancy is consistent over time.
    # ------------------------------------------------------------------
    train_col = "train_expectancy" if "train_expectancy" in merged.columns else None
    val_col = "val_expectancy" if "val_expectancy" in merged.columns else None
    oos_col = "oos_expectancy" if "oos_expectancy" in merged.columns else None

    wf_stability_scores: list[float] = []
    oos_expectancy_list: list[float | None] = []

    for row in merged.to_dicts():
        exps: list[float] = []
        for col in (train_col, val_col, oos_col):
            if col is not None:
                val = _safe_float(row.get(col))
                if val is not None:
                    exps.append(val)

        oos_exp = _safe_float(row.get(oos_col)) if oos_col else None
        oos_expectancy_list.append(oos_exp)

        if len(exps) >= 2:
            mean_exp = sum(exps) / len(exps)
            if abs(mean_exp) > EPSILON:
                var_exp = sum((e - mean_exp) ** 2 for e in exps) / len(exps)
                cv = math.sqrt(var_exp) / abs(mean_exp)
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

    # ------------------------------------------------------------------
    # Normalise scoring components.
    # ------------------------------------------------------------------
    expectancy_norm = normalize_series(merged.get_column("expectancy").to_list())
    oos_exp_norm = normalize_series(merged.get_column("_oos_exp_calc").to_list())
    win_rate_norm = normalize_series(merged.get_column("win_rate").to_list())
    profit_factor_norm = normalize_series(
        [_safe_float(v) for v in merged.get_column("profit_factor").to_list()]
    )
    frequency_norm = normalize_series(
        merged.get_column("signal_frequency").cast(pl.Float64).to_list()
    )
    wf_stability_norm = normalize_series(merged.get_column("_wf_stability").to_list())
    robustness_norm = normalize_series(
        [1.0 if r else 0.0 for r in merged.get_column("is_robust").to_list()]
    )

    # ------------------------------------------------------------------
    # Detect suspicious rules and collect ancillary columns.
    # ------------------------------------------------------------------
    win_rates = [float(w) for w in merged.get_column("win_rate").to_list()]
    frequencies = [int(f) for f in merged.get_column("signal_frequency").to_list()]
    profit_factors = [_safe_float(v) for v in merged.get_column("profit_factor").to_list()]
    suspicious_flags = _detect_suspicious_rules(win_rates, frequencies, profit_factors)
    is_robust_list = merged.get_column("is_robust").to_list()
    combination_sizes = [int(s) for s in merged.get_column("combination_size").to_list()]

    # ------------------------------------------------------------------
    # Build composite scores.
    # ------------------------------------------------------------------
    composite: list[float] = []
    for i, (e, o, wr, p, f, wfs, r) in enumerate(
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
            (_EXPECTANCY_WEIGHT * e)
            + (_OOS_EXPECTANCY_WEIGHT * o)
            + (_WIN_RATE_WEIGHT * wr)
            + (_PROFIT_FACTOR_WEIGHT * p)
            + (_FREQUENCY_WEIGHT * f)
            + (_WF_STABILITY_WEIGHT * wfs)
            + (_ROBUSTNESS_WEIGHT * r)
        )

        # Complexity penalty: prefer fewer features when quality is similar.
        size = combination_sizes[i]
        if size > 2:
            score -= _COMPLEXITY_PENALTY_PER_EXTRA_FEATURE * (size - 2)

        # Suspicious-rule confidence penalty.
        if suspicious_flags[i]:
            penalty = (
                _SUSPICIOUS_PENALTY_ROBUST if is_robust_list[i] else _SUSPICIOUS_PENALTY_NOT_ROBUST
            )
            score *= 1.0 - penalty

        composite.append(max(0.0, score))

    ranked = merged.with_columns([pl.Series("composite_score", composite, dtype=pl.Float64)])

    # Sort deterministically: primary desc by score, secondary asc by label.
    output_cols = [
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
        "composite_score",
    ]

    sorted_ranked = (
        ranked.sort(["composite_score", "rule_label"], descending=[True, False])
        .with_row_index("rank", offset=1)
        .with_columns(
            [
                pl.col("composite_score").round(6),
                pl.col("win_rate").round(6),
                pl.col("expectancy").round(6),
                pl.col("average_return").round(6),
                pl.col("median_return").round(6),
            ]
        )
        .select([c for c in output_cols if c in ranked.columns or c == "rank"])
    )

    # Deduplicate near-identical rules (keeps highest-scoring representative).
    return _deduplicate_ranked(sorted_ranked)


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
            "composite_score": pl.Series([], dtype=pl.Float64),
        }
    )
