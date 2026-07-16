"""Signal discovery engine orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from apex_lab.research.signal_discovery.combination import (
    CombinationConfig,
    rank_feature_combinations,
)
from apex_lab.research.signal_discovery.importance import (
    ALL_TARGETS,
    analyze_categorical_features,
    analyze_feature_importance,
)
from apex_lab.research.signal_discovery.report import build_summary_payload, write_reports
from apex_lab.research.signal_discovery.statistics import (
    correlation,
    normalize_series,
    stability_label,
)

DEFAULT_OUTPUT_DIR = Path("reports/lab/signal_discovery")


@dataclass(frozen=True)
class SignalDiscoveryResult:
    """Result bundle produced by signal discovery mode."""

    feature_importance: pl.DataFrame
    top_combinations: pl.DataFrame
    categorical_analysis: pl.DataFrame
    stability_report: dict[str, Any]
    summary: dict[str, Any]


def run_signal_discovery(
    dataset_paths: list[Path] | Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    combination_config: CombinationConfig = CombinationConfig(),
) -> SignalDiscoveryResult:
    """Run feature importance and signal discovery analysis."""
    paths = dataset_paths if isinstance(dataset_paths, list) else [dataset_paths]
    if not paths:
        raise ValueError("at least one dataset path is required")

    frames: list[pl.DataFrame] = []
    for path in paths:
        frame = pl.read_parquet(path)
        if "symbol" not in frame.columns:
            frame = frame.with_columns(pl.lit(path.stem).alias("symbol"))
        frames.append(frame)

    dataset = pl.concat(frames, how="diagonal_relaxed")
    feature_columns = _resolve_feature_columns(dataset)

    feature_importance = analyze_feature_importance(dataset, feature_columns)
    categorical_analysis = analyze_categorical_features(dataset, feature_columns)
    top_combinations = rank_feature_combinations(
        dataset,
        feature_importance,
        config=combination_config,
    )
    stability_report = _build_stability_report(dataset, feature_importance)
    summary = build_summary_payload(
        feature_importance,
        top_combinations,
        stability_report,
        categorical_analysis,
    )

    write_reports(
        output_dir=output_dir,
        feature_importance=feature_importance,
        combinations=top_combinations,
        stability_report=stability_report,
        summary=summary,
    )

    return SignalDiscoveryResult(
        feature_importance=feature_importance,
        top_combinations=top_combinations,
        categorical_analysis=categorical_analysis,
        stability_report=stability_report,
        summary=summary,
    )


def _resolve_feature_columns(df: pl.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "symbol",
        "interval",
        "session_id",
        "weekday",
        "trend_regime",
        "volatility_regime",
        "market_regime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *ALL_TARGETS,
        "future_high_return",
        "future_low_return",
        "maximum_favorable_excursion",
        "maximum_adverse_excursion",
        "direction",
        "label_strong_bull_move",
        "label_bull_move",
        "label_neutral",
        "label_bear_move",
        "label_strong_bear_move",
    }
    return [column for column in df.columns if column not in excluded]


def _build_stability_report(df: pl.DataFrame, importance: pl.DataFrame, top_n: int = 20) -> dict[str, Any]:
    if "symbol" not in df.columns:
        return {"features": [], "symbols": []}

    symbols = sorted(str(value) for value in df.get_column("symbol").drop_nulls().unique().to_list())
    if not symbols:
        return {"features": [], "symbols": []}

    features = importance.head(top_n).get_column("feature").to_list()
    rows: list[dict[str, Any]] = []

    for feature in features:
        if feature not in df.columns:
            continue
        per_symbol_strength: list[float] = []
        symbol_strength: dict[str, float] = {}
        for symbol in symbols:
            subset = df.filter(pl.col("symbol") == symbol)
            if subset.height < 10:
                continue
            if subset.get_column(feature).dtype.is_numeric() and "future_return_20" in subset.columns:
                strength = correlation(
                    subset.get_column(feature),
                    subset.get_column("future_return_20"),
                    method="spearman",
                )
                if strength is not None:
                    absolute_strength = abs(strength)
                    per_symbol_strength.append(absolute_strength)
                    symbol_strength[symbol] = round(absolute_strength, 6)

        if not per_symbol_strength:
            score = 0.0
        elif len(per_symbol_strength) == 1:
            score = 1.0
        else:
            mean = float(np.mean(per_symbol_strength))
            std = float(np.std(per_symbol_strength))
            cv = std / max(mean, 1e-9)
            score = max(0.0, min(1.0, 1.0 - cv))

        rows.append(
            {
                "feature": feature,
                "score": round(score, 6),
                "stability": stability_label(score),
                "symbol_strength": symbol_strength,
            }
        )

    score_norm = normalize_series([row["score"] for row in rows])
    for index, score in enumerate(score_norm):
        rows[index]["score_normalized"] = round(score, 6)

    return {
        "symbols": symbols,
        "features": sorted(rows, key=lambda item: item["score"], reverse=True),
    }
