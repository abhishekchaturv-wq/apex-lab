"""Signal discovery dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from apex_lab.features import FeatureEngine
from apex_lab.research.alpha.registry import CATEGORY_ORDER
from apex_lab.research.alpha.score import score_trades
from apex_lab.research.alpha.weights import (
    DEFAULT_CONTEXT_LEADERBOARD_PATH,
    DEFAULT_WEIGHTS_FILENAME,
    ensure_weights_file,
)
from apex_lab.research.backtest.backtester import VOLATILITY_HIGH_THRESHOLD
from apex_lab.research.context.engine import _enrich_ohlcv
from apex_lab.research.context.registry import get_registry
from apex_lab.research.factors.factor_engine import FACTOR_REGISTRY
from apex_lab.research.signal_dataset.labels import (
    SignalLabelConfig,
    append_signal_classes,
    label_columns,
)
from apex_lab.research.signal_dataset.schema import build_schema_payload
from apex_lab.research.signal_dataset.writer import DEFAULT_OUTPUT_DIR, write_dataset_artifacts

_QUANTILE_PRECISION = 6


@dataclass(frozen=True)
class SignalDatasetConfig:
    """Configuration for signal dataset generation."""

    symbol: str = "UNKNOWN"
    interval: str = "UNKNOWN"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    session_id: str | None = None
    drop_duplicate_timestamps: bool = True
    label_config: SignalLabelConfig = field(default_factory=SignalLabelConfig)
    alpha_weights_path: Path = Path("reports/lab/alpha") / DEFAULT_WEIGHTS_FILENAME
    context_leaderboard_path: Path = DEFAULT_CONTEXT_LEADERBOARD_PATH
    generation_timestamp: str | None = None


@dataclass(frozen=True)
class SignalDatasetBuildResult:
    """In-memory result and written artifact paths."""

    dataset: pl.DataFrame
    schema: dict[str, Any]
    summary: dict[str, Any]
    feature_columns: list[str]
    label_columns: list[str]
    metadata_columns: list[str]
    output_paths: dict[str, Path]


class SignalDatasetBuilder:
    """Build candle-level supervised dataset for signal discovery."""

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self.feature_engine = feature_engine or FeatureEngine()

    def build(self, historical_df: pl.DataFrame, config: SignalDatasetConfig) -> SignalDatasetBuildResult:
        """Build dataset and write all required artifacts."""
        if "timestamp" not in historical_df.columns:
            raise ValueError("historical_df must include a timestamp column")

        duplicate_timestamps = _count_duplicate_timestamps(historical_df)
        base = historical_df.sort("timestamp")
        if config.drop_duplicate_timestamps:
            base = base.unique(subset=["timestamp"], keep="first", maintain_order=True)

        featured = self.feature_engine.compute(base)
        enriched = _enrich_ohlcv(featured)
        enriched = _append_factor_outputs(enriched)
        enriched = _append_context_columns(enriched)
        enriched = _append_alpha_columns(
            enriched,
            weights_path=config.alpha_weights_path,
            context_leaderboard_path=config.context_leaderboard_path,
        )
        enriched = _append_metadata_columns(enriched, config)
        dataset = append_signal_classes(enriched, config.label_config)

        labels = label_columns(config.label_config)
        metadata = ["symbol", "interval", "timestamp", "session_id", "weekday", "market_regime"]
        raw_columns = ["open", "high", "low", "close", "volume"]
        feature_columns = [
            column
            for column in dataset.columns
            if column not in labels and column not in metadata and column not in raw_columns
        ]

        schema_payload = build_schema_payload(dataset, feature_columns, labels, metadata)
        quantiles_payload = _build_quantiles_payload(dataset, feature_columns)
        summary_payload = _build_summary_payload(
            dataset,
            feature_columns=feature_columns,
            labels=labels,
            symbol=config.symbol,
            interval=config.interval,
            duplicate_timestamps=duplicate_timestamps,
            generation_timestamp=config.generation_timestamp,
        )

        output_paths = write_dataset_artifacts(
            dataset,
            output_dir=config.output_dir,
            schema_payload=schema_payload,
            summary_payload=summary_payload,
            feature_columns=feature_columns,
            quantiles_payload=quantiles_payload,
        )

        return SignalDatasetBuildResult(
            dataset=dataset,
            schema=schema_payload,
            summary=summary_payload,
            feature_columns=feature_columns,
            label_columns=labels,
            metadata_columns=metadata,
            output_paths=output_paths,
        )


def _append_factor_outputs(df: pl.DataFrame) -> pl.DataFrame:
    result = df
    for name, factor in FACTOR_REGISTRY.items():
        result = factor.compute(result)
        result = result.with_columns(factor.signal(result).cast(pl.Int8).alias(f"factor_{name.lower()}_signal"))
    return result


def _append_context_columns(df: pl.DataFrame) -> pl.DataFrame:
    registry = get_registry()
    with_context = df
    for feature in registry.values():
        with_context = feature.compute(with_context)

    label_columns_expr = [feature.label(with_context).alias(f"ctx_{name}") for name, feature in registry.items()]
    numeric_columns_expr = [
        feature.numeric(with_context).cast(pl.Float64).alias(f"num_{name}")
        for name, feature in registry.items()
    ]
    return with_context.with_columns(label_columns_expr).with_columns(numeric_columns_expr)


def _append_alpha_columns(df: pl.DataFrame, weights_path: Path, context_leaderboard_path: Path) -> pl.DataFrame:
    weights = ensure_weights_file(weights_path, context_leaderboard_path)
    context_columns = [column for column in df.columns if column.startswith("ctx_")]

    score_source = df.select(["timestamp", *context_columns]).rename({"timestamp": "entry_time"})
    scored = score_trades(score_source, weights)

    keep_columns = [
        "entry_time",
        "alpha_score",
        "score_bucket",
        *[f"{category}_score" for category in CATEGORY_ORDER],
    ]
    return df.join(scored.select(keep_columns), left_on="timestamp", right_on="entry_time", how="left")


def _append_metadata_columns(df: pl.DataFrame, config: SignalDatasetConfig) -> pl.DataFrame:
    session_id = config.session_id or f"{config.symbol}:{config.interval}"
    return df.with_columns(
        [
            pl.lit(config.symbol).alias("symbol"),
            pl.lit(config.interval).alias("interval"),
            pl.lit(session_id).alias("session_id"),
            pl.col("timestamp").dt.weekday().alias("weekday"),
            pl.when(pl.col("close") > pl.col("ema_200"))
            .then(pl.lit("above_ema200"))
            .otherwise(pl.lit("below_ema200"))
            .cast(pl.Utf8)
            .alias("trend_regime"),
            pl.when(pl.col("atr_pct").is_null())
            .then(pl.lit("unknown"))
            .when(pl.col("atr_pct") >= VOLATILITY_HIGH_THRESHOLD)
            .then(pl.lit("high"))
            .otherwise(pl.lit("low"))
            .cast(pl.Utf8)
            .alias("volatility_regime"),
        ]
    ).with_columns(
        pl.concat_str([pl.col("trend_regime"), pl.lit("_"), pl.col("volatility_regime")]).alias(
            "market_regime"
        )
    )


def _build_summary_payload(
    dataset: pl.DataFrame,
    feature_columns: list[str],
    labels: list[str],
    symbol: str,
    interval: str,
    duplicate_timestamps: int,
    generation_timestamp: str | None,
) -> dict[str, Any]:
    start_ts = dataset.get_column("timestamp").min() if dataset.height else None
    end_ts = dataset.get_column("timestamp").max() if dataset.height else None
    return {
        "row_count": dataset.height,
        "feature_count": len(feature_columns),
        "label_count": len(labels),
        "date_range": {
            "start": start_ts.isoformat() if hasattr(start_ts, "isoformat") else None,
            "end": end_ts.isoformat() if hasattr(end_ts, "isoformat") else None,
        },
        "symbol": symbol,
        "interval": interval,
        "missing_values": {
            column: int(dataset.get_column(column).null_count())
            for column in dataset.columns
            if int(dataset.get_column(column).null_count()) > 0
        },
        "duplicate_timestamps": duplicate_timestamps,
        "generation_timestamp": generation_timestamp or datetime.now(UTC).isoformat(),
    }


def _build_quantiles_payload(
    dataset: pl.DataFrame,
    feature_columns: list[str],
    bins: int = 4,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "_meta": {
            "bins": bins,
            "label_scheme": "zero_based",
        }
    }
    quantile_levels = np.linspace(0.0, 1.0, bins + 1)

    for column in feature_columns:
        series = dataset.get_column(column)
        if not series.dtype.is_numeric():
            continue
        values = np.asarray(series.to_list(), dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        edges = np.quantile(finite, quantile_levels)
        if len(edges) < 2:
            continue
        payload[column] = {
            f"q{index}": [
                round(float(edges[index]), _QUANTILE_PRECISION),
                round(float(edges[index + 1]), _QUANTILE_PRECISION),
            ]
            for index in range(len(edges) - 1)
        }
    return payload


def _count_duplicate_timestamps(df: pl.DataFrame) -> int:
    counts = (
        df.group_by("timestamp")
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum().alias("duplicate_count"))
    )
    if counts.height == 0:
        return 0
    value = counts[0, "duplicate_count"]
    return int(value or 0)
