"""Research lab script for EMA crossover forward-return analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_DATA_PATH = Path("data/raw/30minute/NIFTY BANK.parquet")
DEFAULT_CSV_OUTPUT = Path("reports/lab/csv/ema_cross_returns.csv")
DEFAULT_JSON_OUTPUT = Path("reports/lab/json/ema_cross_summary.json")
FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run EMA crossover forward-return research.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input OHLCV parquet file.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="Path to the CSV report for bullish crossover forward returns.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Path to the JSON summary report.",
    )
    return parser.parse_args()


def load_ohlcv(path: Path) -> pl.DataFrame:
    """Load and validate an OHLCV parquet file."""
    df = pl.read_parquet(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"input parquet is missing required columns: {missing}")

    return df.sort("timestamp")


def compute_ema_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Append EMA values, crossover signals, and forward-return columns."""
    enriched = df.with_columns(
        [
            pl.col("close").ewm_mean(span=20, adjust=False).alias("ema_20"),
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
        ]
    ).with_columns(
        [
            (
                (pl.col("ema_20") > pl.col("ema_50"))
                & (pl.col("ema_20").shift(1) <= pl.col("ema_50").shift(1))
            )
            .fill_null(False)
            .alias("bullish_crossover"),
            (
                (pl.col("ema_20") < pl.col("ema_50"))
                & (pl.col("ema_20").shift(1) >= pl.col("ema_50").shift(1))
            )
            .fill_null(False)
            .alias("bearish_crossover"),
        ]
    )

    forward_return_columns = [
        ((pl.col("close").shift(-horizon) / pl.col("close")) - 1.0).mul(100.0).alias(
            f"forward_return_{horizon}"
        )
        for horizon in FORWARD_RETURN_HORIZONS
    ]
    return enriched.with_columns(forward_return_columns)


def build_bullish_returns_report(df: pl.DataFrame) -> pl.DataFrame:
    """Extract bullish crossover rows for CSV export."""
    return df.filter(pl.col("bullish_crossover")).select(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_20",
            "ema_50",
            pl.lit("bullish_crossover").alias("signal"),
            *[f"forward_return_{horizon}" for horizon in FORWARD_RETURN_HORIZONS],
        ]
    )


def summarize_forward_returns(
    bullish_returns: pl.DataFrame,
    bullish_crossovers: int,
    bearish_crossovers: int,
) -> dict[str, Any]:
    """Build summary metrics for each forward-return horizon."""
    forward_returns: dict[str, Any] = {}

    for horizon in FORWARD_RETURN_HORIZONS:
        column = f"forward_return_{horizon}"
        values = bullish_returns.get_column(column).drop_nulls()
        num_signals = len(values)

        if num_signals == 0:
            forward_returns[str(horizon)] = {
                "num_signals": 0,
                "win_rate": None,
                "mean_return": None,
                "median_return": None,
                "standard_deviation": None,
                "maximum_gain": None,
                "maximum_loss": None,
            }
            continue

        wins = values.gt(0).sum()
        forward_returns[str(horizon)] = {
            "num_signals": num_signals,
            "win_rate": float(wins / num_signals * 100.0),
            "mean_return": float(values.mean()),
            "median_return": float(values.median()),
            "standard_deviation": float(values.std(ddof=0)),
            "maximum_gain": float(values.max()),
            "maximum_loss": float(values.min()),
        }

    return {
        "bullish_crossovers": bullish_crossovers,
        "bearish_crossovers": bearish_crossovers,
        "forward_returns": forward_returns,
    }


def write_reports(
    bullish_returns: pl.DataFrame,
    summary: dict[str, Any],
    csv_output: Path,
    json_output: Path,
) -> None:
    """Persist CSV and JSON research outputs."""
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    bullish_returns.write_csv(csv_output)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_research_lab(
    data_path: Path,
    csv_output: Path = DEFAULT_CSV_OUTPUT,
    json_output: Path = DEFAULT_JSON_OUTPUT,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run EMA crossover research and write reports to disk."""
    df = load_ohlcv(data_path)
    enriched = compute_ema_signals(df)
    bullish_returns = build_bullish_returns_report(enriched)
    summary = summarize_forward_returns(
        bullish_returns=bullish_returns,
        bullish_crossovers=int(enriched.get_column("bullish_crossover").sum()),
        bearish_crossovers=int(enriched.get_column("bearish_crossover").sum()),
    )
    write_reports(bullish_returns, summary, csv_output, json_output)
    return bullish_returns, summary


def main() -> None:
    """Execute the research lab CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    bullish_returns, summary = run_research_lab(args.data, args.csv_output, args.json_output)
    logger.info(
        "Wrote %d bullish crossover rows to %s",
        bullish_returns.height,
        args.csv_output,
    )
    logger.info(
        "Summary written to %s (%d bullish, %d bearish crossovers)",
        args.json_output,
        summary["bullish_crossovers"],
        summary["bearish_crossovers"],
    )


if __name__ == "__main__":
    main()
