"""Strategy research report generation.

Writes four report files to ``reports/lab/strategy/``:

- ``leaderboard.csv``    – strategies ranked by composite score
- ``summary.json``       – high-level summary (top strategy, counts)
- ``metrics.csv``        – full scorecard for every strategy
- ``top_strategy.json``  – detailed metrics for the best-ranked strategy

Previous runs are not overwritten: when a file already exists, a timestamp
suffix is appended to the new filename.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


def write_strategy_reports(
    leaderboard: pl.DataFrame,
    metrics_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Write all strategy research report files to *output_dir*.

    Args:
        leaderboard: Ranked DataFrame produced by the research engine.
        metrics_df: Full per-strategy scorecard DataFrame.
        output_dir: Destination directory.  Created if it does not exist.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(leaderboard, output_dir / "leaderboard.csv")
    _write_csv(metrics_df, output_dir / "metrics.csv")

    summary = _build_summary(leaderboard, metrics_df)
    _write_json(summary, output_dir / "summary.json")

    if leaderboard.height > 0:
        top = _build_top_strategy(leaderboard)
        _write_json(top, output_dir / "top_strategy.json")
    else:
        _write_json({}, output_dir / "top_strategy.json")

    logger.info("Strategy reports written to %s", output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_path(path: Path) -> Path:
    """Return *path* unchanged if it does not exist; otherwise append a timestamp."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{stem}_{ts}{suffix}")


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    dest = _safe_path(path)
    df.write_csv(dest)
    logger.info("Written %s (%d rows)", dest, df.height)


def _write_json(data: dict[str, Any], path: Path) -> None:
    dest = _safe_path(path)
    dest.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Written %s", dest)


def _json_default(obj: object) -> object:
    """Fallback JSON serialiser for types not natively handled by ``json``."""
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _build_summary(leaderboard: pl.DataFrame, metrics_df: pl.DataFrame) -> dict[str, Any]:
    """Build the high-level summary dictionary."""
    n_strategies = metrics_df.height
    top_name: str | None = None
    top_score: float | None = None

    if leaderboard.height > 0 and "composite_score" in leaderboard.columns:
        top_row = leaderboard.row(0, named=True)
        top_name = str(top_row["strategy"])
        raw_score = top_row.get("composite_score")
        top_score = float(raw_score) if raw_score is not None else None

    return {
        "strategies_evaluated": n_strategies,
        "top_strategy": top_name,
        "top_composite_score": top_score,
        "strategy_names": metrics_df["strategy"].to_list() if metrics_df.height > 0 else [],
    }


def _build_top_strategy(leaderboard: pl.DataFrame) -> dict[str, Any]:
    """Extract the top-ranked strategy metrics as a plain dict."""
    top_row = leaderboard.row(0, named=True)
    # Convert all values to JSON-serialisable types
    result: dict[str, Any] = {}
    for key, value in top_row.items():
        if isinstance(value, float) and value != value:
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value
    return result
