"""Portfolio report writing utilities.

Writes the six CSV/JSON artefacts produced by a portfolio simulation run to
``reports/lab/portfolio/`` (or a caller-supplied directory).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_PORTFOLIO_OUTPUT_DIR: Path = Path("reports/lab/portfolio")


def write_portfolio_reports(
    equity_df: pl.DataFrame,
    summary: dict[str, Any],
    monthly_returns: pl.DataFrame,
    yearly_returns: pl.DataFrame,
    rolling_sharpe: pl.DataFrame,
    rolling_drawdown: pl.DataFrame,
    output_dir: Path = DEFAULT_PORTFOLIO_OUTPUT_DIR,
) -> None:
    """Persist all portfolio simulation outputs to *output_dir*.

    Files written:

    - ``equity.csv``
    - ``summary.json``
    - ``monthly_returns.csv``
    - ``yearly_returns.csv``
    - ``rolling_sharpe.csv``
    - ``rolling_drawdown.csv``

    Args:
        equity_df: Per-trade equity DataFrame from
            :func:`~apex_lab.research.portfolio.portfolio.simulate_portfolio`.
        summary: Metrics dictionary from
            :func:`~apex_lab.research.portfolio.metrics.compute_portfolio_metrics`.
        monthly_returns: Monthly aggregation from
            :func:`~apex_lab.research.portfolio.metrics.compute_monthly_returns`.
        yearly_returns: Yearly aggregation from
            :func:`~apex_lab.research.portfolio.metrics.compute_yearly_returns`.
        rolling_sharpe: Rolling Sharpe series from
            :func:`~apex_lab.research.portfolio.metrics.compute_rolling_sharpe`.
        rolling_drawdown: Rolling drawdown series from
            :func:`~apex_lab.research.portfolio.metrics.compute_rolling_drawdown`.
        output_dir: Directory where all output files will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / "equity.csv"
    summary_path = output_dir / "summary.json"
    monthly_path = output_dir / "monthly_returns.csv"
    yearly_path = output_dir / "yearly_returns.csv"
    rolling_sharpe_path = output_dir / "rolling_sharpe.csv"
    rolling_drawdown_path = output_dir / "rolling_drawdown.csv"

    equity_df.write_csv(equity_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    monthly_returns.write_csv(monthly_path)
    yearly_returns.write_csv(yearly_path)
    rolling_sharpe.write_csv(rolling_sharpe_path)
    rolling_drawdown.write_csv(rolling_drawdown_path)

    logger.info("Portfolio equity written to %s (%d rows)", equity_path, equity_df.height)
    logger.info("Portfolio summary written to %s", summary_path)
    logger.info(
        "Monthly returns written to %s (%d rows)", monthly_path, monthly_returns.height
    )
    logger.info(
        "Yearly returns written to %s (%d rows)", yearly_path, yearly_returns.height
    )
    logger.info(
        "Rolling Sharpe written to %s (%d rows)", rolling_sharpe_path, rolling_sharpe.height
    )
    logger.info(
        "Rolling drawdown written to %s (%d rows)",
        rolling_drawdown_path,
        rolling_drawdown.height,
    )
