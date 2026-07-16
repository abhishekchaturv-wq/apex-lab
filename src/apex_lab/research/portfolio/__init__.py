"""Portfolio simulation and position sizing module."""

from __future__ import annotations

from apex_lab.research.portfolio.metrics import (
    compute_monthly_returns,
    compute_portfolio_metrics,
    compute_rolling_drawdown,
    compute_rolling_sharpe,
    compute_yearly_returns,
)
from apex_lab.research.portfolio.portfolio import PositionSizing, simulate_portfolio
from apex_lab.research.portfolio.report import (
    DEFAULT_PORTFOLIO_OUTPUT_DIR,
    write_portfolio_reports,
)

__all__ = [
    "simulate_portfolio",
    "PositionSizing",
    "compute_portfolio_metrics",
    "compute_monthly_returns",
    "compute_yearly_returns",
    "compute_rolling_sharpe",
    "compute_rolling_drawdown",
    "write_portfolio_reports",
    "DEFAULT_PORTFOLIO_OUTPUT_DIR",
]
