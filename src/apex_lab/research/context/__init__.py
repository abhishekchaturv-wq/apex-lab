"""Alpha Discovery Engine — market context research package.

Identifies which market context variables (trend, volatility, momentum, VWAP,
market structure, session and time context) genuinely improve EMA crossover
trading performance.

Example::

    import polars as pl
    from apex_lab.research.context.engine import run_context_research

    df = pl.read_parquet("data/raw/30minute/NIFTY BANK.parquet")
    run_context_research(df)
"""

from __future__ import annotations

from apex_lab.research.context.engine import DEFAULT_OUTPUT_DIR, run_context_research
from apex_lab.research.context.registry import (
    ContextFeature,
    get_registry,
    register_context_feature,
)

__all__ = [
    "ContextFeature",
    "DEFAULT_OUTPUT_DIR",
    "get_registry",
    "register_context_feature",
    "run_context_research",
]
