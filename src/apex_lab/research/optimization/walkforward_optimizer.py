"""Walk-forward parameter optimization for EMA crossover strategies.

Evaluates multiple EMA crossover parameter pairs using true walk-forward
validation: a rolling train/test split is applied across the full dataset,
backtested with the event-driven engine, and results are aggregated into a
ranked leaderboard.

Example:
    >>> import polars as pl
    >>> from apex_lab.research.optimization.walkforward_optimizer import optimize
    >>> df = pl.read_parquet("data/raw/30minute/NIFTY BANK.parquet")
    >>> summary, leaderboard, best = optimize(df)
"""

from __future__ import annotations

import bisect
import calendar
import datetime
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import polars as pl

from apex_lab.research.backtest.backtester import compute_metrics, run_backtest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EMA (fast, slow) pairs to evaluate.
EMA_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 20),
    (8, 21),
    (10, 30),
    (13, 34),
    (20, 50),
    (50, 200),
)

#: Default train window length in calendar years.
TRAIN_YEARS: int = 2

#: Default out-of-sample test window in calendar months.
TEST_MONTHS: int = 6

#: Default advance step between consecutive windows in calendar months.
ADVANCE_MONTHS: int = 6

#: ATR look-back for the True Range rolling mean.
_ATR_PERIOD: int = 14

#: Rolling window size for the ATR percentile rank.
_ATR_PERCENTILE_WINDOW: int = 100

#: Default directory for optimization output files.
DEFAULT_OUTPUT_DIR: Path = Path("reports/lab/walkforward")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _add_months(dt: datetime.date, months: int) -> datetime.date:
    """Return *dt* advanced by *months* calendar months.

    The day is clamped to the last valid day of the target month so that, for
    example, 31 January + 1 month = 28/29 February.

    Args:
        dt: Starting date.
        months: Number of calendar months to add (must be non-negative).

    Returns:
        A new :class:`datetime.date` instance.
    """
    total_months = dt.month - 1 + months
    year = dt.year + total_months // 12
    month = total_months % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _to_date(value: datetime.date | datetime.datetime) -> datetime.date:
    """Normalise a *datetime* to a plain :class:`datetime.date`.

    Args:
        value: A :class:`datetime.date` or :class:`datetime.datetime`.

    Returns:
        The corresponding :class:`datetime.date`.
    """
    return value.date() if isinstance(value, datetime.datetime) else value


# ---------------------------------------------------------------------------
# Signal building
# ---------------------------------------------------------------------------


def _rolling_percentile_rank(series: pl.Series, window: int) -> pl.Series:
    """Compute the rolling percentile rank (0–100) for *series*.

    Null values in *series* remain null in the output.  The window tracks the
    most recent *window* non-null observations.

    Args:
        series: Input numeric series (may contain nulls).
        window: Maximum number of non-null values in the rolling window.

    Returns:
        A :class:`polars.Series` of :class:`polars.Float64` percentile ranks.
    """
    values = series.to_list()
    out: list[float | None] = [None] * len(values)
    active_window: deque[float] = deque()
    sorted_window: list[float] = []

    for index, current in enumerate(values):
        if current is not None:
            bisect.insort(sorted_window, current)
            active_window.append(current)

        if len(active_window) > window:
            expired = active_window.popleft()
            expired_index = bisect.bisect_left(sorted_window, expired)
            del sorted_window[expired_index]

        if current is None or not sorted_window:
            continue

        rank_position = bisect.bisect_right(sorted_window, current)
        out[index] = rank_position / len(sorted_window) * 100.0

    return pl.Series(out, dtype=pl.Float64)


def _build_ema_signals(df: pl.DataFrame, fast_period: int, slow_period: int) -> pl.DataFrame:
    """Compute EMA crossover signals for the given fast/slow EMA periods.

    The function appends the columns required by
    :func:`~apex_lab.research.backtest.backtester.run_backtest`:
    ``bullish_crossover``, ``bearish_crossover``, ``ema_200``, and
    ``atr_pct``.  A trend EMA-200 is always computed for regime detection
    regardless of the *fast_period* / *slow_period* values.

    Args:
        df: Raw OHLCV DataFrame with at least ``timestamp``, ``open``,
            ``high``, ``low``, ``close``, and ``volume`` columns.
        fast_period: Span for the fast EMA.
        slow_period: Span for the slow EMA.

    Returns:
        The input DataFrame enriched with signal and regime columns.
    """
    fast_col = f"_ema_fast_{fast_period}"
    slow_col = f"_ema_slow_{slow_period}"

    epsilon = pl.lit(1e-9)
    prev_close = pl.col("close").shift(1)
    true_range = (
        pl.max_horizontal(
            [
                pl.col("high") - pl.col("low"),
                (pl.col("high") - prev_close).abs(),
                (pl.col("low") - prev_close).abs(),
            ]
        )
        .cast(pl.Float64)
        .alias("_tr")
    )

    enriched = df.with_columns(
        [
            pl.col("close").ewm_mean(span=fast_period, adjust=False).alias(fast_col),
            pl.col("close").ewm_mean(span=slow_period, adjust=False).alias(slow_col),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200"),
            true_range,
        ]
    ).with_columns(
        [
            pl.col("_tr").rolling_mean(window_size=_ATR_PERIOD).alias("atr_14"),
            (
                (pl.col(fast_col) > pl.col(slow_col))
                & (pl.col(fast_col).shift(1) <= pl.col(slow_col).shift(1))
            )
            .fill_null(False)
            .alias("bullish_crossover"),
            (
                (pl.col(fast_col) < pl.col(slow_col))
                & (pl.col(fast_col).shift(1) >= pl.col(slow_col).shift(1))
            )
            .fill_null(False)
            .alias("bearish_crossover"),
        ]
    )

    atr_pct_series = _rolling_percentile_rank(enriched["atr_14"], _ATR_PERCENTILE_WINDOW)
    return enriched.with_columns(
        [
            atr_pct_series.alias("atr_pct"),
            (pl.col("atr_14") / (pl.col("close") + epsilon) * 100.0).alias("atr_norm"),
        ]
    ).drop(["_tr"])


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


def _generate_windows(
    timestamps: pl.Series,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    advance_months: int = ADVANCE_MONTHS,
) -> list[tuple[datetime.date, datetime.date, datetime.date, datetime.date]]:
    """Generate walk-forward (train_start, train_end, test_start, test_end) tuples.

    Windows slide forward by *advance_months* until the test window would
    exceed the last available date.

    Args:
        timestamps: The ``timestamp`` column from the OHLCV DataFrame.
        train_years: Number of calendar years in each training window.
        test_months: Number of calendar months in each test window.
        advance_months: Step size in calendar months between consecutive
            training windows.

    Returns:
        A list of ``(train_start, train_end, test_start, test_end)`` tuples
        where each element is a :class:`datetime.date`.
    """
    dates = timestamps.cast(pl.Date)
    min_val = dates.min()
    max_val = dates.max()

    if min_val is None or max_val is None:
        return []

    min_date = _to_date(min_val)
    max_date = _to_date(max_val)

    windows: list[tuple[datetime.date, datetime.date, datetime.date, datetime.date]] = []
    train_start = min_date

    while True:
        train_end = _add_months(train_start, train_years * 12)
        test_start = train_end
        test_end = _add_months(test_start, test_months)
        if test_end > max_date:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start = _add_months(train_start, advance_months)

    return windows


# ---------------------------------------------------------------------------
# Per-window metric helpers
# ---------------------------------------------------------------------------


def _compute_cagr(trades: pl.DataFrame, years: float) -> float | None:
    """Compute the compound annual growth rate from a trade log.

    Uses the sum of ``return_pct`` values as the total return over *years*.

    Args:
        trades: Completed-trade DataFrame from :func:`run_backtest`.
        years: Duration of the evaluation window in decimal years.

    Returns:
        CAGR as a percentage, or ``None`` when the computation is not
        possible (no trades, non-positive duration, or arithmetic error).
    """
    if trades.is_empty() or years <= 0:
        return None
    total_return = float(trades["return_pct"].sum())
    try:
        cagr = ((1.0 + total_return / 100.0) ** (1.0 / years) - 1.0) * 100.0
    except (ValueError, ZeroDivisionError):
        return None
    return cagr


def _compute_sharpe(trades: pl.DataFrame, years: float) -> float | None:
    """Compute an annualised Sharpe ratio from trade-level returns.

    The ratio is scaled to annual frequency using the observed number of
    trades per year.  A risk-free rate of zero is assumed.

    Args:
        trades: Completed-trade DataFrame from :func:`run_backtest`.
        years: Duration of the evaluation window in decimal years.

    Returns:
        Annualised Sharpe ratio, or ``None`` when there are fewer than two
        trades or the standard deviation of returns is zero.
    """
    if trades.is_empty() or years <= 0 or trades.height < 2:
        return None
    returns = trades["return_pct"]
    std_ret = float(returns.std())
    if std_ret == 0.0:
        return None
    mean_ret = float(returns.mean())
    trades_per_year = trades.height / years
    return (mean_ret / std_ret) * (trades_per_year**0.5)


# ---------------------------------------------------------------------------
# Core optimization logic
# ---------------------------------------------------------------------------


def _empty_summary_df() -> pl.DataFrame:
    """Return a zero-row summary DataFrame with the expected schema.

    Returns:
        An empty :class:`polars.DataFrame` matching the summary schema.
    """
    return pl.DataFrame(
        {
            "window_id": pl.Series([], dtype=pl.Int64),
            "fast_ema": pl.Series([], dtype=pl.Int64),
            "slow_ema": pl.Series([], dtype=pl.Int64),
            "train_start": pl.Series([], dtype=pl.Utf8),
            "train_end": pl.Series([], dtype=pl.Utf8),
            "test_start": pl.Series([], dtype=pl.Utf8),
            "test_end": pl.Series([], dtype=pl.Utf8),
            "number_of_trades": pl.Series([], dtype=pl.Int64),
            "win_rate": pl.Series([], dtype=pl.Float64),
            "expectancy": pl.Series([], dtype=pl.Float64),
            "average_return": pl.Series([], dtype=pl.Float64),
            "profit_factor": pl.Series([], dtype=pl.Float64),
            "maximum_drawdown": pl.Series([], dtype=pl.Float64),
            "cagr": pl.Series([], dtype=pl.Float64),
            "sharpe_ratio": pl.Series([], dtype=pl.Float64),
        }
    )


def _build_leaderboard(summary: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-window metrics into a leaderboard ranked by EMA pair.

    Args:
        summary: DataFrame produced by :func:`run_walkforward_optimization`.

    Returns:
        A :class:`polars.DataFrame` with one row per EMA pair containing
        mean performance metrics and the number of evaluated windows.
    """
    if summary.is_empty():
        return pl.DataFrame(
            {
                "fast_ema": pl.Series([], dtype=pl.Int64),
                "slow_ema": pl.Series([], dtype=pl.Int64),
                "mean_profit_factor": pl.Series([], dtype=pl.Float64),
                "mean_expectancy": pl.Series([], dtype=pl.Float64),
                "mean_win_rate": pl.Series([], dtype=pl.Float64),
                "mean_drawdown": pl.Series([], dtype=pl.Float64),
                "number_of_windows": pl.Series([], dtype=pl.UInt32),
            }
        )

    return (
        summary.group_by(["fast_ema", "slow_ema"])
        .agg(
            [
                pl.col("profit_factor").mean().alias("mean_profit_factor"),
                pl.col("expectancy").mean().alias("mean_expectancy"),
                pl.col("win_rate").mean().alias("mean_win_rate"),
                pl.col("maximum_drawdown").mean().alias("mean_drawdown"),
                pl.len().alias("number_of_windows"),
            ]
        )
        .sort(["fast_ema", "slow_ema"])
    )


def _select_best_parameters(leaderboard: pl.DataFrame) -> dict[str, Any]:
    """Pick the best EMA pair from the leaderboard.

    Ranking criteria (in order of priority):

    1. Highest ``mean_profit_factor``
    2. Lowest ``mean_drawdown``
    3. Highest ``mean_expectancy``

    Rows with ``None`` in a primary sort key are pushed to the bottom.

    Args:
        leaderboard: DataFrame produced by :func:`_build_leaderboard`.

    Returns:
        A dictionary describing the winning EMA pair, or an empty dict when
        the leaderboard is empty.
    """
    if leaderboard.is_empty():
        return {}

    ranked = leaderboard.sort(
        ["mean_profit_factor", "mean_drawdown", "mean_expectancy"],
        descending=[True, False, True],
        nulls_last=True,
    )
    best = ranked.row(0, named=True)
    return {
        "fast_ema": int(best["fast_ema"]),
        "slow_ema": int(best["slow_ema"]),
        "mean_profit_factor": best["mean_profit_factor"],
        "mean_expectancy": best["mean_expectancy"],
        "mean_win_rate": best["mean_win_rate"],
        "mean_drawdown": best["mean_drawdown"],
        "number_of_windows": int(best["number_of_windows"]),
        "selection_criteria": [
            "highest mean_profit_factor",
            "lowest mean_drawdown",
            "highest mean_expectancy",
        ],
    }


def run_walkforward_optimization(
    df: pl.DataFrame,
    ema_pairs: tuple[tuple[int, int], ...] = EMA_PAIRS,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    advance_months: int = ADVANCE_MONTHS,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Run walk-forward optimization across all EMA pairs.

    For every walk-forward window the backtester is run on each EMA pair using
    only the out-of-sample test slice.  EMA values are computed on the full
    train+test slice so that the test window has historically accurate warm-up
    state.

    Args:
        df: Sorted OHLCV DataFrame with ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, and ``volume`` columns.
        ema_pairs: Sequence of ``(fast_period, slow_period)`` tuples to
            evaluate.
        train_years: Calendar years for the training (warm-up) window.
        test_months: Calendar months for the out-of-sample test window.
        advance_months: Calendar months to advance the window each iteration.

    Returns:
        A three-element tuple:

        - ``summary`` – one row per (window, EMA pair) with raw metrics.
        - ``leaderboard`` – one row per EMA pair with aggregated means.
        - ``best_params`` – dictionary describing the top-ranked EMA pair.

    Raises:
        ValueError: If the DataFrame contains no rows or the ``timestamp``
            column is missing.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    if df.is_empty():
        raise ValueError("DataFrame must not be empty.")

    windows = _generate_windows(df["timestamp"], train_years, test_months, advance_months)

    if not windows:
        logger.warning("Insufficient data to generate any walk-forward windows.")
        empty_summary = _empty_summary_df()
        empty_leaderboard = _build_leaderboard(empty_summary)
        return empty_summary, empty_leaderboard, {}

    rows: list[dict[str, Any]] = []

    for window_id, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        logger.info(
            "Window %d: train=%s–%s  test=%s–%s",
            window_id,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        # Slice train+test together to give EMAs their full warm-up history.
        window_df = df.filter(
            (pl.col("timestamp").cast(pl.Date) >= train_start)
            & (pl.col("timestamp").cast(pl.Date) < test_end)
        )
        if window_df.is_empty():
            continue

        for fast_ema, slow_ema in ema_pairs:
            enriched = _build_ema_signals(window_df, fast_ema, slow_ema)

            # Evaluate only on the out-of-sample test slice.
            test_df = enriched.filter(
                (pl.col("timestamp").cast(pl.Date) >= test_start)
                & (pl.col("timestamp").cast(pl.Date) < test_end)
            )

            trades = run_backtest(test_df, exit_mode="opposite_crossover")
            metrics = compute_metrics(trades)

            test_duration_years = (test_end - test_start).days / 365.25
            cagr = _compute_cagr(trades, test_duration_years)
            sharpe = _compute_sharpe(trades, test_duration_years)

            rows.append(
                {
                    "window_id": window_id,
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "number_of_trades": metrics["number_of_trades"],
                    "win_rate": metrics["win_rate"],
                    "expectancy": metrics["expectancy"],
                    "average_return": metrics["average_return"],
                    "profit_factor": metrics["profit_factor"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "cagr": cagr,
                    "sharpe_ratio": sharpe,
                }
            )

    summary = pl.DataFrame(rows) if rows else _empty_summary_df()
    leaderboard = _build_leaderboard(summary)
    best_params = _select_best_parameters(leaderboard)

    return summary, leaderboard, best_params


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_optimization_reports(
    summary: pl.DataFrame,
    leaderboard: pl.DataFrame,
    best_params: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Persist optimization results to *output_dir*.

    Creates three files:

    - ``summary.csv`` – one row per (window, EMA pair).
    - ``leaderboard.csv`` – one row per EMA pair with aggregated means.
    - ``best_parameters.json`` – top-ranked EMA pair with selection metadata.

    Args:
        summary: DataFrame produced by :func:`run_walkforward_optimization`.
        leaderboard: DataFrame produced by :func:`_build_leaderboard`.
        best_params: Dictionary produced by :func:`_select_best_parameters`.
        output_dir: Directory to write output files into.  Created if absent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    leaderboard_path = output_dir / "leaderboard.csv"
    best_path = output_dir / "best_parameters.json"

    summary.write_csv(summary_path)
    leaderboard.write_csv(leaderboard_path)
    best_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    logger.info("Summary written: %d rows → %s", summary.height, summary_path)
    logger.info("Leaderboard written → %s", leaderboard_path)
    logger.info("Best parameters written → %s", best_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimize(
    df: pl.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ema_pairs: tuple[tuple[int, int], ...] = EMA_PAIRS,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    advance_months: int = ADVANCE_MONTHS,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Run walk-forward optimization and write all reports to *output_dir*.

    This is the primary entry point for the optimization pipeline.  It calls
    :func:`run_walkforward_optimization` and then
    :func:`write_optimization_reports`.

    Args:
        df: Sorted OHLCV DataFrame.
        output_dir: Directory for output files.
        ema_pairs: EMA (fast, slow) pairs to evaluate.
        train_years: Calendar years in the training window.
        test_months: Calendar months in the test window.
        advance_months: Calendar months to advance between windows.

    Returns:
        A three-element tuple of ``(summary, leaderboard, best_params)``.
    """
    summary, leaderboard, best_params = run_walkforward_optimization(
        df,
        ema_pairs=ema_pairs,
        train_years=train_years,
        test_months=test_months,
        advance_months=advance_months,
    )
    write_optimization_reports(summary, leaderboard, best_params, output_dir)
    return summary, leaderboard, best_params
