"""Rendering logic for the Pine Script generator.

This module is responsible for assembling the Pine Script template blocks into
a complete, ready-to-deploy strategy file.  It does **not** perform any file
I/O; that responsibility belongs to :mod:`apex_lab.export.pine_generator`.
"""

from __future__ import annotations

from typing import Any

from apex_lab.export import template as T
from apex_lab.export.rule_translator import TranslatedSignal


def _category_weight(weights: list[dict[str, Any]], category: str) -> float:
    """Return the summed weight for a given alpha *category*.

    Args:
        weights: List of weight entries from ``weights.json``.
        category: The category key to sum (e.g. ``"trend"``).

    Returns:
        The total weight for the category, rounded to 6 decimal places.
    """
    return round(
        sum(w["weight"] for w in weights if w.get("category") == category),
        6,
    )


def render_inputs(fast_ema: int, slow_ema: int) -> str:
    """Render the strategy inputs block with walk-forward defaults.

    Args:
        fast_ema: Default value for the EMA Fast input.
        slow_ema: Default value for the EMA Slow input.

    Returns:
        Formatted Pine Script inputs block.
    """
    return T.INPUTS_TEMPLATE.format(
        ema_fast_default=fast_ema,
        ema_slow_default=slow_ema,
    )


def render_context_filters(best_features: dict[str, Any]) -> str:
    """Render context filter lines from the context discovery engine output.

    Each recognised feature key is translated into a named Pine Script boolean
    variable.  Unknown keys are emitted as comments so the generated script
    always compiles cleanly.

    Args:
        best_features: Mapping from feature name to its best bucket string,
            as loaded from ``best_features.json``.

    Returns:
        Formatted Pine Script context filters block.
    """
    FEATURE_MAP: dict[str, str] = {
        "atr_state": "atrNeutral     = atrVal / close * 100 < 1.5",
        "atr_percentile": "atrHighRegime  = atrVal > ta.sma(atrVal, 20)",
        "gap_pct_bucket": "gapFlat        = math.abs(open - close[1]) / close[1] * 100 < 0.3",
        "hour": "timeFilter     = hour >= 9 and hour <= 14",
        "day_of_week": "dayFilter      = dayofweek != dayofweek.sunday and dayofweek != dayofweek.saturday",
        "ema200_slope": "ema200SlopeLine = ema200 - ema200[5]",
        "or_position": "aboveORHigh    = close > ta.highest(high[1], 13)",
        "inside_or": "outsideOR      = close > ta.highest(high[1], 13) or close < ta.lowest(low[1], 13)",
    }

    extra_lines: list[str] = []
    for feature in best_features:
        if feature in FEATURE_MAP:
            extra_lines.append(FEATURE_MAP[feature])
        elif feature not in (
            "ema200_side",
            "dist_ema200",
            "dist_ema50",
            "ema50_slope",
            "vwap_side",
            "dist_vwap",
            "vwap_slope",
            "higher_high",
            "higher_low",
            "lower_high",
            "lower_low",
            "swing_distance",
            "rsi_bucket",
            "macd_hist_bucket",
            "adx_bucket",
            "roc10_bucket",
            "roc20_bucket",
            "realized_vol_20",
            "bb_width_pct",
            "month",
            "quarter",
        ):
            extra_lines.append(f"// context filter: {feature} = {best_features[feature]}")

    context_filter_lines = "\n".join(extra_lines) if extra_lines else "// (no additional filters)"
    return T.CONTEXT_FILTERS_TEMPLATE.format(context_filter_lines=context_filter_lines)


def render_alpha_score(weights: list[dict[str, Any]]) -> str:
    """Render the alpha score block using per-category weights.

    Only categories with a non-zero total weight are emitted.  This prevents
    redundant ``score = expr * 0`` lines in the generated Pine Script.

    Args:
        weights: List of weight entries loaded from ``weights.json``.

    Returns:
        Formatted Pine Script alpha score block.
    """
    # Map category -> (Pine expression, weight value)
    CATEGORY_EXPRS: dict[str, str] = {
        "trend": "close > ema200 ? 1.0 : 0.0",
        "momentum": "rsiVal > 50 ? 1.0 : 0.0",
        "volatility": "atrVal / close * 100 < 1.5 ? 1.0 : 0.0",
        "vwap": "close > vwapVal ? 1.0 : 0.0",
        "market_structure": "close > ta.highest(high[1], 5) ? 1.0 : 0.0",
        "opening_range": "close > ta.highest(high[1], 13) ? 1.0 : 0.0",
        "time": "hour >= 9 and hour <= 14 ? 1.0 : 0.0",
    }

    # Variable name for each category score
    CATEGORY_VARS: dict[str, str] = {
        "trend": "trendScore",
        "momentum": "momentumScore",
        "volatility": "volatilityScore",
        "vwap": "vwapScore",
        "market_structure": "mktStructureScore",
        "opening_range": "orScore",
        "time": "timeScore",
    }

    active: list[tuple[str, str, float]] = []  # (var_name, expr, weight)
    for category, var_name in CATEGORY_VARS.items():
        w = _category_weight(weights, category)
        if w != 0.0 and category in CATEGORY_EXPRS:
            active.append((var_name, CATEGORY_EXPRS[category], w))

    total = round(sum(w for _, _, w in active), 6)
    if total == 0.0:
        total = 1.0  # prevent division by zero

    lines: list[str] = [
        "// ── Alpha Score ──────────────────────────────────────────────────────────",
        "// Generated from Alpha Scoring Engine",
    ]
    raw_parts: list[str] = []
    for var_name, expr, w in active:
        lines.append(f"{var_name:<20} = ({expr}) * {w}")
        raw_parts.append(var_name)

    raw_sum = " + ".join(raw_parts) if raw_parts else "0.0"
    lines.append(f"rawAlpha            = {raw_sum}")
    lines.append(f"totalWeight         = {total}")
    lines.append("alphaScore          = totalWeight > 0 ? math.min(rawAlpha / totalWeight * 100, 100) : 0.0")

    return "\n".join(lines)


def render_pine_script(
    best_params: dict[str, Any],
    best_features: dict[str, Any],
    weights_data: dict[str, Any],
    translated_signal: TranslatedSignal | None = None,
) -> str:
    """Assemble the complete Pine Script v5 strategy.

    Args:
        best_params: Walk-forward optimisation result (``best_parameters.json``).
        best_features: Context discovery best features (``best_features.json``).
        weights_data: Alpha scoring weights (``weights.json``).
        translated_signal: Optional discovered signal payload for signal-pattern
            export mode.

    Returns:
        A string containing the full, deployable Pine Script v5 strategy.
    """
    if translated_signal is not None:
        return render_signal_pattern_script(translated_signal)

    fast_ema: int = int(best_params.get("fast_ema", 50))
    slow_ema: int = int(best_params.get("slow_ema", 200))
    weights: list[dict[str, Any]] = weights_data.get("weights", [])

    sections: list[str] = [
        T.VERSION_HEADER,
        "",
        T.STRATEGY_DECLARATION,
        "",
        render_inputs(fast_ema, slow_ema),
        "",
        T.WALKFORWARD_COMMENT,
        T.INDICATORS,
        "",
        render_context_filters(best_features),
        "",
        render_alpha_score(weights),
        "",
        T.ENTRY_LOGIC,
        "",
        T.EXIT_LOGIC,
        "",
        T.ORDER_EXECUTION,
        "",
        T.PLOTS,
        "",
        T.ALERTS,
    ]

    return "\n".join(sections) + "\n"


def render_signal_pattern_script(translated_signal: TranslatedSignal) -> str:
    """Render a Pine strategy from a discovered signal pattern."""
    pattern = translated_signal.pattern
    metadata = T.SIGNAL_METADATA_TEMPLATE.format(
        rule_lines="\n// ".join(pattern.conditions),
        signal_frequency=_display_value(pattern.signal_frequency),
        win_rate=_format_metric(pattern.win_rate, percentage=True),
        average_return=_format_metric(pattern.average_return, percentage=True),
        expectancy=_format_metric(pattern.expectancy, percentage=True),
        average_mfe=_format_metric(pattern.average_mfe, percentage=True),
        average_mae=_format_metric(pattern.average_mae, percentage=True),
        diversity_score=_format_metric(pattern.diversity_score),
        composite_score=_format_metric(pattern.composite_score),
    )
    indicators = T.SIGNAL_BASE_INDICATORS.format(
        indicator_lines="\n".join(translated_signal.indicator_lines) or "// (no extra indicators)"
    )
    entry_condition = "\n    and ".join(translated_signal.entry_conditions)
    entry_logic = T.SIGNAL_ENTRY_LOGIC_TEMPLATE.format(
        research_score=_format_metric(pattern.composite_score, default="0.0"),
        entry_condition=entry_condition,
    )

    sections: list[str] = [
        T.VERSION_HEADER,
        "",
        metadata,
        "",
        T.STRATEGY_DECLARATION,
        "",
        T.SIGNAL_STRATEGY_INPUTS,
        "",
        indicators,
        "",
        entry_logic,
        "",
        T.SIGNAL_EXIT_LOGIC,
        "",
        T.SIGNAL_ORDER_EXECUTION,
        "",
        T.SIGNAL_VISUALS,
        "",
        T.SIGNAL_ALERTS,
    ]
    return "\n".join(sections) + "\n"


def _format_metric(value: float | None, *, percentage: bool = False, default: str = "n/a") -> str:
    if value is None:
        return default
    formatted = format(value, ".6f").rstrip("0").rstrip(".")
    return f"{formatted}%" if percentage else formatted


def _display_value(value: int | None) -> str:
    return "n/a" if value is None else str(value)
