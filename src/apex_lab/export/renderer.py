"""Rendering logic for the Pine Script generator.

This module is responsible for assembling the Pine Script template blocks into
a complete, ready-to-deploy strategy file.  It does **not** perform any file
I/O; that responsibility belongs to :mod:`apex_lab.export.pine_generator`.
"""

from __future__ import annotations

from typing import Any

from apex_lab.export import template as T


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
) -> str:
    """Assemble the complete Pine Script v5 strategy.

    Args:
        best_params: Walk-forward optimisation result (``best_parameters.json``).
        best_features: Context discovery best features (``best_features.json``).
        weights_data: Alpha scoring weights (``weights.json``).

    Returns:
        A string containing the full, deployable Pine Script v5 strategy.
    """
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
