"""Pine Script v5 template strings for the Apex Research Engine export module.

Each constant in this module is a self-contained block of Pine Script text.
The renderer combines these blocks into the final strategy file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version header
# ---------------------------------------------------------------------------

VERSION_HEADER = "//@version=5"

# ---------------------------------------------------------------------------
# Strategy declaration
# ---------------------------------------------------------------------------

STRATEGY_DECLARATION = """\
strategy(
    "Apex Research Strategy",
    overlay=true,
    pyramiding=0,
    process_orders_on_close=true,
    initial_capital=25000
)"""

# ---------------------------------------------------------------------------
# Input declarations template
# (placeholders: {ema_fast_default}, {ema_slow_default})
# ---------------------------------------------------------------------------

INPUTS_TEMPLATE = """\
// ── Strategy Inputs ──────────────────────────────────────────────────────
emaFastLen         = input.int({ema_fast_default},   title="EMA Fast",              minval=1,   group="EMA Parameters")
emaSlowLen         = input.int({ema_slow_default},  title="EMA Slow",              minval=1,   group="EMA Parameters")
alphaThreshold     = input.float(50.0, title="Alpha Threshold (0-100)",  minval=0, maxval=100, group="Alpha Score")
atrStopMult        = input.float(2.0,  title="ATR Stop Multiplier",      minval=0.1,           group="Risk Management")
riskPct            = input.float(1.0,  title="Risk %",                   minval=0.01,          group="Risk Management")
enableLong         = input.bool(true,  title="Enable Long",                                    group="Trade Direction")
enableShort        = input.bool(false, title="Enable Short",                                   group="Trade Direction")
useAtrStop         = input.bool(true,  title="Use ATR Stop",                                   group="Exits")
useTrailingStop    = input.bool(false, title="Use Trailing Stop",                               group="Exits")
showAlphaBg        = input.bool(true,  title="Show Alpha Background",                          group="Visuals")"""

# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

INDICATORS = """\
// ── Indicators ───────────────────────────────────────────────────────────
fastEMA   = ta.ema(close, emaFastLen)
slowEMA   = ta.ema(close, emaSlowLen)
ema200    = ta.ema(close, 200)
vwapVal   = ta.vwap(hlc3)
atrVal    = ta.atr(14)
rsiVal    = ta.rsi(close, 14)
[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)"""

# ---------------------------------------------------------------------------
# Walk-forward parameter comment
# ---------------------------------------------------------------------------

WALKFORWARD_COMMENT = "// Generated from Walk Forward Optimization"

# ---------------------------------------------------------------------------
# Context filter template
# (placeholder: {context_filter_lines})
# ---------------------------------------------------------------------------

CONTEXT_FILTERS_TEMPLATE = """\
// ── Context Filters ──────────────────────────────────────────────────────
// Generated from Context Discovery Engine
aboveEMA200   = close > ema200
aboveVWAP     = close > vwapVal
ema200Slope   = ema200 - ema200[5]
ema200Rising  = ema200Slope > 0
{context_filter_lines}"""

# ---------------------------------------------------------------------------
# Alpha score template
# (placeholders: {trend_weight}, {momentum_weight}, {volatility_weight},
#                {vwap_weight}, {market_structure_weight},
#                {opening_range_weight}, {time_weight})
# ---------------------------------------------------------------------------

ALPHA_SCORE_TEMPLATE = """\
// ── Alpha Score ──────────────────────────────────────────────────────────
// Generated from Alpha Scoring Engine
trendScore          = (close > ema200 ? 1.0 : 0.0) * {trend_weight}
momentumScore       = (rsiVal > 50 ? 1.0 : 0.0) * {momentum_weight}
volatilityScore     = (atrVal / close * 100 < 1.5 ? 1.0 : 0.0) * {volatility_weight}
vwapScore           = (close > vwapVal ? 1.0 : 0.0) * {vwap_weight}
mktStructureScore   = (close > ta.highest(high[1], 5) ? 1.0 : 0.0) * {market_structure_weight}
orScore             = (close > ta.highest(high[1], 13) ? 1.0 : 0.0) * {opening_range_weight}
timeScore           = (hour >= 9 and hour <= 14 ? 1.0 : 0.0) * {time_weight}
rawAlpha            = trendScore + momentumScore + volatilityScore + vwapScore + mktStructureScore + orScore + timeScore
totalWeight         = {total_weight}
alphaScore          = totalWeight > 0 ? math.min(rawAlpha / totalWeight * 100, 100) : 0.0"""

# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------

ENTRY_LOGIC = """\
// ── Entry Logic ──────────────────────────────────────────────────────────
longCondition  = ta.crossover(fastEMA, slowEMA)
    and alphaScore >= alphaThreshold
    and aboveEMA200
    and aboveVWAP
    and enableLong

shortCondition = ta.crossunder(fastEMA, slowEMA)
    and alphaScore >= alphaThreshold
    and not aboveEMA200
    and not aboveVWAP
    and enableShort"""

# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------

EXIT_LOGIC = """\
// ── Exit Logic ───────────────────────────────────────────────────────────
longExitCondition  = ta.crossunder(fastEMA, slowEMA)
shortExitCondition = ta.crossover(fastEMA, slowEMA)"""

# ---------------------------------------------------------------------------
# Position sizing and order execution
# ---------------------------------------------------------------------------

ORDER_EXECUTION = """\
// ── Position Sizing & Order Execution ────────────────────────────────────
stopDistance  = useAtrStop ? atrStopMult * atrVal : close * 0.02
trailOffset   = useTrailingStop ? atrStopMult * atrVal : na

if longCondition
    strategy.entry("Long", strategy.long)
    if useAtrStop
        strategy.exit("Long Exit", "Long", stop=close - stopDistance, trail_offset=useTrailingStop ? trailOffset : na)

if shortCondition
    strategy.entry("Short", strategy.short)
    if useAtrStop
        strategy.exit("Short Exit", "Short", stop=close + stopDistance, trail_offset=useTrailingStop ? trailOffset : na)

if longExitCondition
    strategy.close("Long")

if shortExitCondition
    strategy.close("Short")"""

# ---------------------------------------------------------------------------
# Plot declarations
# ---------------------------------------------------------------------------

PLOTS = """\
// ── Plots ────────────────────────────────────────────────────────────────
plot(fastEMA,  title="EMA Fast",  color=color.new(color.blue,   0), linewidth=1)
plot(slowEMA,  title="EMA Slow",  color=color.new(color.orange, 0), linewidth=2)
plot(ema200,   title="EMA 200",   color=color.new(color.gray,   0), linewidth=1)
plot(vwapVal,  title="VWAP",      color=color.new(color.purple, 0), linewidth=1)

// Alpha Score background (requires a separate pane in TradingView)
alphaBgColor = alphaScore >= 70 ? color.new(color.green,  85)
             : alphaScore >= 40 ? color.new(color.yellow, 85)
             :                    color.new(color.red,    85)
bgcolor(showAlphaBg ? alphaBgColor : na, title="Alpha Background")"""

# ---------------------------------------------------------------------------
# Alert conditions
# ---------------------------------------------------------------------------

ALERTS = """\
// ── Alerts ───────────────────────────────────────────────────────────────
alertcondition(longCondition,              title="Long Entry",            message="Apex: Long Entry signal triggered")
alertcondition(longExitCondition,          title="Long Exit",             message="Apex: Long Exit signal triggered")
alertcondition(alphaScore >= alphaThreshold, title="Alpha Above Threshold", message="Apex: Alpha score above threshold")
alertcondition(alphaScore < alphaThreshold,  title="Alpha Below Threshold", message="Apex: Alpha score below threshold")"""

# ---------------------------------------------------------------------------
# Signal-pattern export templates
# ---------------------------------------------------------------------------

SIGNAL_METADATA_TEMPLATE = """\
// Generated by Apex Lab
//
// Rule:
// {rule_lines}
//
// Signal Frequency: {signal_frequency}
// Win Rate: {win_rate}
// Average Return: {average_return}
// Expectancy: {expectancy}
// Average MFE: {average_mfe}
// Average MAE: {average_mae}
// Diversity Score: {diversity_score}
// Composite Score: {composite_score}"""

SIGNAL_STRATEGY_INPUTS = """\
// ── Strategy Inputs ──────────────────────────────────────────────────────
enableLong         = input.bool(true,  title="Enable Long",                 group="Trade Direction")
enableShort        = input.bool(false, title="Enable Short",                group="Trade Direction")
stopLossPct        = input.float(1.0,  title="Stop Loss %",                 minval=0.1, group="Risk Management")
takeProfitPct      = input.float(2.0,  title="Take Profit %",               minval=0.1, group="Risk Management")
useAtrStop         = input.bool(true,  title="ATR Stop",                    group="Risk Management")
atrMultiplier      = input.float(2.0,  title="ATR Multiplier",              minval=0.1, group="Risk Management")
riskReward         = input.float(2.0,  title="Risk Reward",                 minval=0.1, group="Risk Management")
useTrailingStop    = input.bool(false, title="Trailing Stop",               group="Risk Management")
maxHoldingBars     = input.int(20,     title="Maximum Holding Bars",        minval=1,   group="Risk Management")
showEntryArrows    = input.bool(true,  title="Entry Arrows",                group="Visuals")
showExitArrows     = input.bool(true,  title="Exit Arrows",                 group="Visuals")
showLabels         = input.bool(true,  title="Labels",                      group="Visuals")
showBackground     = input.bool(true,  title="Background Colouring",        group="Visuals")
showSignalScore    = input.bool(true,  title="Signal Score Plot",           group="Visuals")"""

SIGNAL_BASE_INDICATORS = """\
// ── Indicators ───────────────────────────────────────────────────────────
atrVal            = ta.atr(14)
vwapVal           = ta.vwap(hlc3)
rsiVal            = ta.rsi(close, 14)
[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)
{indicator_lines}"""

SIGNAL_ENTRY_LOGIC_TEMPLATE = """\
// ── Entry Logic ──────────────────────────────────────────────────────────
researchScore      = {research_score}
signalCondition    = {entry_condition}

longCondition      = enableLong and signalCondition and strategy.position_size <= 0
shortCondition     = enableShort and signalCondition and strategy.position_size >= 0"""

SIGNAL_EXIT_LOGIC = """\
// ── Exit Logic ───────────────────────────────────────────────────────────
var int longEntryBar = na
var int shortEntryBar = na

if longCondition
    longEntryBar := bar_index

if shortCondition
    shortEntryBar := bar_index

maxHoldLong       = strategy.position_size > 0 and not na(longEntryBar) and (bar_index - longEntryBar) >= maxHoldingBars
maxHoldShort      = strategy.position_size < 0 and not na(shortEntryBar) and (bar_index - shortEntryBar) >= maxHoldingBars"""

SIGNAL_ORDER_EXECUTION = """\
// ── Order Execution ──────────────────────────────────────────────────────
longRiskDistance  = useAtrStop ? atrVal * atrMultiplier : strategy.position_avg_price * stopLossPct / 100.0
shortRiskDistance = useAtrStop ? atrVal * atrMultiplier : strategy.position_avg_price * stopLossPct / 100.0
longTakeProfit    = useAtrStop ? strategy.position_avg_price + (longRiskDistance * riskReward) : strategy.position_avg_price * (1 + takeProfitPct / 100.0)
shortTakeProfit   = useAtrStop ? strategy.position_avg_price - (shortRiskDistance * riskReward) : strategy.position_avg_price * (1 - takeProfitPct / 100.0)
trailOffset       = useTrailingStop ? atrVal * atrMultiplier : na

if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if strategy.position_size > 0
    strategy.exit(
        "Long Risk",
        "Long",
        stop=strategy.position_avg_price - longRiskDistance,
        limit=longTakeProfit,
        trail_offset=useTrailingStop ? trailOffset : na
    )

if strategy.position_size < 0
    strategy.exit(
        "Short Risk",
        "Short",
        stop=strategy.position_avg_price + shortRiskDistance,
        limit=shortTakeProfit,
        trail_offset=useTrailingStop ? trailOffset : na
    )

if maxHoldLong
    strategy.close("Long", comment="Max Holding Bars")

if maxHoldShort
    strategy.close("Short", comment="Max Holding Bars")"""

SIGNAL_VISUALS = """\
// ── Visual Components ────────────────────────────────────────────────────
signalScore       = signalCondition ? researchScore : 0.0
bgcolor(showBackground and signalCondition ? color.new(color.green, 88) : na, title="Signal Background")
plot(showSignalScore ? signalScore : na, title="Signal Score", color=color.new(color.teal, 0), linewidth=2)
plotshape(showEntryArrows and longCondition,  title="Long Entry",  style=shape.triangleup,   location=location.belowbar, color=color.new(color.green, 0), size=size.tiny, text="L")
plotshape(showEntryArrows and shortCondition, title="Short Entry", style=shape.triangledown, location=location.abovebar, color=color.new(color.red, 0),   size=size.tiny, text="S")
plotshape(showExitArrows and maxHoldLong,     title="Long Exit",   style=shape.xcross,       location=location.abovebar, color=color.new(color.orange, 0), size=size.tiny, text="XL")
plotshape(showExitArrows and maxHoldShort,    title="Short Exit",  style=shape.xcross,       location=location.belowbar, color=color.new(color.orange, 0), size=size.tiny, text="XS")

if showLabels and longCondition
    label.new(bar_index, low,  text="Rule\\nScore: " + str.tostring(researchScore, "#.######"), style=label.style_label_up,   color=color.new(color.green, 0), textcolor=color.white)

if showLabels and shortCondition
    label.new(bar_index, high, text="Rule\\nScore: " + str.tostring(researchScore, "#.######"), style=label.style_label_down, color=color.new(color.red, 0),   textcolor=color.white)"""

SIGNAL_ALERTS = """\
// ── Alerts ───────────────────────────────────────────────────────────────
alertcondition(signalCondition, title="Signal Match", message="Apex: discovered signal rule matched")
alertcondition(longCondition,   title="Long Entry",   message="Apex: long signal entry triggered")
alertcondition(shortCondition,  title="Short Entry",  message="Apex: short signal entry triggered")
alertcondition(maxHoldLong,     title="Long Exit",    message="Apex: long exit triggered")
alertcondition(maxHoldShort,    title="Short Exit",   message="Apex: short exit triggered")"""

# ---------------------------------------------------------------------------
# Section separator comment template
# (placeholder: {section_title})
# ---------------------------------------------------------------------------

SECTION_COMMENT_TEMPLATE = "\n// {section_title}\n"
