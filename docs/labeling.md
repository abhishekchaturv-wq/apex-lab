# Label Engine

## Overview

The Label Engine produces deterministic supervised-learning targets from OHLC(+ATR) candles.

## Label types

- `BOTTOM`
- `TOP`
- `NONE`

## Objective definition

Given ATR-scaled thresholds and a lookahead window:

- `BOTTOM`: upside reward is hit before downside failure.
- `TOP`: downside reward is hit before upside failure (mirror rule).
- `NONE`: no valid directional objective is achieved first.

Default rule set:

- `reward_multiplier = 2.0`
- `risk_multiplier = 1.0`
- `lookahead_window = 12`
- `atr_multiplier = 1.0`

## Output columns

Every input candle receives:

- `label`
- `confidence`
- `future_return`
- `bars_to_target`
- `bars_to_failure`

## Configuration

Use `LabelingRules` to parameterize ATR source and thresholds:

- `atr_column`
- `atr_multiplier`
- `reward_multiplier`
- `risk_multiplier`
- `lookahead_window`

## Evaluation

Use `evaluate_labels` to compute:

- total rows
- total directional labels
- positive % (`BOTTOM`)
- negative % (`TOP`)
- class balance
- average absolute move
- median absolute move
