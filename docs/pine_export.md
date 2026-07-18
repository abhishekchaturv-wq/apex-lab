# Pine Export

## Overview

The Pine export subsystem extends the existing TradingView strategy generator.
It supports two compatible inputs:

1. Legacy walk-forward/context/alpha research outputs
2. Discovered signal patterns plus persisted quantile metadata

## Signal Pattern Export Workflow

```text
Raw OHLCV
    ↓
signal_dataset
    ↓
signal_discovery
    ↓
signal_patterns
    ↓
top_signals.json
    ↓
Pine Export
    ↓
TradingView Strategy (.pine)
```

## Required Artifacts

Signal-pattern driven Pine export consumes:

- `reports/lab/signal_patterns/top_signals.json`
- `reports/lab/signal_dataset/quantiles.json`

The dataset builder now persists `quantiles.json` alongside:

- `dataset.parquet`
- `schema.json`
- `summary.json`
- `feature_list.json`

## Translation Architecture

```text
SignalPatternLoader
        ↓
RuleTranslator
        ↓
FeatureTranslatorRegistry
        ↓
Renderer
        ↓
Template
```

Responsibilities remain separated:

- loader: reads `top_signals.json` / CSV records
- translator: validates and converts discovered rules into Pine conditions
- registry: resolves feature-specific translators
- renderer: assembles the final Pine strategy
- serializer: writes deterministic summary metadata

## CLI Usage

Legacy export:

```bash
python scripts/research_lab.py --mode pine
```

Signal-pattern export:

```bash
python scripts/research_lab.py \
    --mode pine \
    --signal-patterns reports/lab/signal_patterns/top_signals.json \
    --quantiles reports/lab/signal_dataset/quantiles.json \
    --output generated
```

Optional Pine CLI arguments:

- `--signal-patterns`
- `--quantiles`
- `--output`

## Generated Strategy Features

The generated Pine strategy keeps strategy-level behavior and adds configurable
risk-management controls around discovered entries:

- `strategy()`
- long entries with optional short support
- stop loss %
- take profit %
- ATR stop
- ATR multiplier
- risk reward
- trailing stop
- maximum holding bars
- alerts
- labels
- entry/exit arrows
- background colouring
- signal score plot

## Validation Rules

Before rendering Pine, the export pipeline validates:

- discovered rule syntax
- supported feature translators
- quantile metadata availability
- duplicate condition removal
- deterministic ordering of translated conditions and indicator lines
