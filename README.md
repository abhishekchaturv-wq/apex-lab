# Apex Lab

A laboratory for experimentation and development.

## Getting Started

This repository is set up for collaborative development and exploration.

## Historical Data CLI

The historical data engine is available from the command line via `scripts/data.py`.

```bash
python scripts/data.py refresh-instruments

python scripts/data.py download \
    --symbol BANKNIFTY \
    --interval 30minute \
    --from 2016-01-01

python scripts/data.py update \
    --symbol BANKNIFTY \
    --interval 30minute
```

## Research Lab

The research lab is the entry point for all quantitative research modes.

```bash
# Forward-return analysis (default)
python scripts/research_lab.py

# Event-driven backtest
python scripts/research_lab.py --mode event

# Walk-forward optimisation
python scripts/research_lab.py --mode optimize

# Factor combination research
python scripts/research_lab.py --mode factors

# Portfolio simulation
python scripts/research_lab.py --mode portfolio

# Context / alpha discovery
python scripts/research_lab.py --mode context

# Alpha scoring
python scripts/research_lab.py --mode alpha

# Signal discovery dataset builder
python scripts/research_lab.py --mode signal_dataset \
    --data ~/kite-test/apex-data-lake/raw/30minute/NIFTY\ BANK.parquet

# Pine Script generator
python scripts/research_lab.py --mode pine
```

### Strategy Research Framework

Evaluate all built-in strategies and produce a ranked leaderboard:

```bash
python scripts/research_lab.py --mode strategies
```

Evaluate a single strategy by name:

```bash
python scripts/research_lab.py --mode strategies --strategy "EMA Crossover"
```

Specify a custom output directory:

```bash
python scripts/research_lab.py --mode strategies --strategies-output-dir reports/lab/strategy
```

#### Registered strategies

| Strategy | Entry condition | Exit condition |
|---|---|---|
| EMA Crossover | EMA20 crosses above EMA50 | EMA20 crosses below EMA50 |
| Opening Range Breakout | Close crosses above session ORB high | Close crosses below session ORB low |
| VWAP Trend | Close crosses above session VWAP | Close crosses below session VWAP |
| EMA + VWAP | EMA crossover AND close > VWAP | EMA bearish crossover |
| EMA + RSI | EMA crossover AND RSI > 50 | EMA bearish crossover |
| EMA + ATR Expansion | EMA crossover AND ATR percentile in [20, 80] | EMA bearish crossover |

#### Output files

Reports are written to `reports/lab/strategy/` (previous runs are never overwritten):

- `leaderboard.csv` — strategies ranked by composite score
- `metrics.csv` — full scorecard for every strategy
- `summary.json` — high-level summary (top strategy, counts)
- `top_strategy.json` — detailed metrics for the best-ranked strategy

#### Composite score

Strategies are ranked by a weighted composite score computed from normalised metrics:

| Metric | Weight | Direction |
|---|---|---|
| CAGR | 25% | Higher is better |
| Sharpe Ratio | 25% | Higher is better |
| Profit Factor | 20% | Higher is better |
| Maximum Drawdown | 15% | Lower is better |
| Expectancy | 15% | Higher is better |

## Project Structure

- `/docs` - Documentation
- `/src` - Source code
- `/tests` - Test files

## Contributing

Contributions are welcome! Please ensure code follows the project conventions.
