# APEX Lab Architecture

## Data Pipeline

The APEX Lab system follows a linear data pipeline designed for quantitative research and automated trading:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

    Downloader
        │
        ├─ Fetches OHLC data from Zerodha Kite Connect
        ├─ Handles NIFTY, BANKNIFTY, F&O stocks
        └─ Stores raw data in Polars DataFrames
        │
        ▼
    Feature Engine
        │
        ├─ Computes technical indicators
        ├─ Generates statistical features
        └─ Normalizes and scales features
        │
        ▼
    Label Engine
        │
        ├─ Identifies reversal patterns
        ├─ Generates classification labels
        └─ Handles class imbalance
        │
        ▼
    Model Training
        │
        ├─ Trains ML models (sklearn, XGBoost, LightGBM)
        ├─ Hyperparameter optimization
        └─ Cross-validation and model selection
        │
        ▼
    Backtesting
        │
        ├─ Simulates historical trading
        ├─ Calculates performance metrics
        └─ Risk analysis and drawdown studies
        │
        ▼
    Signal Dataset
        │
        ├─ Builds candle-level supervised datasets
        ├─ Persists feature lists and quantile metadata
        └─ Produces reusable research artifacts
        │
        ▼
    Signal Discovery
        │
        ├─ Ranks predictive features
        ├─ Evaluates feature combinations
        └─ Identifies exportable market signals
        │
        ▼
    Signal Patterns
        │
        ├─ Generates top-ranked rule combinations
        ├─ Validates robustness and diversity
        └─ Writes top_signals.json for export
        │
        ▼
    Pine Export
        │
        ├─ Loads discovered rules and quantile metadata
        ├─ Translates signal conditions into Pine expressions
        └─ Generates TradingView Pine Script strategies
           for deployable signal execution
```

## Module Overview

- **downloader**: Zerodha Kite Connect integration for market data retrieval
- **features**: Technical indicator computation and feature engineering
- **labels**: Target variable generation for supervised learning
- **models**: Machine learning model implementations and training
- **backtest**: Historical strategy simulation and performance evaluation
- **research.signal_dataset**: Candle-level supervised dataset builder and quantile artifacts
- **research.signal_discovery**: Feature ranking and predictive signal analysis
- **research.signal_patterns**: Robust signal-pattern discovery and ranking
- **export**: Pine strategy generation from legacy research outputs and discovered rules
- **visualization**: Charting and analysis visualization utilities
- **utils**: Common utilities and helper functions
