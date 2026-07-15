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
    Pine Export
        │
        └─ Generates TradingView Pine Script indicators
           for real-time reversal detection
```

## Module Overview

- **downloader**: Zerodha Kite Connect integration for market data retrieval
- **features**: Technical indicator computation and feature engineering
- **labels**: Target variable generation for supervised learning
- **models**: Machine learning model implementations and training
- **backtest**: Historical strategy simulation and performance evaluation
- **visualization**: Charting and analysis visualization utilities
- **utils**: Common utilities and helper functions
