# APEX Lab Architecture

## Data Pipeline

The APEX Lab system follows a linear data pipeline designed for quantitative research and automated trading:

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        QUANTITATIVE TRADING PIPELINE                                 ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    Strategy & Rules
        │
        ├─ Entry/Exit Logic
        ├─ Risk Management
        └─ Position Sizing
        │
        ▼
    Feature Engine
        │
        ├─ Technical Indicators (RSI, MACD, Bollinger Bands)
        ├─ Statistical Features (Returns, Volatility)
        ├─ Pattern Recognition
        └─ Aggregated Features
        │
        ▼
    Label Engine
        │
        ├─ Identify Reversal Patterns
        ├─ Generate Classification Labels
        ├─ Handle Class Imbalance
        └─ Create Training Targets
        │
        ▼
    Model Training
        │
        ├─ Train Classification Models
        ├─ Hyperparameter Optimization
        ├─ Cross-Validation
        └─ Model Selection (XGBoost, LightGBM, etc.)
        │
        ▼
    Backtesting Engine
        │
        ├─ Historical Simulation
        ├─ Performance Metrics
        ├─ Risk Analysis
        ├─ Drawdown Studies
        └─ Walk-Forward Testing
        │
        ▼
    Pine Script Export
        │
        └─ TradingView Indicator
            └─ Real-time Reversal Signals


╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        DATA LAYER (Market Data Provider)                             ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    MarketDataProvider (Abstract Interface)
        ↓
        ├─ KiteProvider (Zerodha Kite Connect)
        │   └─ Live/Historical data from Zerodha
        │
        ├─ CSVProvider
        │   └─ Load data from CSV files
        │
        ├─ ParquetProvider
        │   └─ Load data from Parquet files
        │
        ├─ NSEProvider
        │   └─ Load data from NSE India
        │
        └─ Future Providers
            ├─ PolygonProvider (Polygon.io)
            ├─ AlpacaProvider (Alpaca Markets)
            ├─ LiveFeedProvider (Real-time feeds)
            └─ Other data sources


╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        KITE PROVIDER IMPLEMENTATION                                  ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    KiteClient (Main Entry Point)
        ↓
        ├─ Authentication (KiteAuthenticator)
        │   ├─ Credential validation
        │   ├─ Session management
        │   └─ Token refresh
        │
        ├─ Retry Logic (Exponential Backoff)
        │   ├─ Transient error retry
        │   ├─ Non-transient error handling
        │   └─ Rate limit handling
        │
        └─ API Methods
            ├─ get_history() - Historical OHLC data
            ├─ validate_symbol() - Symbol validation
            └─ health_check() - Provider health
                ↓
            Zerodha Kite Connect API
                ↓
            Zerodha Servers


╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        ERROR HANDLING & RESILIENCE                                   ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    Exception Hierarchy
        │
        └─ ApexError (Base)
            ├─ ConfigurationError
            ├─ AuthenticationError (non-retriable)
            ├─ DownloadError
            ├─ ValidationError (non-retriable)
            └─ RateLimitError (retriable)

    Retry Strategy
        ├─ Transient Errors → Retry with exponential backoff
        │   ├─ ConnectionError
        │   ├─ TimeoutError
        │   └─ RateLimitError
        │
        └─ Non-Transient Errors → Fail immediately
            ├─ AuthenticationError
            ├─ ValidationError
            └─ ConfigurationError
```

## Module Overview

- **data/provider.py**: Abstract `MarketDataProvider` interface
- **data/kite/client.py**: `KiteClient` implementation (main entry point)
- **data/kite/auth.py**: `KiteAuthenticator` for session management
- **data/kite/retry.py**: Retry decorator with exponential backoff
- **downloader**: Will be deprecated in favor of the data layer
- **features**: Technical indicator computation and feature engineering
- **labels**: Target variable generation for supervised learning
- **models**: Machine learning model implementations and training
- **backtest**: Historical strategy simulation and performance evaluation
- **visualization**: Charting and analysis visualization utilities
- **utils**: Common utilities and helper functions
- **config**: Configuration management and logging

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Abstraction**: MarketDataProvider allows swapping data sources
3. **Type Safety**: Full type hints throughout
4. **Error Handling**: Explicit exception hierarchy with meaningful messages
5. **Retry Logic**: Automatic retry for transient failures
6. **No Hardcoding**: All configuration via settings
7. **Logging**: Structured logging instead of print statements
8. **Testing**: Comprehensive unit tests with mocks (no external API calls)
