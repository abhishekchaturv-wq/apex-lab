# Feature Engineering Platform

## Overview

The **APEX Lab Feature Engineering Platform** provides a production-grade,
extensible system for computing technical indicators from OHLCV (Open, High,
Low, Close, Volume) candlestick data.  The platform is built exclusively on
[Polars](https://pola.rs/) for fully vectorised, high-throughput computation.

```
FeatureRegistry  ←  FeatureGroup subclasses (self-register on import)
       ↓
FeatureEngine.compute(df) → enriched DataFrame
```

---

## Architecture

| Component | Role |
|-----------|------|
| `FeatureGroup` | Abstract base class every feature group must implement |
| `FeatureRegistry` | Duplicate-safe store mapping group names to instances |
| `FeatureEngine` | Orchestrator — contains zero feature-specific logic |

### FeatureGroup

```python
class MyGroup(FeatureGroup):
    @property
    def name(self) -> str:
        return "my_group"

    @property
    def warm_up_periods(self) -> int:
        return 20  # e.g. 20-bar rolling window

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        )
```

### FeatureRegistry

```python
from apex_lab.features.registry import default_registry

default_registry.register(MyGroup())
default_registry.list_groups()   # ['price', 'trend', …, 'my_group']
default_registry.get("price")    # PriceFeatures instance
```

### FeatureEngine

```python
from apex_lab.features import FeatureEngine

engine = FeatureEngine()                         # uses default_registry
df_out = engine.compute(df)                      # all groups
df_out = engine.compute(df, groups=["price"])    # specific groups
warmup  = engine.warm_up_periods()               # max warm-up bars
```

---

## Built-in Feature Groups

### Price

**Required columns:** `open`, `high`, `low`, `close`  
**Warm-up:** 14 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `atr_14` | SMA₁₄(TrueRange) | Absolute volatility per bar | ≥ 0 |
| `atr_pct` | Percentile rank of ATR over 100 bars | ATR relative to recent history | 0–100 |
| `atr_norm` | ATR₁₄ / close × 100 | Volatility as % of price | ≥ 0 |
| `body_pct` | \|close − open\| / (high − low) × 100 | Candle body dominance | 0–100 |
| `upper_wick_pct` | (high − max(o,c)) / (high − low) × 100 | Upper rejection strength | 0–100 |
| `lower_wick_pct` | (min(o,c) − low) / (high − low) × 100 | Lower rejection strength | 0–100 |
| `range` | high − low | Absolute bar range | ≥ 0 |
| `gap_pct` | (open − prev\_close) / prev\_close × 100 | Overnight / pre-bar gap | any |
| `typical_price` | (H + L + C) / 3 | Balanced price level | > 0 |
| `median_price` | (H + L) / 2 | Mid-bar price | > 0 |
| `weighted_price` | (H + L + 2C) / 4 | Close-weighted midpoint | > 0 |

**Dependencies:** none (pure OHLC)

---

### Trend

**Required columns:** `high`, `low`, `close`, `volume`  
**Warm-up:** 200 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `ema_9` | EWM(close, span=9) | Short-term trend | > 0 |
| `ema_20` | EWM(close, span=20) | Medium-term trend | > 0 |
| `ema_50` | EWM(close, span=50) | Intermediate trend | > 0 |
| `ema_200` | EWM(close, span=200) | Long-term trend | > 0 |
| `ema_slope` | (EMA₉ − EMA₉ₜ₋₁) / EMA₉ₜ₋₁ × 100 | Momentum of short EMA (% per bar) | any |
| `vwap` | Σ(typical\_price × vol) / Σ(vol) | Session fair value | > 0 |
| `vwap_dist` | (close − VWAP) / VWAP × 100 | Deviation from fair value (%) | any |

**Dependencies:** typical\_price = (H + L + C) / 3

---

### Momentum

**Required columns:** `close`  
**Warm-up:** 35 bars (MACD slow 26 + signal 9)

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `rsi` | 100 − 100 / (1 + RS), RS = EWMA₁₄(gain) / EWMA₁₄(loss) | Overbought / oversold oscillator | 0–100 |
| `rsi_slope` | RSIₜ − RSIₜ₋₁ | Momentum of RSI | any |
| `macd` | EMA₁₂(close) − EMA₂₆(close) | Trend divergence | any |
| `macd_signal` | EMA₉(MACD) | Signal line | any |
| `macd_hist` | MACD − Signal | Histogram bar | any |
| `roc` | (close − close₁₀) / close₁₀ × 100 | 10-bar rate of change (%) | any |

**Dependencies:** none (pure close)

---

### Volatility

**Required columns:** `high`, `low`, `close`  
**Warm-up:** 20 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `rolling_std` | RollingStd₂₀(log\_return) × 100 | Realised volatility (%) | ≥ 0 |
| `bb_width` | (BB\_upper − BB\_lower) / BB\_mid × 100 | Bollinger band expansion (%) | ≥ 0 |
| `atr_expansion` | ATR₁₄ / ATR₁₄ₜ₋₁₄ | ATR ratio vs 14 bars ago | > 0 |

**Dependencies:** ATR (computed internally, not exposed)

---

### Volume

**Required columns:** `close`, `volume`  
**Warm-up:** 20 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `obv` | Cumsum(sign(Δclose) × volume) | Cumulative buying / selling pressure | any |
| `rel_volume` | volume / avg\_volume₂₀ | Current volume vs recent average | ≥ 0 |
| `avg_volume` | RollingMean₂₀(volume) | Baseline volume level | ≥ 0 |
| `volume_spike` | 1 if volume > 2 × avg\_volume else 0 | Binary unusual volume flag | 0 or 1 |

**Dependencies:** none (close + volume)

---

### Structure

**Required columns:** `high`, `low`  
**Warm-up:** 10 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `swing_high` | RollingMax₁₀(high) | Recent swing high level | ≥ 0 |
| `swing_low` | RollingMin₁₀(low) | Recent swing low level | ≥ 0 |
| `higher_high` | 1 if swing\_high > prev\_swing\_high | Bullish structure signal | 0 or 1 |
| `lower_high` | 1 if swing\_high < prev\_swing\_high | Bearish structure signal | 0 or 1 |
| `higher_low` | 1 if swing\_low > prev\_swing\_low | Bullish structure signal | 0 or 1 |
| `lower_low` | 1 if swing\_low < prev\_swing\_low | Bearish structure signal | 0 or 1 |

**Dependencies:** none (pure HL)

---

### Time

**Required columns:** `timestamp` (`pl.Datetime`)  
**Warm-up:** 0 bars

| Feature | Formula | Interpretation | Expected Range |
|---------|---------|----------------|----------------|
| `hour` | timestamp.dt.hour() | Hour of day | 0–23 |
| `minute` | timestamp.dt.minute() | Minute within the hour | 0–59 |
| `weekday` | timestamp.dt.weekday() | ISO weekday (1=Mon, 7=Sun) | 1–7 |
| `month` | timestamp.dt.month() | Calendar month | 1–12 |
| `quarter` | timestamp.dt.quarter() | Calendar quarter | 1–4 |
| `trading_session` | 0=pre-market, 1=regular, 2=post-market, −1=outside | NSE session label | −1, 0, 1, 2 |

**Dependencies:** datetime column  
**Session boundaries (IST):**  
- Pre-market: 09:00 – 09:14  
- Regular: 09:15 – 15:29  
- Post-market: 15:30 – 15:59  

---

## Adding a Custom Feature Group

1. Subclass `FeatureGroup`.
2. Implement `name`, `warm_up_periods`, and `compute`.
3. Register the instance:

```python
from apex_lab.features.registry import default_registry
from apex_lab.features.base import FeatureGroup
import polars as pl

class MyGroup(FeatureGroup):
    @property
    def name(self) -> str:
        return "my_group"

    @property
    def warm_up_periods(self) -> int:
        return 5

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("close").rolling_mean(window_size=5).alias("sma_5")
        )

default_registry.register(MyGroup())
```

4. The engine will now include `"my_group"` in all `compute()` calls
   (unless `groups=` is specified).

---

## Performance

The platform targets **100,000 candles in under 2 seconds** for
individual, vectorised groups.  End-to-end (all seven built-in groups)
completes well within 10 seconds on commodity hardware.

The single non-vectorised operation is the rolling ATR percentile rank in
`PriceFeatures`, which uses a Python-level fallback for the percentile
calculation.  If sub-second performance is required for very large datasets,
this can be replaced with an approximate rank using Polars `rank` over a
rolling frame.

---

## Conventions

- Polars only — no Pandas  
- All public methods are fully type-hinted  
- Google-style docstrings throughout  
- No `print()` — standard `logging` used in every module  
- No placeholder (`...`) implementations  
- Ruff + Black compliant (line length 100)

---

*Last updated: 2026-07-15*
