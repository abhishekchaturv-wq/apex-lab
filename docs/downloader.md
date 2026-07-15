# Historical Data Downloader

## Overview

The `apex_lab.data` module is the single source of truth for all historical
market data in APEX.  It provides a production-grade pipeline that downloads,
validates, stores and incrementally updates OHLCV candle data from Zerodha
Kite Connect — with no manual scripts required.

---

## Architecture

```
apex_lab/data/
├── __init__.py      Public API (download_symbol, update_symbol, …)
├── downloader.py    KiteDownloader + DataEngine facade
├── instruments.py   InstrumentManager  (symbol → token lookup)
├── storage.py       Parquet I/O + path helpers
├── validator.py     OHLCV sanity checks
├── updater.py       Incremental update logic
└── metadata.py      SymbolMetadata dataclass + JSON I/O
```

### Component relationships

```
Public functions (data/__init__.py)
        │
        ▼
   DataEngine  (downloader.py)
   ├── InstrumentManager  (instruments.py)
   │       └── instruments.parquet  (data/reference/)
   ├── KiteDownloader  (downloader.py)
   │       ├── split_into_chunks()
   │       ├── _fetch_with_retry()
   │       └── chunk cache  (data/cache/)
   └── update_symbol()  (updater.py)
```

All Parquet I/O is routed through `storage.py`.
All validation is routed through `validator.py`.
All metadata is routed through `metadata.py`.

---

## Directory Layout

```
data/
├── raw/
│   ├── minute/
│   │   └── BANKNIFTY.parquet
│   ├── 30minute/
│   │   ├── BANKNIFTY.parquet
│   │   └── BANKNIFTY.metadata.json
│   └── day/
│       └── RELIANCE.parquet
├── reference/
│   └── instruments.parquet
└── cache/
    └── BANKNIFTY_30minute/   ← temporary chunks (deleted after download)
        ├── chunk_0000.parquet
        └── chunk_0001.parquet
```

All paths are derived from `settings.data_dir`.  Nothing is hardcoded.

---

## Parquet Schema

Every raw symbol file has the following columns:

| Column      | Type      | Required | Notes                    |
|-------------|-----------|----------|--------------------------|
| `timestamp` | Datetime  | ✓        | Candle open time (IST)   |
| `open`      | Float64   | ✓        |                          |
| `high`      | Float64   | ✓        |                          |
| `low`       | Float64   | ✓        |                          |
| `close`     | Float64   | ✓        |                          |
| `volume`    | Int64     | ✓        |                          |
| `oi`        | Int64     | optional | Open interest (futures)  |

---

## Supported Intervals

| Interval   | Max days per API request |
|------------|--------------------------|
| `minute`   | 60                       |
| `3minute`  | 100                      |
| `5minute`  | 100                      |
| `10minute` | 100                      |
| `15minute` | 100                      |
| `30minute` | 100                      |
| `60minute` | 400                      |
| `day`      | 2 000                    |

---

## Download Pipeline

```
download_symbol("BANKNIFTY", "30minute", "2016-01-01", "today")
        │
        ├─ 1. Resolve instrument token via InstrumentManager
        │
        ├─ 2. split_into_chunks(start, end, interval)
        │      → [(2016-01-01, 2016-04-10), (2016-04-11, …), …]
        │
        ├─ 3. For each chunk:
        │      a. Check local cache (resume support)
        │      b. If missing → kite.historical_data() with retry/backoff
        │      c. Save chunk to data/cache/BANKNIFTY_30minute/chunk_NNNN.parquet
        │
        ├─ 4. merge_and_dedup(all_chunks)
        │      → deduplicated, timestamp-sorted DataFrame
        │
        ├─ 5. assert_valid_ohlcv(df)
        │
        ├─ 6. write_parquet(df, data/raw/30minute/BANKNIFTY.parquet)
        │
        ├─ 7. save_metadata(…, data/raw/30minute/BANKNIFTY.metadata.json)
        │
        └─ 8. Clean up chunk cache files
```

---

## Update Workflow

`update_symbol` appends only the missing candles:

```
update_symbol("BANKNIFTY", "30minute")
        │
        ├─ 1. Read existing data/raw/30minute/BANKNIFTY.parquet
        │
        ├─ 2. Find latest timestamp → new_start = latest_date + 1 day
        │
        ├─ 3. If new_start > today → already up to date, return early
        │
        ├─ 4. download_symbol("BANKNIFTY", "30minute", new_start, "today")
        │
        ├─ 5. merge_and_dedup(existing + new)
        │
        ├─ 6. Validate, overwrite Parquet, update metadata
        │
        └─ 7. Return updated DataFrame
```

---

## Resume Workflow

If a download is interrupted mid-way:

1. Completed chunks remain in `data/cache/SYMBOL_INTERVAL/`.
2. Re-running `download_symbol(…)` checks for existing chunk files.
3. Present chunks are loaded from disk; missing chunks are re-fetched.
4. No already-downloaded data is wasted.

---

## Incremental Workflow

```python
# Full initial download
from apex_lab.data import download_symbol, update_symbol

download_symbol("BANKNIFTY", "30minute", "2016-01-01", "today")

# Next day — fetch only yesterday's missing candles
update_symbol("BANKNIFTY", "30minute")
```

---

## Instrument Management

Instruments must be refreshed before the first download:

```python
from apex_lab.data import refresh_instruments, download_symbol

refresh_instruments()                           # writes data/reference/instruments.parquet
download_symbol("BANKNIFTY", "day", "2020-01-01", "today")
```

The `InstrumentManager` caches the DataFrame in memory; subsequent calls
within the same process do not hit disk again.  Call
`engine.instruments.invalidate_cache()` if you need to force a reload.

---

## Retry / Backoff

Every Kite API call is wrapped in `_fetch_with_retry`, which:

* Retries `kiteconnect.exceptions.NetworkException` and `DataException`.
* Uses exponential backoff: `delay = base_delay × 2^attempt`.
* Default: 3 retries, 1 s base delay.

Non-retryable exceptions (`TokenException`, `InputException`, etc.) are
propagated immediately.

---

## Validation

`assert_valid_ohlcv(df)` checks:

| Rule                           | Description                          |
|--------------------------------|--------------------------------------|
| Required columns present       | `timestamp`, `open`, `high`, `low`, `close`, `volume` |
| No duplicate timestamps        | Each candle's timestamp is unique    |
| Timestamps sorted ascending    | No out-of-order candles              |
| `high >= open`                 | OHLC sanity                          |
| `high >= close`                | OHLC sanity                          |
| `low  <= open`                 | OHLC sanity                          |
| `low  <= close`                | OHLC sanity                          |
| `volume >= 0`                  | Non-negative volume                  |

Validation failures raise `ValueError` with a pipe-separated list of all
failing rules.

---

## Configuration

All paths and credentials are read from project settings (`.env` / env vars).
No values are hardcoded.

| Setting               | Description                        | Default       |
|-----------------------|------------------------------------|---------------|
| `KITE_API_KEY`        | Zerodha API key                    | *(required)*  |
| `KITE_API_SECRET`     | Zerodha API secret                 | *(required)*  |
| `KITE_ACCESS_TOKEN`   | Session access token               | `""`          |
| `DATA_DIR`            | Root directory for all data        | `./data`      |

---

## Usage Examples

```python
from apex_lab.data import (
    download_symbol,
    download_universe,
    update_symbol,
    update_universe,
    refresh_instruments,
)

# Refresh instrument master (run once per trading day)
refresh_instruments()

# Download complete history
download_symbol("BANKNIFTY", "30minute", "2016-01-01", "today")

# Download multiple symbols
download_universe(
    ["BANKNIFTY", "NIFTY 50", "RELIANCE"],
    "day",
    "2020-01-01",
    "today",
)

# Incremental update (run daily)
update_symbol("BANKNIFTY", "30minute")

# Update multiple symbols
update_universe(["BANKNIFTY", "RELIANCE"], "30minute")
```

For programmatic use, inject a `KiteConnect` instance directly to avoid
reading credentials from the environment (useful in testing):

```python
from kiteconnect import KiteConnect
from apex_lab.data import download_symbol

kite = KiteConnect(api_key="...")
kite.set_access_token("...")

df = download_symbol("BANKNIFTY", "30minute", "2024-01-01", "today", kite=kite)
```
