# APEX Lab Development Conventions

## Branch Strategy

- **main**: Production-ready code. Protected branch.
- **develop**: Integration branch for features (optional, use if needed)
- **feature/***:  New features. Branch from `main`. Example: `feature/kite-downloader`
- **bugfix/***:  Bug fixes. Example: `bugfix/data-validation-error`
- **chore/***:  Maintenance and non-functional changes. Example: `chore/update-dependencies`

## Commit Message Convention

Follow **Conventional Commits** (https://www.conventionalcommits.org/).

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `refactor`: Code refactoring without feature changes
- `perf`: Performance improvements
- `test`: Test additions or updates
- `docs`: Documentation changes
- `chore`: Build, dependencies, tooling
- `ci`: CI/CD configuration

### Examples

```
feat(downloader): add kite historical data fetch
fix(labels): correct reversal threshold calculation
test(models): add unit tests for xgboost pipeline
docs(architecture): update data pipeline diagram
chore(deps): upgrade scikit-learn to 1.4.0
```

## Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff with pyupgrade enabled
- **Pre-commit**: Highly recommended

Run before commit:
```bash
black src/ tests/
ruff check --fix src/ tests/
```

## Type Hints (Required)

All functions and methods must have **complete type hints**.

```python
def fetch_ohlc(
    symbol: str,
    interval: str,
    days: int
) -> pl.DataFrame:
    """Fetch OHLC data from Zerodha."""
    ...

def compute_rsi(data: pl.DataFrame, period: int = 14) -> pl.Series:
    """Compute RSI indicator."""
    ...
```

Use `from __future__ import annotations` for forward references.

## Docstring Format

Use **Google-style docstrings**.

```python
def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2
) -> tuple[XGBClassifier, dict]:
    """Train XGBoost reversal detection model.

    Args:
        features: Feature matrix of shape (n_samples, n_features)
        labels: Binary labels (0 or 1) for reversal detection
        test_size: Fraction of data to use for testing

    Returns:
        Trained model and evaluation metrics dictionary

    Raises:
        ValueError: If features and labels have mismatched lengths

    Example:
        >>> model, metrics = train_model(X, y)
        >>> print(metrics['accuracy'])
    """
    ...
```

## Testing Expectations

Every feature must have corresponding tests.

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test module interactions
- **Fixtures**: Use pytest fixtures for setup/teardown
- **Coverage**: Aim for >80% code coverage
- **Test naming**: `test_<function>_<scenario>` (e.g., `test_fetch_ohlc_valid_symbol`)

Run tests:
```bash
pytest tests/ -v --cov=src/apex_lab
```

## Naming Conventions

### Variables and Functions
- `snake_case` for variables and functions
- Descriptive names (avoid single letters except loop counters)
  - ✅ `daily_returns`
  - ❌ `dr`
  - ✅ `compute_sharpe_ratio`
  - ❌ `csr`

### Constants
- `UPPER_SNAKE_CASE` for module-level constants
  ```python
  DEFAULT_INTERVAL = "5m"
  MAX_RETRIES = 3
  CACHE_TTL_HOURS = 24
  ```

### Classes
- `PascalCase` for class names
  ```python
  class KiteDownloader:
      pass

  class FeatureEngine:
      pass
  ```

### Private members
- Prefix with single underscore `_`
  ```python
  def _validate_symbol(symbol: str) -> bool:
      """Internal validation."""
      ...
  ```

## Data Storage Layout

All data paths must be configurable via settings. **Never hardcode paths.**

```
data/
├── raw/
│   ├── nifty/
│   ├── banknifty/
│   └── stocks/
├── processed/
│   ├── features/
│   └── labels/
├── models/
│   ├── xgboost/
│   ├── lightgbm/
│   └── catboost/
├── cache/
│   └── ohlc_cache/
└── reports/
    ├── backtest/
    └── analysis/
```

### Access Pattern

```python
from apex_lab.config import settings

raw_data_path = settings.data_dir / "raw" / "nifty"
processed_data_path = settings.data_dir / "processed" / "features"
model_path = settings.data_dir / "models" / "xgboost"
```

## Environment Variables

Never commit `.env` files. Use `.env.example` as a template.

```bash
# .env.example (commit this)
KITE_API_KEY=your_key_here
KITE_API_SECRET=your_secret_here
KITE_ACCESS_TOKEN=your_token_here
DATA_DIR=./data
LOG_LEVEL=INFO
TIMEZONE=Asia/Kolkata
```

## Import Organization

```python
# Standard library
import os
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import polars as pl
from pydantic import BaseModel

# Local
from apex_lab.config import settings
from apex_lab.utils import get_logger
```

## Review Checklist

Before submitting a PR:

- [ ] Code follows Black/Ruff standards
- [ ] All functions have type hints and docstrings
- [ ] Tests added and passing
- [ ] No `print()` statements (use logger)
- [ ] No hardcoded paths (use settings)
- [ ] Commit message follows Conventional Commits
- [ ] Coverage maintained or improved

---

**Last Updated**: 2026-07-15
