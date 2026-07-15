# Baseline Predictive Models

## Objective

Sprint 7 introduces the first predictive reversal model to measure predictive power (not profitability).

## Implemented models

The training pipeline supports:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Modules

`src/apex_lab/models/`

- `trainer.py`: baseline model training and output persistence
- `metrics.py`: precision, recall, F1, ROC AUC, PR AUC, confusion matrix, calibration
- `inference.py`: model loading and probability inference
- `importance.py`: permutation and model-native feature importance
- `calibration.py`: reliability diagnostics and Brier score

## Output artifacts

Training writes:

- model artifact (`<model_name>.pkl`)
- `metrics.json`
- `feature_importance.csv`
- `prediction_probabilities.csv`

## Research sprint baseline analysis

Use the reproducible research pipeline to run end-to-end baseline predictive analysis in one command:

```bash
python -m apex_lab.models.experiment --input-dir data/raw/30m --output-dir reports
```

Expected report files:

- `reports/metrics.json`
- `reports/feature_importance.csv`
- `reports/threshold_analysis.csv`
- `reports/confusion_matrix.csv`
- `reports/calibration.csv`
- `reports/probability_distribution.csv`
- `reports/feature_correlation.csv`
- `reports/experiment.json`

## Usage

Use `train_baseline_model` with a generated dataset that has numeric feature columns and one binary target column.

```python
from pathlib import Path

from apex_lab.models import train_baseline_model

result = train_baseline_model(
    dataset=df,
    target_column="target",
    model_name="logistic_regression",
    output_dir=Path("artifacts/models/baseline"),
)
```

Use `predict_from_artifact` or `predict_probabilities` to generate prediction probabilities from trained artifacts.

## Tests

`tests/test_models.py` covers:

- model training
- inference
- metrics computation
- serialization roundtrip
