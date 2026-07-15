# Dataset Builder

## Overview

The dataset pipeline builds a reproducible ML dataset in one command by combining:

- historical OHLCV input
- `FeatureEngine`
- `LabelEngine`

The output includes:

- full dataset
- metadata
- train split
- validation split
- test split

## Modules

- `builder.py`: End-to-end orchestration
- `metadata.py`: Versioned metadata + deterministic dataset ID
- `splitter.py`: Chronological train/validation/test split
- `validator.py`: Dataset quality checks
- `serializer.py`: Parquet + JSON persistence

## One-command build

Use `build_reproducible_dataset` with `DatasetBuildConfig`:

- computes features
- computes labels
- validates dataset
- splits dataset
- writes artifacts (if `output_dir` provided)

## Metadata fields

- dataset id (`dataset_id`)
- git SHA (`git_sha`, optional)
- feature version
- label version
- date range
- symbols
- timeframe
- number of rows
- class balance
- generation timestamp

## Validation rules

- no duplicate timestamps
- no missing labels
- no null/NaN values outside warm-up (except explicitly allowed columns)
- consistent schema (optional schema lock)

## Persistence

Artifacts are stored under:

`<output_dir>/<dataset_id>/`

Files:

- `dataset.parquet`
- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `metadata.json`
