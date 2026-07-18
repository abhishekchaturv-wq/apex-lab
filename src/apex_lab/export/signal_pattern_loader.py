"""Load discovered signal-pattern artifacts for Pine export."""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SignalPattern:
    """Normalized signal-pattern record loaded from JSON or CSV."""

    rule_label: str
    features: tuple[str, ...]
    conditions: tuple[str, ...]
    combination_size: int
    signal_frequency: int | None
    win_rate: float | None
    average_return: float | None
    expectancy: float | None
    average_mfe: float | None
    average_mae: float | None
    robustness: bool | None
    diversity_score: float | None
    composite_score: float | None
    rank: int | None = None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [stripped]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    return [str(value).strip()]


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return tuple(ordered)


def _derive_features(conditions: tuple[str, ...]) -> tuple[str, ...]:
    features: list[str] = []
    for condition in conditions:
        feature = condition.split(" ", 1)[0].strip()
        if feature:
            features.append(feature)
    return _dedupe(features)


def _normalize_record(record: dict[str, Any]) -> SignalPattern:
    rule_label = str(record.get("rule_label") or "").strip()
    conditions = _dedupe(_as_list(record.get("conditions")))
    if not conditions and rule_label:
        conditions = _dedupe([part.strip() for part in rule_label.split(" AND ") if part.strip()])
    if not rule_label and conditions:
        rule_label = " AND ".join(conditions)
    if not rule_label:
        raise ValueError("signal pattern record is missing rule_label/conditions")

    raw_features = record.get("features")
    features = _dedupe(_as_list(raw_features)) if raw_features else _derive_features(conditions)
    if not conditions:
        raise ValueError(f"signal pattern '{rule_label}' does not contain any conditions")

    combination_size = _parse_int(record.get("combination_size")) or len(features)
    return SignalPattern(
        rank=_parse_int(record.get("rank")),
        rule_label=rule_label,
        features=features,
        conditions=conditions,
        combination_size=combination_size,
        signal_frequency=_parse_int(record.get("signal_frequency")),
        win_rate=_parse_float(record.get("win_rate")),
        average_return=_parse_float(record.get("average_return")),
        expectancy=_parse_float(record.get("expectancy")),
        average_mfe=_parse_float(record.get("average_mfe")),
        average_mae=_parse_float(record.get("average_mae")),
        robustness=_parse_bool(record.get("is_robust", record.get("robustness"))),
        diversity_score=_parse_float(record.get("diversity_score")),
        composite_score=_parse_float(record.get("composite_score")),
    )


class SignalPatternLoader:
    """Load top-ranked signal patterns from research artifacts."""

    def load_top_signal(self, path: Path) -> SignalPattern:
        """Load the highest-ranked signal pattern from *path*."""
        records = self.load_records(path)
        if not records:
            raise ValueError(f"signal pattern file is empty: {path}")
        sorted_records = sorted(
            records,
            key=lambda item: (
                item.rank is None,
                item.rank if item.rank is not None else 0,
                item.rule_label,
            ),
        )
        return sorted_records[0]

    def load_records(self, path: Path) -> list[SignalPattern]:
        """Load all signal patterns from JSON or CSV."""
        if not path.exists():
            raise RuntimeError(f"Signal pattern file does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._load_json(path)
        if suffix == ".csv":
            return self._load_csv(path)
        raise RuntimeError(f"Unsupported signal pattern file format: {path.suffix}")

    def _load_json(self, path: Path) -> list[SignalPattern]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("top_20_signals"), list):
            records = payload["top_20_signals"]
        else:
            raise ValueError(
                f"Unsupported signal pattern JSON structure in {path}; "
                "expected a list or a payload containing top_20_signals."
            )
        return [_normalize_record(dict(record)) for record in records]

    def _load_csv(self, path: Path) -> list[SignalPattern]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [_normalize_record(dict(row)) for row in reader]
