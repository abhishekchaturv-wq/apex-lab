"""Translate discovered signal rules into Pine Script expressions."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex_lab.export.signal_pattern_loader import SignalPattern

_CONDITION_RE = re.compile(
    r"^\s*(?P<feature>[A-Za-z0-9_]+)\s*(?P<operator>==|!=|>=|<=|>|<)\s*(?P<value>.+?)\s*$"
)
_QUANTILE_RE = re.compile(r"^q(?P<index>\d+)$")


@dataclass(frozen=True)
class ParsedCondition:
    """Structured representation of one discovered rule clause."""

    feature: str
    operator: str
    value: str
    original: str


@dataclass(frozen=True)
class TranslatedSignal:
    """Fully translated signal payload for rendering."""

    pattern: SignalPattern
    parsed_conditions: tuple[ParsedCondition, ...]
    entry_conditions: tuple[str, ...]
    indicator_lines: tuple[str, ...]


class BaseFeatureTranslator(ABC):
    """Base class for Pine feature translators."""

    @abstractmethod
    def matches(self, feature: str) -> bool:
        """Return whether this translator supports *feature*."""

    @abstractmethod
    def indicator_lines(self, feature: str) -> tuple[str, ...]:
        """Return Pine indicator declarations required for *feature*."""

    @abstractmethod
    def pine_expression(self, feature: str) -> str:
        """Return the Pine expression used to reference *feature*."""

    def translate(
        self,
        condition: ParsedCondition,
        quantiles: dict[str, dict[str, list[float]]],
    ) -> str:
        """Translate a parsed rule condition into a Pine boolean expression."""
        if condition.operator != "==":
            raise ValueError(
                f"Unsupported operator '{condition.operator}' in condition '{condition.original}'. "
                "Only equality rules exported by signal discovery are supported."
            )

        quantile_match = _QUANTILE_RE.fullmatch(condition.value)
        expression = self.pine_expression(condition.feature)
        if quantile_match is not None:
            bounds = _resolve_quantile_bounds(condition.feature, condition.value, quantiles)
            return _range_expression(expression, bounds, label=condition.value)

        return f'{expression} == "{condition.value}"'


@dataclass(frozen=True)
class ExactFeatureTranslator(BaseFeatureTranslator):
    """Translator for a fixed set of feature names."""

    feature_names: tuple[str, ...]
    expression: str
    lines: tuple[str, ...] = ()

    def matches(self, feature: str) -> bool:
        return feature in self.feature_names

    def indicator_lines(self, feature: str) -> tuple[str, ...]:
        return self.lines

    def pine_expression(self, feature: str) -> str:
        return self.expression


class DynamicEMATranslator(BaseFeatureTranslator):
    """Translator for arbitrary ``ema_<period>`` features."""

    def matches(self, feature: str) -> bool:
        return bool(re.fullmatch(r"ema_\d+", feature))

    def indicator_lines(self, feature: str) -> tuple[str, ...]:
        period = feature.split("_", 1)[1]
        return (f"ema{period}Val        = ta.ema(close, {int(period)})",)

    def pine_expression(self, feature: str) -> str:
        period = feature.split("_", 1)[1]
        return f"ema{period}Val"


class FeatureTranslatorRegistry:
    """Ordered registry of feature translators."""

    def __init__(self, translators: list[BaseFeatureTranslator] | None = None) -> None:
        self._translators = translators or _default_translators()

    def resolve(self, feature: str) -> BaseFeatureTranslator:
        """Resolve the translator responsible for *feature*."""
        for translator in self._translators:
            if translator.matches(feature):
                return translator
        raise ValueError(
            f"Unsupported feature '{feature}' in discovered signal rule. "
            "Register a feature translator before exporting this pattern."
        )


class RuleTranslator:
    """Translate discovered signal rules into Pine-renderable expressions."""

    def __init__(self, registry: FeatureTranslatorRegistry | None = None) -> None:
        self.registry = registry or FeatureTranslatorRegistry()

    def load_quantiles(self, path: Path) -> dict[str, dict[str, list[float]]]:
        """Load persisted quantile metadata."""
        if not path.exists():
            raise RuntimeError(f"Quantile metadata file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid quantile metadata payload in {path}")
        normalized: dict[str, dict[str, list[float]]] = {}
        for feature, buckets in payload.items():
            if feature.startswith("_"):
                continue
            if not isinstance(buckets, dict):
                continue
            feature_buckets: dict[str, list[float]] = {}
            for bucket_name, raw_bounds in buckets.items():
                if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
                    continue
                feature_buckets[str(bucket_name)] = [float(raw_bounds[0]), float(raw_bounds[1])]
            if feature_buckets:
                normalized[str(feature)] = feature_buckets
        return normalized

    def translate(
        self,
        pattern: SignalPattern,
        quantiles: dict[str, dict[str, list[float]]],
    ) -> TranslatedSignal:
        """Translate *pattern* using persisted *quantiles* metadata."""
        parsed = tuple(_parse_condition(condition) for condition in pattern.conditions)
        seen_conditions: set[str] = set()
        entry_conditions: list[str] = []
        indicator_lines: list[str] = []
        seen_lines: set[str] = set()

        for condition in parsed:
            key = condition.original.strip()
            if key in seen_conditions:
                continue
            seen_conditions.add(key)
            translator = self.registry.resolve(condition.feature)
            entry_conditions.append(translator.translate(condition, quantiles))
            for line in translator.indicator_lines(condition.feature):
                if line not in seen_lines:
                    indicator_lines.append(line)
                    seen_lines.add(line)

        if not entry_conditions:
            raise ValueError(f"Signal pattern '{pattern.rule_label}' did not produce any Pine conditions")

        return TranslatedSignal(
            pattern=pattern,
            parsed_conditions=parsed,
            entry_conditions=tuple(entry_conditions),
            indicator_lines=tuple(indicator_lines),
        )


def _parse_condition(condition: str) -> ParsedCondition:
    match = _CONDITION_RE.fullmatch(condition)
    if match is None:
        raise ValueError(f"Invalid rule condition: '{condition}'")
    value = match.group("value").strip().strip("'").strip('"')
    return ParsedCondition(
        feature=match.group("feature"),
        operator=match.group("operator"),
        value=value,
        original=condition.strip(),
    )


def _resolve_quantile_bounds(
    feature: str,
    bucket: str,
    quantiles: dict[str, dict[str, list[float]]],
) -> tuple[float, float, bool, bool]:
    feature_buckets = quantiles.get(feature)
    if not feature_buckets:
        raise ValueError(
            f"Missing quantile metadata for feature '{feature}'. "
            "Regenerate the signal dataset quantiles.json artifact before exporting."
        )
    if bucket not in feature_buckets:
        available = ", ".join(sorted(feature_buckets))
        raise ValueError(
            f"Missing quantile bucket '{bucket}' for feature '{feature}'. "
            f"Available buckets: {available}"
        )
    lower, upper = feature_buckets[bucket]
    ordered = sorted(
        (
            (int(match.group("index")), name)
            for name in feature_buckets
            if (match := _QUANTILE_RE.fullmatch(name)) is not None
        ),
        key=lambda item: item[0],
    )
    if not ordered:
        return lower, upper, True, True
    first = ordered[0][1]
    last = ordered[-1][1]
    return lower, upper, bucket == first, bucket == last


def _format_number(value: float) -> str:
    return format(value, ".10g")


def _range_expression(
    expression: str,
    bounds: tuple[float, float, bool, bool],
    *,
    label: str,
) -> str:
    lower, upper, is_first, is_last = bounds
    lower_str = _format_number(lower)
    upper_str = _format_number(upper)
    if lower == upper:
        return f"{expression} == {lower_str}"

    parts: list[str] = []
    if is_first:
        parts.append(f"{expression} >= {lower_str}")
    else:
        parts.append(f"{expression} >= {lower_str}")

    if is_last:
        parts.append(f"{expression} <= {upper_str}")
    else:
        parts.append(f"{expression} < {upper_str}")

    return f"({' and '.join(parts)})"


def _default_translators() -> list[BaseFeatureTranslator]:
    return [
        DynamicEMATranslator(),
        ExactFeatureTranslator(("rsi",), "rsiVal", ("rsiVal          = ta.rsi(close, 14)",)),
        ExactFeatureTranslator(
            ("macd",),
            "macdLine",
            ("[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)",),
        ),
        ExactFeatureTranslator(
            ("macd_signal",),
            "signalLine",
            ("[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)",),
        ),
        ExactFeatureTranslator(
            ("macd_hist",),
            "macdHist",
            ("[macdLine, signalLine, macdHist] = ta.macd(close, 12, 26, 9)",),
        ),
        ExactFeatureTranslator(("atr_14",), "atrVal", ("atrVal          = ta.atr(14)",)),
        ExactFeatureTranslator(
            ("atr_norm",),
            "atrNormVal",
            (
                "atrVal          = ta.atr(14)",
                "atrNormVal      = atrVal / close * 100.0",
            ),
        ),
        ExactFeatureTranslator(("vwap",), "vwapVal", ("vwapVal         = ta.vwap(hlc3)",)),
        ExactFeatureTranslator(
            ("vwap_dist",),
            "vwapDistVal",
            (
                "vwapVal         = ta.vwap(hlc3)",
                "vwapDistVal     = (close - vwapVal) / vwapVal * 100.0",
            ),
        ),
        ExactFeatureTranslator(("swing_high",), "swingHighVal", ("swingHighVal    = ta.highest(high, 10)",)),
        ExactFeatureTranslator(("swing_low",), "swingLowVal", ("swingLowVal     = ta.lowest(low, 10)",)),
        ExactFeatureTranslator(
            ("volume",),
            "volume",
        ),
        ExactFeatureTranslator(
            ("avg_volume",),
            "avgVolumeVal",
            ("avgVolumeVal    = ta.sma(volume, 20)",),
        ),
        ExactFeatureTranslator(
            ("rel_volume",),
            "relVolumeVal",
            (
                "avgVolumeVal    = ta.sma(volume, 20)",
                "relVolumeVal    = volume / avgVolumeVal",
            ),
        ),
        ExactFeatureTranslator(("roc",), "rocVal", ("rocVal          = ta.roc(close, 10)",)),
        ExactFeatureTranslator(
            ("bb_width",),
            "bbWidthVal",
            (
                "bbBasis         = ta.sma(close, 20)",
                "bbDev           = ta.stdev(close, 20) * 2.0",
                "bbWidthVal      = (bbBasis + bbDev - (bbBasis - bbDev)) / bbBasis * 100.0",
            ),
        ),
        ExactFeatureTranslator(
            ("higher_high",),
            "higherHighVal",
            (
                "swingHighVal    = ta.highest(high, 10)",
                "higherHighVal   = swingHighVal > swingHighVal[1] ? 1.0 : 0.0",
            ),
        ),
        ExactFeatureTranslator(
            ("higher_low",),
            "higherLowVal",
            (
                "swingLowVal     = ta.lowest(low, 10)",
                "higherLowVal    = swingLowVal > swingLowVal[1] ? 1.0 : 0.0",
            ),
        ),
        ExactFeatureTranslator(
            ("lower_high",),
            "lowerHighVal",
            (
                "swingHighVal    = ta.highest(high, 10)",
                "lowerHighVal    = swingHighVal < swingHighVal[1] ? 1.0 : 0.0",
            ),
        ),
        ExactFeatureTranslator(
            ("lower_low",),
            "lowerLowVal",
            (
                "swingLowVal     = ta.lowest(low, 10)",
                "lowerLowVal     = swingLowVal < swingLowVal[1] ? 1.0 : 0.0",
            ),
        ),
    ]
