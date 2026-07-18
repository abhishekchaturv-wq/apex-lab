"""Main Pine Script generator for the Apex Research Engine.

This module loads the three required research output files, delegates rendering
to :mod:`apex_lab.export.renderer`, builds the strategy summary via
:mod:`apex_lab.export.serializer`, and writes the outputs to disk.

Usage (programmatic)::

    from apex_lab.export.pine_generator import generate_pine_strategy

    pine_path, summary_path = generate_pine_strategy()

Usage (CLI)::

    python scripts/research_lab.py --mode pine
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apex_lab.export.renderer import render_pine_script
from apex_lab.export.rule_translator import RuleTranslator, TranslatedSignal
from apex_lab.export.serializer import build_summary
from apex_lab.export.signal_pattern_loader import SignalPatternLoader

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_WALKFORWARD_PARAMS = Path("reports/lab/walkforward/best_parameters.json")
DEFAULT_CONTEXT_FEATURES = Path("reports/lab/context/best_features.json")
DEFAULT_ALPHA_WEIGHTS = Path("reports/lab/alpha/weights.json")

DEFAULT_OUTPUT_DIR = Path("generated")
DEFAULT_PINE_OUTPUT = DEFAULT_OUTPUT_DIR / "strategy.pine"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "strategy_summary.json"
DEFAULT_SIGNAL_PATTERNS = Path("reports/lab/signal_patterns/top_signals.json")
DEFAULT_QUANTILES = Path("reports/lab/signal_dataset/quantiles.json")


def _load_json(path: Path) -> Any:
    """Load a JSON file, raising a clear :class:`RuntimeError` if absent.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        RuntimeError: When *path* does not exist.
    """
    if not path.exists():
        raise RuntimeError(
            f"Required input file is missing: {path}\n"
            f"Run the corresponding research mode first to generate this file.\n"
            f"  - {DEFAULT_WALKFORWARD_PARAMS}: run --mode optimize\n"
            f"  - {DEFAULT_CONTEXT_FEATURES}: run --mode context\n"
            f"  - {DEFAULT_ALPHA_WEIGHTS}: run --mode alpha"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class PineGenerator:
    """Generates a TradingView Pine Script v5 strategy from research outputs.

    Args:
        walkforward_params_path: Path to ``best_parameters.json``.
        context_features_path: Path to ``best_features.json``.
        alpha_weights_path: Path to ``weights.json``.
        output_dir: Directory where ``strategy.pine`` and
            ``strategy_summary.json`` will be written.
    """

    def __init__(
        self,
        walkforward_params_path: Path = DEFAULT_WALKFORWARD_PARAMS,
        context_features_path: Path = DEFAULT_CONTEXT_FEATURES,
        alpha_weights_path: Path = DEFAULT_ALPHA_WEIGHTS,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        signal_patterns_path: Path | None = None,
        quantiles_path: Path | None = None,
    ) -> None:
        self.walkforward_params_path = walkforward_params_path
        self.context_features_path = context_features_path
        self.alpha_weights_path = alpha_weights_path
        self.output_dir = output_dir
        self.signal_patterns_path = signal_patterns_path
        self.quantiles_path = quantiles_path
        self.signal_pattern_loader = SignalPatternLoader()
        self.rule_translator = RuleTranslator()

    def _load_inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Load and validate all three required input files.

        Returns:
            A three-tuple of (best_params, best_features, weights_data).

        Raises:
            RuntimeError: If any required file is missing.
        """
        best_params: dict[str, Any] = _load_json(self.walkforward_params_path)
        best_features: dict[str, Any] = _load_json(self.context_features_path)
        weights_data: dict[str, Any] = _load_json(self.alpha_weights_path)
        return best_params, best_features, weights_data

    def _load_signal_inputs(self) -> tuple[TranslatedSignal, dict[str, dict[str, list[float]]]]:
        """Load discovered signal pattern inputs for signal export mode."""
        if self.signal_patterns_path is None:
            raise RuntimeError("signal_patterns_path is required for signal-pattern Pine export")
        if self.quantiles_path is None:
            raise RuntimeError("quantiles_path is required for signal-pattern Pine export")
        pattern = self.signal_pattern_loader.load_top_signal(self.signal_patterns_path)
        quantiles = self.rule_translator.load_quantiles(self.quantiles_path)
        translated = self.rule_translator.translate(pattern, quantiles)
        return translated, quantiles

    def generate(self) -> tuple[Path, Path]:
        """Run the full generation pipeline.

        Loads inputs, renders the Pine Script and summary, then writes both
        files to :attr:`output_dir`.

        Returns:
            A two-tuple of (pine_output_path, summary_output_path).

        Raises:
            RuntimeError: If any required input file is missing.
        """
        if self.signal_patterns_path is not None or self.quantiles_path is not None:
            translated_signal, _ = self._load_signal_inputs()
            pine_script = render_pine_script(
                best_params={},
                best_features={},
                weights_data={},
                translated_signal=translated_signal,
            )
            summary = build_summary(
                best_params={},
                best_features={},
                weights_data={},
                input_files=[
                    str(self.signal_patterns_path),
                    str(self.quantiles_path),
                ],
                signal_pattern=translated_signal.pattern,
            )
        else:
            best_params, best_features, weights_data = self._load_inputs()

            pine_script = render_pine_script(best_params, best_features, weights_data)
            summary = build_summary(
                best_params=best_params,
                best_features=best_features,
                weights_data=weights_data,
                input_files=[
                    str(self.walkforward_params_path),
                    str(self.context_features_path),
                    str(self.alpha_weights_path),
                ],
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        pine_path = self.output_dir / "strategy.pine"
        summary_path = self.output_dir / "strategy_summary.json"

        pine_path.write_text(pine_script, encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return pine_path, summary_path


def generate_pine_strategy(
    walkforward_params_path: Path = DEFAULT_WALKFORWARD_PARAMS,
    context_features_path: Path = DEFAULT_CONTEXT_FEATURES,
    alpha_weights_path: Path = DEFAULT_ALPHA_WEIGHTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    signal_patterns_path: Path | None = None,
    quantiles_path: Path | None = None,
) -> tuple[Path, Path]:
    """Convenience function: generate the Pine Script strategy.

    Args:
        walkforward_params_path: Path to ``best_parameters.json``.
        context_features_path: Path to ``best_features.json``.
        alpha_weights_path: Path to ``weights.json``.
        output_dir: Directory for generated output files.
        signal_patterns_path: Optional path to ``top_signals.json``.
        quantiles_path: Optional path to ``quantiles.json``.

    Returns:
        A two-tuple of (pine_output_path, summary_output_path).

    Raises:
        RuntimeError: If any required input file is missing.
    """
    generator = PineGenerator(
        walkforward_params_path=walkforward_params_path,
        context_features_path=context_features_path,
        alpha_weights_path=alpha_weights_path,
        output_dir=output_dir,
        signal_patterns_path=signal_patterns_path,
        quantiles_path=quantiles_path,
    )
    return generator.generate()
