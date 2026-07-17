"""High-Expectancy Signal Discovery Engine."""

from apex_lab.research.signal_patterns.candidate_generator import (
    CandidateGeneratorConfig,
    CandidateRule,
    generate_candidates,
)
from apex_lab.research.signal_patterns.engine import (
    DEFAULT_OUTPUT_DIR,
    SignalPatternsResult,
    run_signal_patterns,
)

__all__ = [
    "CandidateGeneratorConfig",
    "CandidateRule",
    "DEFAULT_OUTPUT_DIR",
    "SignalPatternsResult",
    "generate_candidates",
    "run_signal_patterns",
]
