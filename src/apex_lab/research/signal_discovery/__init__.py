"""Feature importance and signal discovery engine."""

from apex_lab.research.signal_discovery.combination import CombinationConfig
from apex_lab.research.signal_discovery.engine import (
    DEFAULT_OUTPUT_DIR,
    SignalDiscoveryResult,
    run_signal_discovery,
)

__all__ = [
    "CombinationConfig",
    "DEFAULT_OUTPUT_DIR",
    "SignalDiscoveryResult",
    "run_signal_discovery",
]
