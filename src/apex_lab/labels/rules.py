"""Configurable rules for objective reversal labels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelingRules:
    """Parameterized thresholds used by the labeling engine.

    Args:
        atr_column: ATR column to use for volatility scaling.
        atr_multiplier: Global ATR scaling factor applied to thresholds.
        reward_multiplier: Reward threshold in ATR units.
        risk_multiplier: Risk/failure threshold in ATR units.
        lookahead_window: Number of future candles examined.
    """

    atr_column: str = "atr_14"
    atr_multiplier: float = 1.0
    reward_multiplier: float = 2.0
    risk_multiplier: float = 1.0
    lookahead_window: int = 12

    def __post_init__(self) -> None:
        """Validate rule values."""
        if not self.atr_column:
            raise ValueError("atr_column must be non-empty")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if self.reward_multiplier <= 0:
            raise ValueError("reward_multiplier must be > 0")
        if self.risk_multiplier <= 0:
            raise ValueError("risk_multiplier must be > 0")
        if self.lookahead_window <= 0:
            raise ValueError("lookahead_window must be > 0")
