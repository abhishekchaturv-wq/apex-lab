"""Feature groups sub-package.

Importing this package triggers self-registration of all built-in feature
groups into :data:`~apex_lab.features.registry.default_registry`.
"""

from apex_lab.features.groups import (
    momentum,
    price,
    structure,
    time,
    trend,
    volatility,
    volume,
)

__all__ = [
    "momentum",
    "price",
    "structure",
    "time",
    "trend",
    "volatility",
    "volume",
]
