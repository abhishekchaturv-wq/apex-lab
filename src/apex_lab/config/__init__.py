"""Configuration module for APEX Lab.

Provides typed settings management and logging factory.
"""

from apex_lab.config.logging import get_logger
from apex_lab.config.settings import Settings, settings

__all__ = ["Settings", "settings", "get_logger"]
