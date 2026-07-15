"""Logging configuration and logger factory.

Provides a centralized logger factory to ensure consistent logging
across the entire application. No print() statements should be used.
"""

import logging
import logging.config

# Standard logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with standardized configuration.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Configured logger instance

    Example:
        >>> from apex_lab.config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)

    # Read log level from settings when available; fall back to INFO so that
    # importing this module never requires Kite credentials to be present
    # (e.g. during offline test runs).
    try:
        from apex_lab.config.settings import settings  # noqa: PLC0415

        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    except Exception:
        log_level = logging.INFO

    logger.setLevel(log_level)
    
    # Return if handlers already configured
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
