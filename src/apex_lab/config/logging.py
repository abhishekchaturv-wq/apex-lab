"""Updated logging configuration with file rotation.

Provides centralized logger factory with rotating file and console output.
"""

import logging
import logging.handlers
from typing import Optional

from apex_lab.config.settings import settings

# Standard logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with standardized configuration.

    Provides both console and file output with rotating file handler.

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

    # Set level from settings
    log_level = getattr(
        logging, settings.logging.level.upper(), logging.INFO
    )
    logger.setLevel(log_level)

    # Return if handlers already configured
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.logging.file,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not configure file logging: {e}")

    return logger
