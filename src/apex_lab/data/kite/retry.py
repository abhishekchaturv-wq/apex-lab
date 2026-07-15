"""Retry logic with exponential backoff.

Provides decorators and utilities for retrying failed operations with
exponential backoff and configurable thresholds.
"""

from __future__ import annotations

import random
import time
from functools import wraps
from typing import Callable, Optional, TypeVar, Any

from apex_lab.config import get_logger
from apex_lab.exceptions import RateLimitError, AuthenticationError, ApexError

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Transient errors that should trigger retries
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    RateLimitError,
)

# Non-transient errors that should NOT be retried
NON_TRANSIENT_ERRORS = (
    AuthenticationError,
    ValueError,
    KeyError,
)


def should_retry(exception: Exception) -> bool:
    """Determine if an exception should trigger a retry.

    Args:
        exception: The exception that was raised

    Returns:
        True if the operation should be retried, False otherwise
    """
    # Never retry non-transient errors
    if isinstance(exception, NON_TRANSIENT_ERRORS):
        return False

    # Retry transient errors
    return isinstance(exception, TRANSIENT_ERRORS)


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> Callable[[F], F]:
    """Decorator for retrying operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delays

    Returns:
        Decorated function that retries on transient failures

    Example:
        @retry(max_retries=3, base_delay=1.0)
        def fetch_data():
            # Function that may fail transiently
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry non-transient errors
                    if not should_retry(e):
                        logger.error(
                            f"Non-transient error in {func.__name__}: {type(e).__name__}: {e}"
                        )
                        raise

                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        logger.error(
                            f"Max retries exceeded for {func.__name__} after {max_retries} attempts"
                        )
                        raise

                    # Calculate delay with jitter
                    actual_delay = min(delay, max_delay)
                    if jitter:
                        actual_delay += random.uniform(0, actual_delay * 0.1)

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {actual_delay:.2f}s. Error: {type(e).__name__}: {e}"
                    )

                    time.sleep(actual_delay)
                    delay *= 2  # Exponential backoff

            # Should never reach here, but just in case
            raise last_exception or ApexError("Unexpected retry failure")

        return wrapper  # type: ignore

    return decorator
