"""Custom exceptions for APEX Lab.

All exceptions inherit from ApexError for centralized error handling.
"""


class ApexError(Exception):
    """Base exception for all APEX Lab errors."""

    pass


class ConfigurationError(ApexError):
    """Raised when configuration is invalid or incomplete."""

    pass


class AuthenticationError(ApexError):
    """Raised when authentication fails or credentials are invalid."""

    pass


class DownloadError(ApexError):
    """Raised when data download fails."""

    pass


class ValidationError(ApexError):
    """Raised when data validation fails."""

    pass


class RateLimitError(ApexError):
    """Raised when rate limit is exceeded."""

    pass
