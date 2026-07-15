"""Authentication module for Zerodha Kite Connect.

Handles session creation, credential validation, and session management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from kiteconnect import KiteConnect

from apex_lab.config import get_logger, settings
from apex_lab.exceptions import AuthenticationError, ConfigurationError

logger = get_logger(__name__)


class KiteAuthenticator:
    """Manages Zerodha Kite authentication and session.

    Responsibilities:
    - Create and maintain Kite session
    - Validate credentials
    - Handle session persistence
    - Never expose raw credentials in logs

    Attributes:
        api_key: Kite API key (from settings)
        api_secret: Kite API secret (from settings)
        _access_token: Current access token (should not be logged)
    """

    def __init__(self) -> None:
        """Initialize authenticator with credentials from settings.

        Raises:
            ConfigurationError: If required credentials are not configured
        """
        # Validate that credentials are configured
        if not settings.kite_api_key or not settings.kite_api_secret:
            raise ConfigurationError(
                "Kite API credentials not configured. "
                "Set KITE_API_KEY and KITE_API_SECRET in .env file."
            )

        self.api_key = settings.kite_api_key
        self.api_secret = settings.kite_api_secret
        self._access_token: Optional[str] = settings.kite_access_token or None

        logger.debug(f"KiteAuthenticator initialized with API key: {self.api_key[:8]}...")

    def create_session(self) -> KiteConnect:
        """Create and return an authenticated Kite session.

        Returns:
            Authenticated KiteConnect instance

        Raises:
            AuthenticationError: If session creation fails

        Note:
            The returned KiteConnect object should be treated as a singleton
            in the client and reused for all API calls.
        """
        try:
            kite = KiteConnect(api_key=self.api_key)

            # If access token is available, set it
            if self._access_token:
                kite.set_access_token(self._access_token)
                logger.debug("Kite session created with existing access token")
            else:
                logger.warning(
                    "No access token available. "
                    "You must obtain an access token via web login."
                )

            return kite

        except Exception as e:
            logger.error(f"Failed to create Kite session: {e}")
            raise AuthenticationError(f"Kite session creation failed: {e}") from e

    def validate_credentials(self) -> bool:
        """Validate that credentials are properly configured.

        Returns:
            True if credentials are valid

        Raises:
            AuthenticationError: If credentials are invalid
        """
        if not self.api_key:
            raise AuthenticationError("API key is not configured")

        if not self.api_secret:
            raise AuthenticationError("API secret is not configured")

        logger.debug("Credentials validation passed")
        return True

    def set_access_token(self, access_token: str) -> None:
        """Set the access token for authentication.

        Args:
            access_token: The access token obtained from Kite login

        Note:
            This token is used for API calls. Store it securely in .env.
        """
        self._access_token = access_token
        logger.debug("Access token updated")

    @property
    def has_access_token(self) -> bool:
        """Check if an access token is available.

        Returns:
            True if access token is set, False otherwise
        """
        return bool(self._access_token)
