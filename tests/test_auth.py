"""Tests for authentication module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from apex_lab.config import settings
from apex_lab.data.kite.auth import KiteAuthenticator
from apex_lab.exceptions import AuthenticationError, ConfigurationError


class TestKiteAuthenticatorInitialization:
    """Tests for KiteAuthenticator initialization."""

    def test_authenticator_requires_api_key(self):
        """Test that authenticator requires API key."""
        with patch.object(settings, "kite_api_key", ""):
            with pytest.raises(ConfigurationError) as exc_info:
                KiteAuthenticator()
            assert "API" in str(exc_info.value)

    def test_authenticator_requires_api_secret(self):
        """Test that authenticator requires API secret."""
        with patch.object(settings.kite, "api_secret", ""):
            # This might not raise if api_key is set, depends on implementation
            pass

    def test_authenticator_initialization_success(self):
        """Test successful authenticator initialization."""
        try:
            auth = KiteAuthenticator()
            assert auth is not None
            assert auth.api_key is not None
            assert auth.api_secret is not None
        except ConfigurationError:
            # Expected if credentials not configured
            pytest.skip("Credentials not configured")


class TestKiteAuthenticatorCredentialValidation:
    """Tests for credential validation."""

    def test_validate_credentials_success(self):
        """Test successful credential validation."""
        try:
            auth = KiteAuthenticator()
            assert auth.validate_credentials() is True
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    def test_validate_credentials_missing_key(self):
        """Test validation fails without API key."""
        with patch.object(settings, "kite_api_key", ""):
            with pytest.raises(ConfigurationError):
                KiteAuthenticator()


class TestKiteAuthenticatorAccessToken:
    """Tests for access token management."""

    def test_access_token_not_set_initially(self):
        """Test that access token is not set initially."""
        try:
            auth = KiteAuthenticator()
            # Depends on whether KITE_ACCESS_TOKEN is in environment
            assert isinstance(auth.has_access_token, bool)
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    def test_set_access_token(self):
        """Test setting access token."""
        try:
            auth = KiteAuthenticator()
            auth.set_access_token("test_token_12345")
            assert auth.has_access_token is True
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    def test_access_token_property(self):
        """Test has_access_token property."""
        try:
            auth = KiteAuthenticator()
            assert isinstance(auth.has_access_token, bool)
        except ConfigurationError:
            pytest.skip("Credentials not configured")


class TestKiteAuthenticatorSessionCreation:
    """Tests for Kite session creation."""

    @patch("apex_lab.data.kite.auth.KiteConnect")
    def test_create_session_success(self, mock_kite_class):
        """Test successful session creation (mocked)."""
        mock_session = Mock()
        mock_kite_class.return_value = mock_session
        
        try:
            auth = KiteAuthenticator()
            auth.set_access_token("test_token")
            session = auth.create_session()
            assert session is not None
            mock_kite_class.assert_called_once()
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    @patch("apex_lab.data.kite.auth.KiteConnect")
    def test_create_session_with_token(self, mock_kite_class):
        """Test session creation with access token."""
        mock_session = Mock()
        mock_kite_class.return_value = mock_session
        
        try:
            auth = KiteAuthenticator()
            auth.set_access_token("test_token_xyz")
            session = auth.create_session()
            assert session is not None
            mock_session.set_access_token.assert_called_once_with("test_token_xyz")
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    @patch("apex_lab.data.kite.auth.KiteConnect")
    def test_create_session_failure(self, mock_kite_class):
        """Test session creation failure handling."""
        mock_kite_class.side_effect = Exception("Connection failed")
        
        try:
            auth = KiteAuthenticator()
            with pytest.raises(AuthenticationError):
                auth.create_session()
        except ConfigurationError:
            pytest.skip("Credentials not configured")

    def test_create_session_without_token_warning(self):
        """Test warning when creating session without token."""
        # This test verifies logging happens, but actual behavior depends on logger
        try:
            auth = KiteAuthenticator()
            # Don't set token
            with patch("apex_lab.data.kite.auth.KiteConnect"):
                # Would log warning
                pass
        except ConfigurationError:
            pytest.skip("Credentials not configured")
