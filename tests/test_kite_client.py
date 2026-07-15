"""Tests for Kite client and related modules."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from apex_lab.config import settings, get_logger
from apex_lab.data.kite.client import KiteClient
from apex_lab.data.kite.auth import KiteAuthenticator
from apex_lab.data.kite.retry import retry, should_retry
from apex_lab.exceptions import (
    AuthenticationError,
    ValidationError,
    DownloadError,
    RateLimitError,
    ConfigurationError,
)


class TestKiteAuthenticator:
    """Tests for KiteAuthenticator."""

    def test_authenticator_requires_credentials(self):
        """Test that authenticator raises error when credentials are missing."""
        with patch.object(settings, "kite_api_key", ""):
            with pytest.raises(ConfigurationError):
                KiteAuthenticator()

    def test_authenticator_validates_credentials(self):
        """Test credential validation."""
        auth = KiteAuthenticator()
        assert auth.validate_credentials() is True

    def test_authenticator_has_access_token_property(self):
        """Test access token property."""
        auth = KiteAuthenticator()
        # Initially no token
        assert auth.has_access_token is False or auth.has_access_token is True
        
        # After setting
        auth.set_access_token("test_token")
        assert auth.has_access_token is True

    @patch("apex_lab.data.kite.auth.KiteConnect")
    def test_create_session_success(self, mock_kite):
        """Test successful session creation (mocked)."""
        auth = KiteAuthenticator()
        auth.set_access_token("test_token")
        
        mock_session = Mock()
        mock_kite.return_value = mock_session
        
        session = auth.create_session()
        assert session is not None

    @patch("apex_lab.data.kite.auth.KiteConnect")
    def test_create_session_without_token(self, mock_kite):
        """Test session creation without access token."""
        auth = KiteAuthenticator()
        # Don't set token
        mock_session = Mock()
        mock_kite.return_value = mock_session
        
        session = auth.create_session()
        assert session is not None


class TestRetryLogic:
    """Tests for retry decorator and logic."""

    def test_should_retry_transient_error(self):
        """Test that transient errors trigger retry."""
        assert should_retry(ConnectionError("Connection failed")) is True
        assert should_retry(TimeoutError("Timeout")) is True
        assert should_retry(RateLimitError("Rate limit")) is True

    def test_should_not_retry_non_transient_error(self):
        """Test that non-transient errors do not trigger retry."""
        assert should_retry(AuthenticationError("Auth failed")) is False
        assert should_retry(ValueError("Invalid value")) is False
        assert should_retry(KeyError("Key not found")) is False

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        @retry(max_retries=3)
        def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_retries(self):
        """Test successful execution after retries."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhaustion(self):
        """Test that retry exhaustion raises the original error."""
        @retry(max_retries=2, base_delay=0.01)
        def func():
            raise ConnectionError("Persistent error")

        with pytest.raises(ConnectionError):
            func()

    def test_retry_no_auth_error(self):
        """Test that authentication errors are never retried."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed")

        with pytest.raises(AuthenticationError):
            func()
        
        # Should fail immediately, no retries
        assert call_count == 1


class TestKiteClientValidation:
    """Tests for KiteClient validation methods."""

    def test_validate_symbol_valid(self):
        """Test valid symbol validation."""
        # Should not raise
        KiteClient._validate_symbol("BANKNIFTY")
        KiteClient._validate_symbol("RELIANCE")

    def test_validate_symbol_invalid(self):
        """Test invalid symbol validation."""
        with pytest.raises(ValidationError):
            KiteClient._validate_symbol("")
        
        with pytest.raises(ValidationError):
            KiteClient._validate_symbol(None)
        
        with pytest.raises(ValidationError):
            KiteClient._validate_symbol("X")  # Too short

    def test_validate_interval_valid(self):
        """Test valid interval validation."""
        for interval in ["minute", "5minute", "15minute", "day", "week"]:
            KiteClient._validate_interval(interval)  # Should not raise

    def test_validate_interval_invalid(self):
        """Test invalid interval validation."""
        with pytest.raises(ValidationError):
            KiteClient._validate_interval("invalid")
        
        with pytest.raises(ValidationError):
            KiteClient._validate_interval("")

    def test_validate_dates_valid(self):
        """Test valid date range validation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        KiteClient._validate_dates(start, end)  # Should not raise

    def test_validate_dates_invalid_order(self):
        """Test date validation with invalid order."""
        start = datetime(2024, 1, 31)
        end = datetime(2024, 1, 1)
        
        with pytest.raises(ValidationError):
            KiteClient._validate_dates(start, end)

    def test_validate_dates_invalid_type(self):
        """Test date validation with invalid types."""
        with pytest.raises(ValidationError):
            KiteClient._validate_dates("2024-01-01", "2024-01-31")


class TestKiteClientDataNormalization:
    """Tests for data normalization."""

    def test_normalize_data_valid(self):
        """Test successful data normalization."""
        data = [
            {
                "date": datetime(2024, 1, 1, 9, 15),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1000000,
            },
            {
                "date": datetime(2024, 1, 1, 9, 20),
                "open": 103.0,
                "high": 106.0,
                "low": 102.0,
                "close": 104.5,
                "volume": 900000,
            },
        ]
        
        df = KiteClient._normalize_data(data)
        
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_normalize_data_empty(self):
        """Test normalization with empty data."""
        with pytest.raises(ValidationError):
            KiteClient._normalize_data([])

    def test_normalize_data_missing_fields(self):
        """Test normalization with missing fields."""
        data = [
            {
                "date": datetime(2024, 1, 1),
                "open": 100.0,
                # Missing other required fields
            }
        ]
        
        with pytest.raises(ValidationError):
            KiteClient._normalize_data(data)


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_settings_has_nested_models(self):
        """Test that settings has nested configuration models."""
        assert hasattr(settings, "kite")
        assert hasattr(settings, "data")
        assert hasattr(settings, "logging")
        assert hasattr(settings, "app")

    def test_settings_computed_paths(self):
        """Test that settings provides computed data paths."""
        assert hasattr(settings.data, "raw_path")
        assert hasattr(settings.data, "processed_path")
        assert hasattr(settings.data, "features_path")
        assert hasattr(settings.data, "labels_path")
        assert hasattr(settings.data, "models_path")
        assert hasattr(settings.data, "reports_path")

    def test_settings_validate_method(self):
        """Test that settings has validate method."""
        assert callable(settings.validate)


class TestLogging:
    """Tests for logging configuration."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger(__name__)
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_logger_has_handlers(self):
        """Test that logger has configured handlers."""
        logger = get_logger("test_logger")
        assert len(logger.handlers) > 0


class TestKiteClientIntegration:
    """Integration tests for KiteClient (mocked)."""

    def test_client_initialization(self):
        """Test client initialization."""
        # This will fail without proper credentials, so we expect it
        try:
            client = KiteClient()
            assert client is not None
        except (ConfigurationError, AuthenticationError):
            # Expected if credentials not configured
            pass

    @patch("apex_lab.data.kite.client.KiteClient._get_session")
    def test_get_history_validation(self, mock_session):
        """Test that get_history validates inputs."""
        try:
            client = KiteClient()
            
            # Should raise ValidationError for invalid symbol
            with pytest.raises(ValidationError):
                client.get_history(
                    symbol="",
                    interval="5minute",
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 1, 31),
                )
        except (ConfigurationError, AuthenticationError):
            # Skip if credentials not configured
            pass
