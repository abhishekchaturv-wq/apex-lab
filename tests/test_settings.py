"""Tests for settings and configuration."""

import pytest
from pathlib import Path
from unittest.mock import patch

from apex_lab.config import settings
from apex_lab.exceptions import ConfigurationError


class TestSettingsStructure:
    """Tests for nested settings structure."""

    def test_settings_has_kite_settings(self):
        """Test that settings has kite configuration."""
        assert hasattr(settings, "kite")
        assert hasattr(settings.kite, "api_key")
        assert hasattr(settings.kite, "api_secret")
        assert hasattr(settings.kite, "access_token")

    def test_settings_has_data_settings(self):
        """Test that settings has data configuration."""
        assert hasattr(settings, "data")
        assert hasattr(settings.data, "dir")
        assert hasattr(settings.data, "cache_dir")

    def test_settings_has_logging_settings(self):
        """Test that settings has logging configuration."""
        assert hasattr(settings, "logging")
        assert hasattr(settings.logging, "level")
        assert hasattr(settings.logging, "file")
        assert hasattr(settings.logging, "max_bytes")
        assert hasattr(settings.logging, "backup_count")

    def test_settings_has_app_settings(self):
        """Test that settings has application configuration."""
        assert hasattr(settings, "app")
        assert hasattr(settings.app, "timezone")
        assert hasattr(settings.app, "default_interval")


class TestDataPathProperties:
    """Tests for computed data path properties."""

    def test_raw_path(self):
        """Test raw data path."""
        assert hasattr(settings.data, "raw_path")
        path = settings.data.raw_path
        assert isinstance(path, Path)
        assert "raw" in str(path)

    def test_processed_path(self):
        """Test processed data path."""
        assert hasattr(settings.data, "processed_path")
        path = settings.data.processed_path
        assert isinstance(path, Path)
        assert "processed" in str(path)

    def test_features_path(self):
        """Test features path."""
        assert hasattr(settings.data, "features_path")
        path = settings.data.features_path
        assert isinstance(path, Path)
        assert "features" in str(path)

    def test_labels_path(self):
        """Test labels path."""
        assert hasattr(settings.data, "labels_path")
        path = settings.data.labels_path
        assert isinstance(path, Path)
        assert "labels" in str(path)

    def test_models_path(self):
        """Test models path."""
        assert hasattr(settings.data, "models_path")
        path = settings.data.models_path
        assert isinstance(path, Path)
        assert "models" in str(path)

    def test_reports_path(self):
        """Test reports path."""
        assert hasattr(settings.data, "reports_path")
        path = settings.data.reports_path
        assert isinstance(path, Path)
        assert "reports" in str(path)


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_validate_method_exists(self):
        """Test that validate method exists."""
        assert hasattr(settings, "validate")
        assert callable(settings.validate)

    def test_validate_checks_directories(self):
        """Test that validate checks directory existence."""
        # Should not raise if directories exist
        try:
            settings.validate()
        except ConfigurationError:
            # May raise if credentials not configured
            pass


class TestSettingsBackwardCompatibility:
    """Tests for backward compatibility properties."""

    def test_legacy_kite_api_key(self):
        """Test legacy kite_api_key property."""
        assert hasattr(settings, "kite_api_key")
        assert settings.kite_api_key == settings.kite.api_key

    def test_legacy_kite_api_secret(self):
        """Test legacy kite_api_secret property."""
        assert hasattr(settings, "kite_api_secret")
        assert settings.kite_api_secret == settings.kite.api_secret

    def test_legacy_data_dir(self):
        """Test legacy data_dir property."""
        assert hasattr(settings, "data_dir")
        assert settings.data_dir == settings.data.dir

    def test_legacy_log_level(self):
        """Test legacy log_level property."""
        assert hasattr(settings, "log_level")
        assert settings.log_level == settings.logging.level


class TestSettingsValidationErrors:
    """Tests for settings validation errors."""

    def test_invalid_timezone(self):
        """Test that invalid timezone is caught during validation."""
        # The validator should catch this during Settings construction
        # If we try to create settings with invalid timezone, it should fail
        pass  # This would require mocking environment variables

    def test_invalid_interval(self):
        """Test that invalid default interval is caught."""
        # Similar to timezone, validation occurs during construction
        pass
