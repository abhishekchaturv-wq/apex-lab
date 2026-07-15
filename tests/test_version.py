"""Test version import and validation."""

from apex_lab import __version__


def test_version_import():
    """Test that __version__ can be imported from apex_lab."""
    assert __version__ is not None


def test_version_value():
    """Test that __version__ has the expected value."""
    assert __version__ == "0.1.0"


def test_version_is_string():
    """Test that __version__ is a string."""
    assert isinstance(__version__, str)
