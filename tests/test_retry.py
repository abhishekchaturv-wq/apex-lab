"""Tests for retry logic."""

import pytest
from unittest.mock import patch
import time

from apex_lab.data.kite.retry import retry, should_retry
from apex_lab.exceptions import (
    AuthenticationError,
    RateLimitError,
)


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_retry_immediate_success(self):
        """Test function succeeds on first attempt."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.001)
        def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 1

    def test_retry_eventual_success(self):
        """Test function succeeds after transient failures."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.001)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        """Test that max retries are respected."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.001)
        def func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent error")

        with pytest.raises(ConnectionError):
            func()
        
        # Should have attempted max_retries + 1 times (initial + retries)
        assert call_count == 3

    def test_retry_authentication_not_retried(self):
        """Test that authentication errors are not retried."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.001)
        def func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed")

        with pytest.raises(AuthenticationError):
            func()
        
        # Should fail immediately, no retries
        assert call_count == 1

    def test_retry_with_arguments(self):
        """Test retry decorator with function arguments."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.001)
        def func(x, y):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient error")
            return x + y

        result = func(1, 2)
        assert result == 3
        assert call_count == 2

    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        @retry(max_retries=2, base_delay=0.01, max_delay=1.0, jitter=False)
        def func():
            raise ConnectionError("Error")

        start = time.time()
        with pytest.raises(ConnectionError):
            func()
        elapsed = time.time() - start
        
        # Should have delays: 0.01 + 0.02 = 0.03 seconds minimum
        assert elapsed >= 0.02

    def test_retry_with_rate_limit(self):
        """Test retry with rate limit error (transient)."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.001)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limited")
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 2


class TestShouldRetryFunction:
    """Tests for should_retry utility function."""

    def test_should_retry_connection_error(self):
        """Test that ConnectionError triggers retry."""
        assert should_retry(ConnectionError("Connection failed")) is True

    def test_should_retry_timeout_error(self):
        """Test that TimeoutError triggers retry."""
        assert should_retry(TimeoutError("Timeout")) is True

    def test_should_retry_rate_limit_error(self):
        """Test that RateLimitError triggers retry."""
        assert should_retry(RateLimitError("Rate limited")) is True

    def test_should_not_retry_authentication_error(self):
        """Test that AuthenticationError does not trigger retry."""
        assert should_retry(AuthenticationError("Auth failed")) is False

    def test_should_not_retry_value_error(self):
        """Test that ValueError does not trigger retry."""
        assert should_retry(ValueError("Invalid value")) is False

    def test_should_not_retry_key_error(self):
        """Test that KeyError does not trigger retry."""
        assert should_retry(KeyError("Key not found")) is False
