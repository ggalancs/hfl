# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for centralized logging configuration."""

import json
import logging

import pytest

from hfl.core.tracing import clear_request_id
from hfl.logging_config import (
    PrettyFormatter,
    StructuredFormatter,
    configure_logging,
    get_logger,
    get_request_id,
    log_error,
    log_model_load,
    log_request,
    set_request_id,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset logging state after each test to prevent pollution."""
    # Store original state
    hfl_logger = logging.getLogger("hfl")
    original_level = hfl_logger.level
    original_handlers = hfl_logger.handlers.copy()
    original_propagate = hfl_logger.propagate

    yield

    # Restore original state
    hfl_logger.setLevel(original_level)
    hfl_logger.handlers = original_handlers
    hfl_logger.propagate = original_propagate


class TestDebugEnvOverride:
    """``HFL_DEBUG`` / ``OLLAMA_DEBUG`` flip the hfl logger to DEBUG
    without code changes."""

    def test_hfl_debug_truthy_forces_debug(self, monkeypatch):
        monkeypatch.setenv("HFL_DEBUG", "1")
        configure_logging(level="INFO")
        assert logging.getLogger("hfl").level == logging.DEBUG

    def test_hfl_debug_alias_values(self, monkeypatch):
        for value in ("true", "TRUE", "yes", "on"):
            monkeypatch.setenv("HFL_DEBUG", value)
            configure_logging(level="WARNING")
            assert logging.getLogger("hfl").level == logging.DEBUG

    def test_ollama_debug_alias(self, monkeypatch):
        monkeypatch.delenv("HFL_DEBUG", raising=False)
        monkeypatch.setenv("OLLAMA_DEBUG", "1")
        configure_logging(level="ERROR")
        assert logging.getLogger("hfl").level == logging.DEBUG

    def test_unset_keeps_explicit_level(self, monkeypatch):
        monkeypatch.delenv("HFL_DEBUG", raising=False)
        monkeypatch.delenv("OLLAMA_DEBUG", raising=False)
        configure_logging(level="WARNING")
        assert logging.getLogger("hfl").level == logging.WARNING

    def test_falsy_values_do_not_force_debug(self, monkeypatch):
        for value in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("HFL_DEBUG", value)
            configure_logging(level="INFO")
            assert logging.getLogger("hfl").level == logging.INFO


class TestRequestIdTracing:
    """Tests for request ID context variable."""

    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        # Reset context
        clear_request_id()

        request_id = set_request_id("test-123")
        assert request_id == "test-123"
        assert get_request_id() == "test-123"

    def test_auto_generate_request_id(self):
        """Test auto-generation of request ID."""
        clear_request_id()

        request_id = set_request_id()
        assert request_id is not None
        assert len(request_id) == 8  # 8 hex chars
        assert get_request_id() == request_id

    def test_get_request_id_when_not_set(self):
        """Test getting request ID when not set."""
        clear_request_id()
        assert get_request_id() is None


class TestStructuredFormatter:
    """Tests for JSON structured formatter."""

    def test_basic_format(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="hfl",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "hfl"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_request_id(self):
        """Test formatting includes request ID."""
        set_request_id("req-456")

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="hfl",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["request_id"] == "req-456"

        # Clean up
        clear_request_id()

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="hfl",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Request",
            args=(),
            exc_info=None,
        )
        record.method = "POST"
        record.path = "/v1/chat/completions"
        record.status = 200
        record.duration_ms = 150.5

        result = formatter.format(record)
        data = json.loads(result)

        assert data["method"] == "POST"
        assert data["path"] == "/v1/chat/completions"
        assert data["status"] == 200
        assert data["duration_ms"] == 150.5


class TestPrettyFormatter:
    """Tests for human-readable formatter."""

    def test_basic_format(self):
        """Test basic formatting."""
        formatter = PrettyFormatter()
        record = logging.LogRecord(
            name="hfl",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "Test message" in result

    def test_format_with_request_id(self):
        """Test formatting includes request ID."""
        set_request_id("abc123")

        formatter = PrettyFormatter()
        record = logging.LogRecord(
            name="hfl",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "[abc123]" in result

        # Clean up
        clear_request_id()


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_default(self):
        """Test default configuration."""
        configure_logging()

        logger = get_logger()
        assert logger.level == logging.INFO

    def test_configure_debug_level(self):
        """Test debug level configuration."""
        configure_logging(level="DEBUG")

        logger = get_logger()
        assert logger.level == logging.DEBUG

    def test_configure_json_format(self):
        """Test JSON format configuration."""
        configure_logging(json_format=True)

        logger = get_logger()
        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)


class TestLogFunctions:
    """Tests for convenience log functions."""

    def test_log_request(self):
        """Test log_request function doesn't raise."""
        configure_logging(level="INFO")
        # Just verify it doesn't raise
        log_request(
            method="GET",
            path="/v1/models",
            status=200,
            duration_ms=10.5,
        )

    def test_log_request_with_model(self):
        """Test log_request with model parameter."""
        configure_logging(level="INFO")
        log_request(
            method="POST",
            path="/v1/chat/completions",
            status=200,
            duration_ms=150.0,
            model="llama-7b",
        )

    def test_log_model_load(self):
        """Test log_model_load function doesn't raise."""
        configure_logging(level="INFO")
        log_model_load(model_name="llama-7b", duration_ms=5000.0)

    def test_log_error(self):
        """Test log_error function doesn't raise."""
        configure_logging(level="ERROR")
        log_error("Something went wrong")

    def test_log_error_with_exception(self):
        """Test log_error with exception."""
        configure_logging(level="ERROR")
        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_error("An error occurred", exc=e)
