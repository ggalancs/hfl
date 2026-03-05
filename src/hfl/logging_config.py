# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Centralized logging configuration for HFL.

Features:
- Structured JSON logging (optional)
- Request ID tracing
- Privacy-safe logging (no sensitive data)
- Configurable log levels
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any

# Context variable for request ID tracing
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str | None = None) -> str:
    """Set a request ID in context. Generates one if not provided."""
    if request_id is None:
        request_id = uuid.uuid4().hex[:8]
    request_id_var.set(request_id)
    return request_id


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields from the record
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "status"):
            log_data["status"] = record.status
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "model"):
            log_data["model"] = record.model

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Get request ID
        request_id = get_request_id()
        rid_str = f"[{request_id}] " if request_id else ""

        # Color based on level
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build message
        msg = record.getMessage()

        # Add extra context
        extras = []
        if hasattr(record, "method"):
            extras.append(f"{record.method} {getattr(record, 'path', '')}")
        if hasattr(record, "status"):
            extras.append(f"status={record.status}")
        if hasattr(record, "duration_ms"):
            extras.append(f"duration={record.duration_ms:.1f}ms")
        if hasattr(record, "model"):
            extras.append(f"model={record.model}")

        extra_str = " ".join(extras)
        if extra_str:
            msg = f"{msg} | {extra_str}"

        formatted = f"{timestamp} {color}{record.levelname:8}{reset} {rid_str}{msg}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON structured logging (for production)
        log_file: Optional file path for file logging
    """
    # Get root logger for hfl
    logger = logging.getLogger("hfl")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = PrettyFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False


def get_logger(name: str = "hfl") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to 'hfl')

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience functions for common log operations
def log_request(
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    model: str | None = None,
) -> None:
    """Log an API request with structured data.

    PRIVACY: This function only logs metadata, never request/response content.
    """
    logger = get_logger()
    extra: dict[str, Any] = {
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": duration_ms,
    }
    if model:
        extra["model"] = model

    logger.info(
        "Request completed",
        extra=extra,
    )


def log_model_load(model_name: str, duration_ms: float) -> None:
    """Log model loading event."""
    logger = get_logger()
    logger.info(
        f"Model loaded: {model_name}",
        extra={"model": model_name, "duration_ms": duration_ms},
    )


def log_error(message: str, exc: Exception | None = None) -> None:
    """Log an error with optional exception."""
    logger = get_logger()
    logger.error(message, exc_info=exc is not None)
