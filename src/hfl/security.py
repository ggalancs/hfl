# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Security utilities for HFL.

Provides:
- Path sanitization to prevent path traversal attacks
- Prompt sanitization to clean user input
- File checksum validation
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""


# =============================================================================
# Prompt Sanitization
# =============================================================================

# Control characters to remove (except common whitespace)
_CONTROL_CHARS_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# Excessive whitespace (3+ consecutive spaces/newlines)
_EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"[ ]{3,}")
_EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{4,}")

# Potential injection patterns (informational only)
_INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"(?i)disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"(?i)forget\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
    r"(?i)you\s+are\s+now\s+(a|an)\s+\w+\s+(named|called)",
    r"(?i)system:\s*",
    r"(?i)\[system\]",
    r"(?i)<\|system\|>",
    r"(?i)### system",
]


def sanitize_prompt(
    text: str,
    *,
    max_length: int | None = None,
    normalize_unicode: bool = True,
    remove_control_chars: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> str:
    """Sanitize prompt text for safe processing.

    Cleans user input by removing or normalizing potentially problematic
    characters while preserving the semantic content.

    Args:
        text: Raw prompt text to sanitize
        max_length: Maximum allowed length (truncates if exceeded)
        normalize_unicode: Normalize Unicode to NFC form
        remove_control_chars: Remove control characters
        collapse_whitespace: Collapse excessive whitespace
        strip: Strip leading/trailing whitespace

    Returns:
        Sanitized prompt text
    """
    if not text:
        return ""

    result = text

    # Normalize Unicode (NFC form - composed characters)
    if normalize_unicode:
        result = unicodedata.normalize("NFC", result)

    # Remove control characters (except tabs, newlines, spaces)
    if remove_control_chars:
        result = _CONTROL_CHARS_PATTERN.sub("", result)

    # Collapse excessive whitespace
    if collapse_whitespace:
        result = _EXCESSIVE_WHITESPACE_PATTERN.sub("  ", result)
        result = _EXCESSIVE_NEWLINES_PATTERN.sub("\n\n\n", result)

    # Strip leading/trailing whitespace
    if strip:
        result = result.strip()

    # Truncate if needed
    if max_length is not None and len(result) > max_length:
        result = result[:max_length]
        logger.debug("Prompt truncated from %s to %s chars", len(text), max_length)

    return result


def sanitize_messages(
    messages: list[dict],
    *,
    max_message_length: int | None = None,
    max_total_length: int | None = None,
) -> list[dict]:
    """Sanitize a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_message_length: Maximum length per message
        max_total_length: Maximum total content length

    Returns:
        Sanitized messages
    """
    sanitized = []
    total_length = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Sanitize content
        if isinstance(content, str):
            content = sanitize_prompt(content, max_length=max_message_length)
        else:
            content = ""

        # Sanitize role (only allow known roles)
        role = sanitize_role(role)

        # Track total length
        if max_total_length is not None:
            remaining = max_total_length - total_length
            if remaining <= 0:
                logger.debug("Max total length reached, truncating messages")
                break
            if len(content) > remaining:
                content = content[:remaining]
            total_length += len(content)

        sanitized.append({"role": role, "content": content})

    return sanitized


def sanitize_role(role: str) -> str:
    """Sanitize a message role.

    Args:
        role: Raw role string

    Returns:
        Sanitized role (one of: user, assistant, system)
    """
    if not isinstance(role, str):
        return "user"

    role = role.lower().strip()

    # Map to standard roles
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "ai", "bot", "model"):
        return "assistant"
    if role == "system":
        return "system"

    # Default to user for unknown roles
    return "user"


def detect_injection_attempt(text: str) -> list[str]:
    """Detect potential prompt injection patterns.

    This is informational only - it doesn't block the request,
    but logs and returns detected patterns for monitoring.

    Args:
        text: Text to analyze

    Returns:
        List of detected pattern descriptions (empty if none)
    """
    if not text:
        return []

    detected = []
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text):
            detected.append(pattern)

    if detected:
        logger.warning(
            f"Potential injection patterns detected: {len(detected)} matches"
        )

    return detected


def is_safe_filename(name: str) -> bool:
    """Check if a filename is safe (no path components).

    Args:
        name: Filename to check

    Returns:
        True if safe, False otherwise
    """
    if not name:
        return False

    # Check for path separators
    if "/" in name or "\\" in name:
        return False

    # Check for parent directory reference
    if ".." in name:
        return False

    # Check for starting with dot (hidden files)
    if name.startswith("."):
        return False

    # Check for control characters
    if _CONTROL_CHARS_PATTERN.search(name):
        return False

    return True


def sanitize_path(base_dir: Path, user_path: str) -> Path:
    """
    Sanitize a user-provided path to prevent path traversal attacks.

    Args:
        base_dir: The base directory that paths must stay within
        user_path: User-provided path (may be relative or absolute)

    Returns:
        Sanitized absolute path within base_dir

    Raises:
        PathTraversalError: If path would escape base_dir
    """
    # Normalize the base directory
    base_dir = base_dir.resolve()

    # Handle user path
    if Path(user_path).is_absolute():
        # For absolute paths, ensure they're within base_dir
        target = Path(user_path).resolve()
    else:
        # For relative paths, join with base and resolve
        target = (base_dir / user_path).resolve()

    # Verify the target is within base_dir
    try:
        target.relative_to(base_dir)
    except ValueError:
        raise PathTraversalError(f"Path '{user_path}' would escape base directory '{base_dir}'")

    return target


def sanitize_model_name(name: str) -> str:
    """
    Sanitize a model name to prevent path injection.

    Args:
        name: User-provided model name

    Returns:
        Sanitized model name safe for use in paths

    Raises:
        ValueError: If name contains invalid characters
    """
    # Remove any path separators
    sanitized = name.replace("/", "--").replace("\\", "--")

    # Remove any parent directory references
    sanitized = sanitized.replace("..", "__")

    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip().strip(".")

    if not sanitized:
        raise ValueError(f"Invalid model name: '{name}'")

    return sanitized


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute the hash of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex-encoded hash digest
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def verify_file_hash(file_path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify a file matches an expected hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected hash value (hex-encoded)
        algorithm: Hash algorithm used (default: sha256)

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_file_hash(file_path, algorithm)
    return actual_hash.lower() == expected_hash.lower()


# =============================================================================
# Audit Logging
# =============================================================================

from dataclasses import dataclass, field

audit_logger = logging.getLogger("hfl.audit")


@dataclass
class AuditEvent:
    """Structured audit event for compliance tracking."""

    event_type: str  # MODEL_ACCESS, MODEL_LOAD, MODEL_DELETE, AUTH_FAILURE, etc.
    action: str  # chat_completion, generate, pull, rm, etc.
    timestamp: str = ""
    client_ip: str = ""
    user_id: str | None = None
    model: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime, timezone

            self.timestamp = datetime.now(timezone.utc).isoformat()


def audit(event: AuditEvent) -> None:
    """Log an audit event for compliance tracking."""
    audit_logger.info(
        "%s: %s",
        event.event_type,
        event.action,
        extra={
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "user_id": event.user_id,
            "client_ip": event.client_ip,
            "model": event.model,
            "action": event.action,
            "details": event.details,
        },
    )
