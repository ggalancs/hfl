# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Rate limiting implementations for API server.

Provides both in-memory and distributed (SQLite-based) rate limiters.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Args:
            client_id: Unique identifier for the client (e.g., IP address)

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        ...

    @abstractmethod
    def reset(self, client_id: str | None = None) -> None:
        """Reset rate limit for a client or all clients.

        Args:
            client_id: Client to reset, or None to reset all
        """
        ...


class InMemoryRateLimiter(RateLimiter):
    """Simple in-memory rate limiter using sliding window.

    Thread-safe but not suitable for multi-process deployments.
    """

    def __init__(
        self,
        requests_per_window: int = 60,
        window_seconds: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Size of the sliding window in seconds
        """
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed (thread-safe)."""
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # Clean old entries
            self._counts[client_id] = [
                t for t in self._counts[client_id] if t > window_start
            ]

            current = len(self._counts[client_id])
            remaining = max(0, self._requests_per_window - current - 1)

            if current >= self._requests_per_window:
                return False, 0

            self._counts[client_id].append(now)
            return True, remaining

    def reset(self, client_id: str | None = None) -> None:
        """Reset rate limit."""
        with self._lock:
            if client_id is None:
                self._counts.clear()
            elif client_id in self._counts:
                del self._counts[client_id]

    @property
    def requests_per_window(self) -> int:
        """Maximum requests per window."""
        return self._requests_per_window

    @property
    def window_seconds(self) -> int:
        """Window size in seconds."""
        return self._window_seconds


class SQLiteRateLimiter(RateLimiter):
    """File-based rate limiter using SQLite.

    Suitable for multi-process deployments. Uses file locking
    provided by SQLite for coordination.
    """

    def __init__(
        self,
        db_path: Path,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        cleanup_interval: int = 300,  # Clean old entries every 5 min
    ):
        """Initialize SQLite rate limiter.

        Args:
            db_path: Path to SQLite database file
            requests_per_window: Maximum requests allowed per window
            window_seconds: Size of the sliding window in seconds
            cleanup_interval: How often to clean old entries (seconds)
        """
        self._db_path = db_path
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = 0.0
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_client_time
                ON rate_limits(client_id, timestamp)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper settings."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            isolation_level="IMMEDIATE",
        )
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    def _maybe_cleanup(self, conn: sqlite3.Connection) -> None:
        """Clean old entries if cleanup interval has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            window_start = now - self._window_seconds
            conn.execute(
                "DELETE FROM rate_limits WHERE timestamp < ?",
                (window_start,),
            )
            self._last_cleanup = now

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed (process-safe via SQLite)."""
        now = time.time()
        window_start = now - self._window_seconds

        with self._get_conn() as conn:
            # Periodic cleanup
            self._maybe_cleanup(conn)

            # Count current requests in window
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM rate_limits
                WHERE client_id = ? AND timestamp > ?
                """,
                (client_id, window_start),
            )
            current = cursor.fetchone()[0]

            if current >= self._requests_per_window:
                conn.commit()
                return False, 0

            # Record new request
            conn.execute(
                "INSERT INTO rate_limits (client_id, timestamp) VALUES (?, ?)",
                (client_id, now),
            )
            conn.commit()

            remaining = self._requests_per_window - current - 1
            return True, remaining

    def reset(self, client_id: str | None = None) -> None:
        """Reset rate limit."""
        with self._get_conn() as conn:
            if client_id is None:
                conn.execute("DELETE FROM rate_limits")
            else:
                conn.execute(
                    "DELETE FROM rate_limits WHERE client_id = ?",
                    (client_id,),
                )
            conn.commit()

    @property
    def requests_per_window(self) -> int:
        """Maximum requests per window."""
        return self._requests_per_window

    @property
    def window_seconds(self) -> int:
        """Window size in seconds."""
        return self._window_seconds


def create_rate_limiter(
    distributed: bool = False,
    db_path: Path | None = None,
    requests_per_window: int = 60,
    window_seconds: int = 60,
) -> RateLimiter:
    """Factory function to create appropriate rate limiter.

    Args:
        distributed: Use SQLite-based distributed rate limiter
        db_path: Path to SQLite database (required if distributed=True)
        requests_per_window: Maximum requests per window
        window_seconds: Size of the sliding window

    Returns:
        RateLimiter instance
    """
    if distributed:
        if db_path is None:
            from hfl.config import config

            db_path = config.cache_dir / "rate_limit.db"
        return SQLiteRateLimiter(
            db_path=db_path,
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
        )
    return InMemoryRateLimiter(
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )
