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
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
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


@dataclass
class _ClientState:
    """Per-client rate limit state using deque for O(1) operations."""

    timestamps: deque[float]  # Bounded deque of request timestamps


class InMemoryRateLimiter(RateLimiter):
    """Simple in-memory rate limiter using sliding window with O(1) operations.

    Thread-safe but not suitable for multi-process deployments.
    Uses OrderedDict for O(1) LRU eviction to prevent unbounded memory growth.
    Uses deque for O(1) timestamp operations.
    """

    def __init__(
        self,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        max_clients: int = 10000,
        cleanup_interval: int = 60,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_window: Maximum requests allowed per window
            window_seconds: Size of the sliding window in seconds
            max_clients: Maximum number of tracked clients (LRU eviction)
            cleanup_interval: How often to clean stale entries (seconds)
        """
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._cleanup_interval = cleanup_interval
        # OrderedDict for O(1) LRU operations (move_to_end, popitem)
        self._clients: OrderedDict[str, _ClientState] = OrderedDict()
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def _maybe_cleanup(self, now: float) -> None:
        """Clean stale entries and enforce max_clients limit.

        Must be called with lock held.
        Uses O(1) operations per client removed.
        """
        # Only cleanup periodically
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        window_start = now - self._window_seconds

        # Remove stale entries (batch collect then delete)
        stale = [
            cid
            for cid, state in self._clients.items()
            if not state.timestamps or state.timestamps[-1] < window_start
        ]
        for cid in stale:
            del self._clients[cid]  # O(1) for OrderedDict

        # Enforce max_clients via LRU eviction - O(1) per eviction
        while len(self._clients) > self._max_clients:
            self._clients.popitem(last=False)  # Remove oldest (O(1))

    def _clean_old_timestamps(self, state: _ClientState, window_start: float) -> int:
        """Remove expired timestamps from client state.

        Uses deque.popleft() which is O(1) per removal.
        Returns count of valid timestamps.
        """
        # Remove expired timestamps from the front (O(1) each)
        while state.timestamps and state.timestamps[0] <= window_start:
            state.timestamps.popleft()
        return len(state.timestamps)

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed (thread-safe, O(1) amortized)."""
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # Periodic cleanup
            self._maybe_cleanup(now)

            # Get or create client state
            if client_id not in self._clients:
                # Use bounded deque to automatically limit memory
                self._clients[client_id] = _ClientState(
                    timestamps=deque(maxlen=self._requests_per_window + 1)
                )

            state = self._clients[client_id]

            # Clean old timestamps (O(k) where k is expired, amortized O(1))
            current = self._clean_old_timestamps(state, window_start)

            # Check rate limit
            if current >= self._requests_per_window:
                # Move to end for LRU tracking even on rate limit
                self._clients.move_to_end(client_id)  # O(1)
                return False, 0

            # Record request
            state.timestamps.append(now)  # O(1)
            self._clients.move_to_end(client_id)  # O(1) LRU update

            remaining = max(0, self._requests_per_window - current - 1)
            return True, remaining

    def reset(self, client_id: str | None = None) -> None:
        """Reset rate limit."""
        with self._lock:
            if client_id is None:
                self._clients.clear()
            elif client_id in self._clients:
                del self._clients[client_id]

    @property
    def requests_per_window(self) -> int:
        """Maximum requests per window."""
        return self._requests_per_window

    @property
    def window_seconds(self) -> int:
        """Window size in seconds."""
        return self._window_seconds

    @property
    def max_clients(self) -> int:
        """Maximum tracked clients."""
        return self._max_clients

    @property
    def client_count(self) -> int:
        """Current number of tracked clients."""
        with self._lock:
            return len(self._clients)


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


class PerModelRateLimiter:
    """Rate limiter with per-model limits and concurrency control.

    Combines global rate limiting with per-model limits to prevent
    GPU saturation from a single model.
    """

    def __init__(
        self,
        global_rpm: int = 60,
        per_model_rpm: int = 20,
        concurrent_per_model: int = 3,
        window_seconds: int = 60,
    ):
        self._global = InMemoryRateLimiter(global_rpm, window_seconds)
        self._per_model: dict[str, InMemoryRateLimiter] = {}
        self._concurrent: dict[str, threading.Semaphore] = {}
        self._per_model_rpm = per_model_rpm
        self._concurrent_limit = concurrent_per_model
        self._window_seconds = window_seconds
        self._lock = threading.Lock()

    def _get_model_limiter(self, model_name: str) -> InMemoryRateLimiter:
        with self._lock:
            if model_name not in self._per_model:
                self._per_model[model_name] = InMemoryRateLimiter(
                    self._per_model_rpm, self._window_seconds
                )
            return self._per_model[model_name]

    def _get_model_semaphore(self, model_name: str) -> threading.Semaphore:
        with self._lock:
            if model_name not in self._concurrent:
                self._concurrent[model_name] = threading.Semaphore(self._concurrent_limit)
            return self._concurrent[model_name]

    def is_allowed(self, client_id: str, model_name: str) -> tuple[bool, int]:
        """Check if request is allowed for client + model combination."""
        # Check global limit first
        global_allowed, global_remaining = self._global.is_allowed(client_id)
        if not global_allowed:
            return False, 0

        # Check per-model limit
        model_limiter = self._get_model_limiter(model_name)
        model_allowed, model_remaining = model_limiter.is_allowed(client_id)
        if not model_allowed:
            return False, 0

        return True, min(global_remaining, model_remaining)

    def acquire_concurrent(self, model_name: str) -> bool:
        """Try to acquire a concurrent request slot for a model."""
        sem = self._get_model_semaphore(model_name)
        return sem.acquire(blocking=False)

    def release_concurrent(self, model_name: str) -> None:
        """Release a concurrent request slot for a model."""
        sem = self._get_model_semaphore(model_name)
        sem.release()

    def reset(self, client_id: str | None = None) -> None:
        """Reset all rate limits."""
        self._global.reset(client_id)
        with self._lock:
            for limiter in self._per_model.values():
                limiter.reset(client_id)

    @property
    def global_limiter(self) -> InMemoryRateLimiter:
        return self._global


def create_rate_limiter(
    distributed: bool = False,
    db_path: Path | None = None,
    requests_per_window: int = 60,
    window_seconds: int = 60,
    *,
    per_model: bool = False,
    per_model_rpm: int = 20,
    concurrent_per_model: int = 3,
) -> RateLimiter | PerModelRateLimiter:
    """Factory function to create appropriate rate limiter.

    Args:
        distributed: Use SQLite-based distributed rate limiter
        db_path: Path to SQLite database (required if distributed=True)
        requests_per_window: Maximum requests per window
        window_seconds: Size of the sliding window
        per_model: Use per-model rate limiter with concurrency control
        per_model_rpm: Requests per minute per model (only if per_model=True)
        concurrent_per_model: Max concurrent requests per model (only if per_model=True)

    Returns:
        RateLimiter or PerModelRateLimiter instance
    """
    if per_model:
        return PerModelRateLimiter(
            global_rpm=requests_per_window,
            per_model_rpm=per_model_rpm,
            concurrent_per_model=concurrent_per_model,
            window_seconds=window_seconds,
        )
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
