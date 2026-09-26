# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Rate limiting for the API server: one in-memory sliding window per client.

HFL is one process by design (see ``core/container.py``), so the limiter
lives in it; a SQLite-shared one and a per-model one were written and
tested but never used, and are gone.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass


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
