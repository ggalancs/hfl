# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Connection pooling for HuggingFace Hub API.

Provides a shared HTTP client with connection pooling for efficient
API requests to HuggingFace Hub.

Usage:
    from hfl.hub.client import get_hf_client, get_async_hf_client

    # Sync client
    client = get_hf_client()
    response = client.get("/api/models/...")

    # Async client
    async_client = get_async_hf_client()
    response = await async_client.get("/api/models/...")
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default configuration
HF_BASE_URL = "https://huggingface.co"
DEFAULT_TIMEOUT = 30.0
MAX_CONNECTIONS = 10
MAX_KEEPALIVE_CONNECTIONS = 5

# Singleton instances
_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None
_sync_lock = threading.Lock()


def get_hf_client(
    base_url: str = HF_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    token: str | None = None,
) -> httpx.Client:
    """Get the singleton sync HTTP client for HuggingFace API.

    Thread-safe singleton pattern. Creates a new client on first call,
    reuses it for subsequent calls.

    Args:
        base_url: HuggingFace API base URL
        timeout: Request timeout in seconds
        token: Optional HuggingFace token for authentication

    Returns:
        httpx.Client with connection pooling
    """
    global _sync_client

    if _sync_client is not None:
        return _sync_client

    with _sync_lock:
        if _sync_client is None:
            headers = {
                "User-Agent": "hfl/0.1.0",
            }
            if token:
                headers["Authorization"] = f"Bearer {token}"

            _sync_client = httpx.Client(
                base_url=base_url,
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=MAX_CONNECTIONS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                ),
                headers=headers,
                # HTTP/2 disabled by default - requires optional h2 package
            )
            logger.debug(f"Created sync HF client with base_url={base_url}")

        return _sync_client


async def get_async_hf_client(
    base_url: str = HF_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    token: str | None = None,
) -> httpx.AsyncClient:
    """Get the singleton async HTTP client for HuggingFace API.

    Creates a new client on first call, reuses it for subsequent calls.

    Args:
        base_url: HuggingFace API base URL
        timeout: Request timeout in seconds
        token: Optional HuggingFace token for authentication

    Returns:
        httpx.AsyncClient with connection pooling
    """
    global _async_client

    if _async_client is not None:
        return _async_client

    headers = {
        "User-Agent": "hfl/0.1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    _async_client = httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        ),
        headers=headers,
        # HTTP/2 disabled by default - requires optional h2 package
    )
    logger.debug(f"Created async HF client with base_url={base_url}")

    return _async_client


def close_hf_client() -> None:
    """Close the sync HTTP client.

    Should be called on application shutdown to release resources.
    """
    global _sync_client

    with _sync_lock:
        if _sync_client is not None:
            _sync_client.close()
            _sync_client = None
            logger.debug("Closed sync HF client")


async def close_async_hf_client() -> None:
    """Close the async HTTP client.

    Should be called on application shutdown to release resources.
    """
    global _async_client

    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
        logger.debug("Closed async HF client")


def reset_hf_clients() -> None:
    """Reset all HTTP clients (for testing).

    Closes existing clients and sets them to None.
    """
    global _sync_client, _async_client

    with _sync_lock:
        if _sync_client is not None:
            _sync_client.close()
            _sync_client = None

    # Note: async client should be closed with await
    # For testing, we just reset the reference
    _async_client = None
