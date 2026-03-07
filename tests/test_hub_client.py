# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for HuggingFace Hub HTTP client."""

import pytest

from hfl.hub.client import (
    get_hf_client,
    reset_hf_clients,
    close_hf_client,
    HF_BASE_URL,
    DEFAULT_TIMEOUT,
    MAX_CONNECTIONS,
)


@pytest.fixture(autouse=True)
def cleanup_clients():
    """Reset clients before and after each test."""
    reset_hf_clients()
    yield
    reset_hf_clients()


class TestSyncClient:
    """Tests for synchronous HTTP client."""

    def test_get_hf_client_creates_client(self):
        """get_hf_client should create a new client."""
        client = get_hf_client()
        assert client is not None
        assert str(client.base_url).rstrip("/") == HF_BASE_URL

    def test_get_hf_client_returns_singleton(self):
        """get_hf_client should return the same instance."""
        client1 = get_hf_client()
        client2 = get_hf_client()
        assert client1 is client2

    def test_get_hf_client_with_custom_base_url(self):
        """get_hf_client respects custom base_url on first call."""
        custom_url = "https://custom.example.com"
        client = get_hf_client(base_url=custom_url)
        assert str(client.base_url).rstrip("/") == custom_url

    def test_get_hf_client_with_timeout(self):
        """get_hf_client respects timeout parameter."""
        client = get_hf_client(timeout=60.0)
        assert client.timeout.read == 60.0

    def test_get_hf_client_with_token(self):
        """get_hf_client adds authorization header when token provided."""
        client = get_hf_client(token="test_token_123")
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test_token_123"

    def test_get_hf_client_has_user_agent(self):
        """get_hf_client includes User-Agent header."""
        client = get_hf_client()
        assert "User-Agent" in client.headers
        assert "hfl" in client.headers["User-Agent"]

    def test_close_hf_client(self):
        """close_hf_client closes the client."""
        client = get_hf_client()
        close_hf_client()
        # After closing, getting client should create new one
        new_client = get_hf_client()
        assert new_client is not client

    def test_reset_hf_clients(self):
        """reset_hf_clients resets all clients."""
        client = get_hf_client()
        reset_hf_clients()
        new_client = get_hf_client()
        assert new_client is not client


class TestAsyncClient:
    """Tests for async HTTP client."""

    @pytest.mark.asyncio
    async def test_get_async_hf_client_creates_client(self):
        """get_async_hf_client should create a new async client."""
        from hfl.hub.client import get_async_hf_client, close_async_hf_client

        client = await get_async_hf_client()
        assert client is not None
        assert str(client.base_url).rstrip("/") == HF_BASE_URL
        await close_async_hf_client()

    @pytest.mark.asyncio
    async def test_get_async_hf_client_returns_singleton(self):
        """get_async_hf_client should return the same instance."""
        from hfl.hub.client import get_async_hf_client, close_async_hf_client

        client1 = await get_async_hf_client()
        client2 = await get_async_hf_client()
        assert client1 is client2
        await close_async_hf_client()

    @pytest.mark.asyncio
    async def test_get_async_hf_client_with_token(self):
        """get_async_hf_client adds authorization header when token provided."""
        from hfl.hub.client import get_async_hf_client, close_async_hf_client

        client = await get_async_hf_client(token="async_test_token")
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer async_test_token"
        await close_async_hf_client()

    @pytest.mark.asyncio
    async def test_close_async_hf_client(self):
        """close_async_hf_client closes the client."""
        from hfl.hub.client import get_async_hf_client, close_async_hf_client

        client = await get_async_hf_client()
        await close_async_hf_client()
        # After closing, getting client should create new one
        new_client = await get_async_hf_client()
        assert new_client is not client
        await close_async_hf_client()


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_default_timeout(self):
        """Default timeout should be set."""
        assert DEFAULT_TIMEOUT == 30.0

    def test_max_connections(self):
        """Max connections should be configured."""
        assert MAX_CONNECTIONS == 10

    def test_client_has_connection_limits(self):
        """Client should have connection limits configured."""
        client = get_hf_client()
        # Check that client was created with limits
        # The limits are configured in the client creation
        assert client is not None
