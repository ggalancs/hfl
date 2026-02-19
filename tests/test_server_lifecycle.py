# SPDX-License-Identifier: HRUL-1.0
"""Tests for server lifecycle and startup."""

from unittest.mock import MagicMock, patch

import pytest

from hfl.api.server import lifespan, start_server, state


class TestLifespan:
    """Tests for server lifespan context manager."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state.api_key = None
        state.engine = None
        state.current_model = None
        yield
        state.api_key = None
        state.engine = None
        state.current_model = None

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_with_loaded_engine(self):
        """Test that lifespan cleans up loaded engine on shutdown."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = True
        mock_engine.unload = MagicMock()

        state.engine = mock_engine

        # Run the lifespan context manager
        async with lifespan(None):
            pass  # Server runs here

        # After context exits, engine should be unloaded
        mock_engine.unload.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_no_engine(self):
        """Test lifespan with no engine loaded."""
        state.engine = None

        # Should not raise any errors
        async with lifespan(None):
            pass

    @pytest.mark.asyncio
    async def test_lifespan_engine_not_loaded(self):
        """Test lifespan with engine that is not loaded."""
        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        mock_engine.unload = MagicMock()

        state.engine = mock_engine

        async with lifespan(None):
            pass

        # Should not call unload if not loaded
        mock_engine.unload.assert_not_called()


class TestStartServer:
    """Tests for start_server function."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset server state before each test."""
        state.api_key = None
        yield
        state.api_key = None

    def test_start_server_sets_api_key(self):
        """Test that start_server sets the API key in state."""
        with patch("hfl.api.server.uvicorn.run") as mock_run:
            start_server(api_key="test-key")

            assert state.api_key == "test-key"
            mock_run.assert_called_once()

    def test_start_server_default_values(self):
        """Test start_server with default host and port."""
        with patch("hfl.api.server.uvicorn.run") as mock_run:
            with patch("hfl.api.server.config") as mock_config:
                mock_config.host = "127.0.0.1"
                mock_config.port = 11434

                start_server()

                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["host"] == "127.0.0.1"
                assert call_kwargs["port"] == 11434

    def test_start_server_custom_host_port(self):
        """Test start_server with custom host and port."""
        with patch("hfl.api.server.uvicorn.run") as mock_run:
            start_server(host="0.0.0.0", port=8080)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080

    def test_start_server_no_api_key(self):
        """Test start_server without API key."""
        with patch("hfl.api.server.uvicorn.run") as mock_run:
            start_server()

            assert state.api_key is None


class TestServerState:
    """Tests for ServerState class."""

    def test_server_state_attributes(self):
        """Test that ServerState has expected attributes."""
        from hfl.api.server import ServerState

        server_state = ServerState()

        assert hasattr(server_state, "engine")
        assert hasattr(server_state, "current_model")
        assert hasattr(server_state, "api_key")

    def test_server_state_initial_values(self):
        """Test initial values of ServerState."""
        from hfl.api.server import ServerState

        server_state = ServerState()

        assert server_state.engine is None
        assert server_state.current_model is None
        assert server_state.api_key is None
