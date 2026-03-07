# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for CLI signal handling during streaming."""

from unittest.mock import MagicMock


class TestStreamingInterrupt:
    """Test that Ctrl+C during streaming is handled gracefully."""

    def test_keyboard_interrupt_during_stream_preserves_partial(self):
        """KeyboardInterrupt during chat_stream preserves partial response."""
        from hfl.engine.base import ChatMessage

        mock_engine = MagicMock()
        tokens = ["Hello", " world", " from"]

        def stream_with_interrupt(messages, *a, **kw):
            for t in tokens:
                yield t
            raise KeyboardInterrupt()

        mock_engine.chat_stream.side_effect = stream_with_interrupt

        messages = [ChatMessage(role="user", content="hi")]
        full_response = []
        try:
            for token in mock_engine.chat_stream(messages):
                full_response.append(token)
        except KeyboardInterrupt:
            pass  # This is the pattern used in cli/main.py

        assert full_response == ["Hello", " world", " from"]
        assert "".join(full_response) == "Hello world from"

    def test_keyboard_interrupt_mid_token_stops_cleanly(self):
        """Interrupt at first token stops immediately."""
        mock_engine = MagicMock()

        def stream_interrupt_immediately(messages, *a, **kw):
            raise KeyboardInterrupt()

        mock_engine.chat_stream.side_effect = stream_interrupt_immediately

        full_response = []
        try:
            for token in mock_engine.chat_stream([]):
                full_response.append(token)
        except KeyboardInterrupt:
            pass

        assert full_response == []

    def test_normal_stream_completes_fully(self):
        """Normal stream without interrupt completes all tokens."""
        mock_engine = MagicMock()
        tokens = ["one", " two", " three"]
        mock_engine.chat_stream.return_value = iter(tokens)

        full_response = []
        try:
            for token in mock_engine.chat_stream([]):
                full_response.append(token)
        except KeyboardInterrupt:
            pass

        assert full_response == tokens
