# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for consolidated API schemas."""

import pytest
from pydantic import ValidationError

from hfl.api.schemas import (
    AudioFormat,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatRequest,
    CompletionRequest,
    GenerateRequest,
    NativeTTSRequest,
    OpenAITTSRequest,
    TTSModelInfo,
)


class TestChatCompletionMessage:
    """Tests for ChatCompletionMessage schema."""

    def test_valid_message(self):
        """Should accept valid message."""
        msg = ChatCompletionMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_empty_role_rejected(self):
        """Should reject empty role."""
        with pytest.raises(ValidationError):
            ChatCompletionMessage(role="", content="Hello")

    def test_role_too_long_rejected(self):
        """Should reject role longer than 32 chars."""
        with pytest.raises(ValidationError):
            ChatCompletionMessage(role="a" * 33, content="Hello")

    def test_content_too_long_rejected(self):
        """Should reject content longer than 2M chars."""
        with pytest.raises(ValidationError):
            ChatCompletionMessage(role="user", content="x" * 2_000_001)


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatCompletionMessage(role="user", content="Hello")],
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.temperature == 0.7
        assert req.stream is False

    def test_empty_model_rejected(self):
        """Should reject empty model name."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="",
                messages=[ChatCompletionMessage(role="user", content="Hi")],
            )

    def test_empty_messages_rejected(self):
        """Should reject empty messages list."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="test", messages=[])

    def test_temperature_bounds(self):
        """Should enforce temperature bounds."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[ChatCompletionMessage(role="user", content="Hi")],
                temperature=2.5,
            )

    def test_total_content_validation(self):
        """Should validate total content across messages."""
        large_content = "x" * 1_500_000
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="test",
                messages=[
                    ChatCompletionMessage(role="user", content=large_content),
                    ChatCompletionMessage(role="assistant", content=large_content),
                ],
            )
        assert "2M characters" in str(exc_info.value)

    def test_optional_fields(self):
        """Should handle optional fields correctly."""
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatCompletionMessage(role="user", content="Hi")],
            max_tokens=100,
            stop=["END"],
            seed=42,
        )
        assert req.max_tokens == 100
        assert req.stop == ["END"]
        assert req.seed == 42


class TestCompletionRequest:
    """Tests for CompletionRequest schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = CompletionRequest(model="test", prompt="Hello, world!")
        assert req.model == "test"
        assert req.prompt == "Hello, world!"
        assert req.max_tokens == 256

    def test_prompt_list(self):
        """Should accept list of prompts."""
        req = CompletionRequest(model="test", prompt=["Hello", "World"])
        assert req.prompt == ["Hello", "World"]

    def test_prompt_too_long_rejected(self):
        """Should reject prompt longer than 2M chars."""
        with pytest.raises(ValidationError):
            CompletionRequest(model="test", prompt="x" * 2_000_001)


class TestChatCompletionContentValidation:
    """Coverage for ``ChatCompletionMessage._bound_content`` branches."""

    def test_empty_content_list_rejected(self):
        """A list with zero parts is rejected (would produce no
        text and no images — meaningless message)."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionMessage(role="user", content=[])
        assert "non-empty" in str(exc_info.value)

    def test_too_many_content_parts_rejected(self):
        """Cap at 64 parts to prevent boundless message bloat."""
        many_parts = [{"type": "text", "text": "x"}] * 65
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionMessage(role="user", content=many_parts)
        assert "64 parts" in str(exc_info.value)

    def test_total_text_across_parts_capped(self):
        """The 2M character cap applies across all text parts in a
        list; one giant text part is rejected the same way as a
        single string would be."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionMessage(
                role="user",
                content=[
                    {"type": "text", "text": "x" * 1_500_000},
                    {"type": "text", "text": "y" * 600_000},
                ],
            )
        assert "2_000_000" in str(exc_info.value)

    def test_content_string_too_long_rejected(self):
        """The string variant of content also enforces the 2M cap."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionMessage(role="user", content="x" * 2_000_001)
        assert "2_000_000" in str(exc_info.value)


class TestStopValidation:
    """Coverage for ``validate_stop`` branches on both
    ChatCompletionRequest and CompletionRequest."""

    def test_chat_stop_string_too_long_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                stop="x" * 257,
            )
        assert "256 characters" in str(exc_info.value)

    def test_chat_stop_too_many_entries_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                stop=[f"s{i}" for i in range(11)],
            )
        assert "10 sequences" in str(exc_info.value)

    def test_chat_stop_entry_too_long_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                stop=["ok", "x" * 257],
            )
        assert "256 characters" in str(exc_info.value)

    def test_completion_stop_string_too_long_rejected(self):
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", stop="x" * 257)

    def test_completion_stop_too_many_entries_rejected(self):
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", stop=[f"s{i}" for i in range(11)])

    def test_completion_stop_entry_too_long_rejected(self):
        with pytest.raises(ValidationError):
            CompletionRequest(model="m", prompt="hi", stop=["ok", "x" * 257])

    def test_chat_stop_none_passes_through(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            stop=None,
        )
        assert req.stop is None


class TestGenerateRequest:
    """Tests for GenerateRequest (Ollama-compatible) schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = GenerateRequest(model="llama3", prompt="Hello")
        assert req.model == "llama3"
        assert req.prompt == "Hello"
        assert req.stream is True  # Default

    def test_with_options(self):
        """Should accept options dict."""
        req = GenerateRequest(
            model="llama3",
            prompt="Hello",
            options={"temperature": 0.5, "top_p": 0.9},
        )
        assert req.options == {"temperature": 0.5, "top_p": 0.9}


class TestChatRequest:
    """Tests for ChatRequest (Ollama-compatible) schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = ChatRequest(
            model="llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.model == "llama3"
        assert len(req.messages) == 1
        assert req.stream is True  # Default

    def test_empty_messages_rejected(self):
        """Should reject empty messages list."""
        with pytest.raises(ValidationError):
            ChatRequest(model="llama3", messages=[])


class TestOpenAITTSRequest:
    """Tests for OpenAITTSRequest schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = OpenAITTSRequest(model="bark", input="Hello, world!")
        assert req.model == "bark"
        assert req.input == "Hello, world!"
        assert req.voice == "alloy"  # Default
        assert req.response_format == "mp3"  # Default
        assert req.speed == 1.0

    def test_speed_bounds(self):
        """Should enforce speed bounds."""
        with pytest.raises(ValidationError):
            OpenAITTSRequest(model="bark", input="Hi", speed=0.1)

        with pytest.raises(ValidationError):
            OpenAITTSRequest(model="bark", input="Hi", speed=5.0)

    def test_input_too_long(self):
        """Should reject input longer than 4096 chars."""
        with pytest.raises(ValidationError):
            OpenAITTSRequest(model="bark", input="x" * 4097)

    def test_valid_formats(self):
        """Should accept valid audio formats."""
        for fmt in ["mp3", "opus", "aac", "flac", "wav", "pcm"]:
            req = OpenAITTSRequest(model="bark", input="Hi", response_format=fmt)
            assert req.response_format == fmt


class TestNativeTTSRequest:
    """Tests for NativeTTSRequest schema."""

    def test_valid_request(self):
        """Should accept valid request."""
        req = NativeTTSRequest(model="coqui", text="Hello!")
        assert req.model == "coqui"
        assert req.text == "Hello!"
        assert req.voice == "default"
        assert req.language == "en"
        assert req.sample_rate == 22050
        assert req.format == "wav"
        assert req.stream is False

    def test_all_options(self):
        """Should accept all options."""
        req = NativeTTSRequest(
            model="bark",
            text="Hola mundo",
            voice="speaker_1",
            language="es",
            speed=1.5,
            sample_rate=44100,
            format="mp3",
            stream=True,
        )
        assert req.language == "es"
        assert req.speed == 1.5
        assert req.sample_rate == 44100
        assert req.format == "mp3"
        assert req.stream is True


class TestTTSModelInfo:
    """Tests for TTSModelInfo schema."""

    def test_valid_model_info(self):
        """Should create valid model info."""
        info = TTSModelInfo(id="bark-small", owned_by="suno-ai")
        assert info.id == "bark-small"
        assert info.object == "model"
        assert info.owned_by == "suno-ai"
        assert info.capabilities == {}

    def test_with_capabilities(self):
        """Should accept capabilities dict."""
        info = TTSModelInfo(
            id="coqui-tts",
            owned_by="coqui",
            capabilities={"tts": True, "voices": True, "languages": ["en", "es"]},
        )
        assert info.capabilities["tts"] is True
        assert "en" in info.capabilities["languages"]


class TestAudioFormat:
    """Tests for AudioFormat enum."""

    def test_all_formats(self):
        """Should have all expected formats."""
        assert AudioFormat.WAV.value == "wav"
        assert AudioFormat.MP3.value == "mp3"
        assert AudioFormat.OGG.value == "ogg"
        assert AudioFormat.OPUS.value == "opus"
        assert AudioFormat.AAC.value == "aac"
        assert AudioFormat.FLAC.value == "flac"
        assert AudioFormat.PCM.value == "pcm"

    def test_string_conversion(self):
        """Should convert to string properly."""
        assert str(AudioFormat.WAV) == "AudioFormat.WAV"
        assert AudioFormat.MP3 == "mp3"  # Enum comparison


class TestSchemaExports:
    """Tests for schema module exports."""

    def test_all_exports_available(self):
        """Should export all schemas from __init__."""
        from hfl.api import schemas

        assert hasattr(schemas, "ChatCompletionMessage")
        assert hasattr(schemas, "ChatCompletionRequest")
        assert hasattr(schemas, "CompletionRequest")
        assert hasattr(schemas, "GenerateRequest")
        assert hasattr(schemas, "ChatRequest")
        assert hasattr(schemas, "OpenAITTSRequest")
        assert hasattr(schemas, "NativeTTSRequest")
        assert hasattr(schemas, "TTSModelInfo")
        assert hasattr(schemas, "AudioFormat")
