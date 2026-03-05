# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for API helpers."""

import json

from hfl.api.helpers import (
    StreamingContext,
    format_ndjson_chunk,
    options_to_config,
    request_to_config,
)
from hfl.engine.base import GenerationConfig


class TestOptionsToConfig:
    """Tests for options_to_config function."""

    def test_empty_options(self):
        """Empty options should return default config."""
        config = options_to_config(None)
        assert isinstance(config, GenerationConfig)

        config = options_to_config({})
        assert isinstance(config, GenerationConfig)

    def test_maps_temperature(self):
        """Should map temperature option."""
        config = options_to_config({"temperature": 0.7})
        assert config.temperature == 0.7

    def test_maps_top_p(self):
        """Should map top_p option."""
        config = options_to_config({"top_p": 0.9})
        assert config.top_p == 0.9

    def test_maps_top_k(self):
        """Should map top_k option."""
        config = options_to_config({"top_k": 50})
        assert config.top_k == 50

    def test_maps_num_predict_to_max_tokens(self):
        """Should map num_predict to max_tokens."""
        config = options_to_config({"num_predict": 256})
        assert config.max_tokens == 256

    def test_maps_repeat_penalty(self):
        """Should map repeat_penalty."""
        config = options_to_config({"repeat_penalty": 1.1})
        assert config.repeat_penalty == 1.1

    def test_maps_stop(self):
        """Should map stop sequences."""
        config = options_to_config({"stop": ["END", "DONE"]})
        assert config.stop == ["END", "DONE"]

    def test_maps_seed(self):
        """Should map seed."""
        config = options_to_config({"seed": 42})
        assert config.seed == 42


class TestRequestToConfig:
    """Tests for request_to_config function."""

    def test_empty_request(self):
        """Empty request should return default config."""
        config = request_to_config()
        assert isinstance(config, GenerationConfig)

    def test_maps_basic_params(self):
        """Should map basic parameters."""
        config = request_to_config(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1000,
            seed=123,
        )
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.max_tokens == 1000
        assert config.seed == 123

    def test_stop_string_to_list(self):
        """Should convert stop string to list."""
        config = request_to_config(stop="STOP")
        assert config.stop == ["STOP"]

    def test_stop_list_preserved(self):
        """Should preserve stop list."""
        config = request_to_config(stop=["A", "B"])
        assert config.stop == ["A", "B"]


class TestStreamingContext:
    """Tests for StreamingContext class."""

    def test_creates_unique_id(self):
        """Should create unique request ID."""
        ctx1 = StreamingContext("model-a")
        ctx2 = StreamingContext("model-b")
        assert ctx1.request_id != ctx2.request_id
        assert ctx1.request_id.startswith("chatcmpl-")

    def test_format_chunk_with_content(self):
        """Should format chunk with content."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(content="Hello")

        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")

        data = json.loads(chunk.replace("data: ", "").strip())
        assert data["id"] == ctx.request_id
        assert data["model"] == "test-model"
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_format_chunk_finish(self):
        """Should format finish chunk."""
        ctx = StreamingContext("test-model")
        chunk = ctx.format_chunk(finish_reason="stop")

        data = json.loads(chunk.replace("data: ", "").strip())
        assert data["choices"][0]["delta"] == {}
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_format_done(self):
        """Should format done marker."""
        ctx = StreamingContext("test-model")
        done = ctx.format_done()
        assert done == "data: [DONE]\n\n"

    def test_timestamp_consistent(self):
        """Timestamp should be consistent across chunks."""
        ctx = StreamingContext("test-model")
        chunk1 = ctx.format_chunk(content="A")
        chunk2 = ctx.format_chunk(content="B")

        data1 = json.loads(chunk1.replace("data: ", "").strip())
        data2 = json.loads(chunk2.replace("data: ", "").strip())

        assert data1["created"] == data2["created"]


class TestFormatNdjsonChunk:
    """Tests for format_ndjson_chunk function."""

    def test_basic_chunk(self):
        """Should format basic NDJSON chunk."""
        chunk = format_ndjson_chunk("Hello", "test-model")

        data = json.loads(chunk)
        assert data["response"] == "Hello"
        assert data["model"] == "test-model"
        assert data["done"] is False
        assert "created_at" in data

    def test_done_chunk(self):
        """Should format done chunk."""
        chunk = format_ndjson_chunk("", "test-model", done=True)

        data = json.loads(chunk)
        assert data["done"] is True

    def test_extra_fields(self):
        """Should include extra fields."""
        chunk = format_ndjson_chunk(
            "content",
            "model",
            total_duration=1000,
            context=[1, 2, 3],
        )

        data = json.loads(chunk)
        assert data["total_duration"] == 1000
        assert data["context"] == [1, 2, 3]

    def test_ends_with_newline(self):
        """Chunk should end with newline."""
        chunk = format_ndjson_chunk("test", "model")
        assert chunk.endswith("\n")
