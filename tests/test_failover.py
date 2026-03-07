# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the FailoverEngine."""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import pytest

from hfl.engine.base import (
    ChatMessage,
    GenerationConfig,
    GenerationResult,
    InferenceEngine,
)
from hfl.engine.failover import FailoverEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyEngine(InferenceEngine):
    """Minimal concrete engine for testing."""

    def __init__(self, name: str = "dummy", *, fail: bool = False) -> None:
        self._name = name
        self._loaded = False
        self._fail = fail

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        if self._fail:
            raise RuntimeError(f"{self._name} generate failed")
        return GenerationResult(text=f"{self._name}:gen:{prompt}")

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        if self._fail:
            raise RuntimeError(f"{self._name} generate_stream failed")
        yield f"{self._name}:stream1"
        yield f"{self._name}:stream2"

    def chat(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> GenerationResult:
        if self._fail:
            raise RuntimeError(f"{self._name} chat failed")
        return GenerationResult(text=f"{self._name}:chat")

    def chat_stream(self, messages: list[ChatMessage], config: GenerationConfig | None = None) -> Iterator[str]:
        if self._fail:
            raise RuntimeError(f"{self._name} chat_stream failed")
        yield f"{self._name}:cs1"
        yield f"{self._name}:cs2"

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFailoverEngineInit:
    def test_empty_engines_raises(self):
        with pytest.raises(ValueError, match="at least one engine"):
            FailoverEngine([])

    def test_single_engine(self):
        e = DummyEngine("only")
        fo = FailoverEngine([e])
        assert fo.model_name == "only"


class TestGenerate:
    def test_single_engine_no_failover(self):
        e = DummyEngine("a")
        fo = FailoverEngine([e])
        result = fo.generate("hello")
        assert result.text == "a:gen:hello"

    def test_failover_on_generate_error(self):
        e1 = DummyEngine("bad", fail=True)
        e2 = DummyEngine("good")
        fo = FailoverEngine([e1, e2])
        result = fo.generate("hello")
        assert result.text == "good:gen:hello"

    def test_all_engines_fail_raises_last_error(self):
        e1 = DummyEngine("bad1", fail=True)
        e2 = DummyEngine("bad2", fail=True)
        fo = FailoverEngine([e1, e2])
        with pytest.raises(RuntimeError, match="bad2 generate failed"):
            fo.generate("hello")


class TestChat:
    def test_failover_on_chat_error(self):
        e1 = DummyEngine("bad", fail=True)
        e2 = DummyEngine("good")
        fo = FailoverEngine([e1, e2])
        msgs = [ChatMessage(role="user", content="hi")]
        result = fo.chat(msgs)
        assert result.text == "good:chat"

    def test_all_chat_engines_fail(self):
        e1 = DummyEngine("b1", fail=True)
        e2 = DummyEngine("b2", fail=True)
        fo = FailoverEngine([e1, e2])
        msgs = [ChatMessage(role="user", content="hi")]
        with pytest.raises(RuntimeError, match="b2 chat failed"):
            fo.chat(msgs)


class TestGenerateStream:
    def test_single_engine_stream(self):
        e = DummyEngine("a")
        fo = FailoverEngine([e])
        tokens = list(fo.generate_stream("p"))
        assert tokens == ["a:stream1", "a:stream2"]

    def test_failover_on_stream_error(self):
        e1 = DummyEngine("bad", fail=True)
        e2 = DummyEngine("good")
        fo = FailoverEngine([e1, e2])
        tokens = list(fo.generate_stream("p"))
        assert tokens == ["good:stream1", "good:stream2"]

    def test_all_stream_engines_fail(self):
        e1 = DummyEngine("b1", fail=True)
        e2 = DummyEngine("b2", fail=True)
        fo = FailoverEngine([e1, e2])
        with pytest.raises(RuntimeError):
            list(fo.generate_stream("p"))


class TestChatStream:
    def test_failover_on_chat_stream_error(self):
        e1 = DummyEngine("bad", fail=True)
        e2 = DummyEngine("good")
        fo = FailoverEngine([e1, e2])
        msgs = [ChatMessage(role="user", content="hi")]
        tokens = list(fo.chat_stream(msgs))
        assert tokens == ["good:cs1", "good:cs2"]

    def test_all_chat_stream_engines_fail(self):
        e1 = DummyEngine("b1", fail=True)
        e2 = DummyEngine("b2", fail=True)
        fo = FailoverEngine([e1, e2])
        msgs = [ChatMessage(role="user", content="hi")]
        with pytest.raises(RuntimeError):
            list(fo.chat_stream(msgs))


class TestStickyRouting:
    def test_remembers_successful_engine(self):
        e1 = DummyEngine("a", fail=True)
        e2 = DummyEngine("b")
        e3 = DummyEngine("c")
        fo = FailoverEngine([e1, e2, e3])

        # First call fails over from a -> b
        result = fo.generate("x")
        assert result.text == "b:gen:x"

        # Second call should start at b (sticky), not a
        assert fo._get_current_index() == 1
        result2 = fo.generate("y")
        assert result2.text == "b:gen:y"

    def test_sticky_routing_after_chat(self):
        e1 = DummyEngine("a", fail=True)
        e2 = DummyEngine("b", fail=True)
        e3 = DummyEngine("c")
        fo = FailoverEngine([e1, e2, e3])

        msgs = [ChatMessage(role="user", content="hi")]
        fo.chat(msgs)
        assert fo._get_current_index() == 2

        # Next generate starts at engine c
        result = fo.generate("z")
        assert result.text == "c:gen:z"

    def test_sticky_routing_on_stream(self):
        e1 = DummyEngine("a", fail=True)
        e2 = DummyEngine("b")
        fo = FailoverEngine([e1, e2])
        list(fo.generate_stream("p"))
        assert fo._get_current_index() == 1


class TestLoadUnload:
    def test_load_propagates_to_all(self):
        e1 = DummyEngine("a")
        e2 = DummyEngine("b")
        fo = FailoverEngine([e1, e2])

        fo.load("/some/path")
        assert e1.is_loaded
        assert e2.is_loaded

    def test_unload_propagates_to_all(self):
        e1 = DummyEngine("a")
        e2 = DummyEngine("b")
        fo = FailoverEngine([e1, e2])

        fo.load("/some/path")
        fo.unload()
        assert not e1.is_loaded
        assert not e2.is_loaded

    def test_is_loaded_true_if_any_loaded(self):
        e1 = DummyEngine("a")
        e2 = DummyEngine("b")
        fo = FailoverEngine([e1, e2])

        assert not fo.is_loaded
        e1._loaded = True
        assert fo.is_loaded

    def test_is_loaded_false_when_none_loaded(self):
        e1 = DummyEngine("a")
        e2 = DummyEngine("b")
        fo = FailoverEngine([e1, e2])
        assert not fo.is_loaded


class TestModelName:
    def test_model_name_returns_current_engine_name(self):
        e1 = DummyEngine("alpha")
        e2 = DummyEngine("beta")
        fo = FailoverEngine([e1, e2])
        assert fo.model_name == "alpha"

    def test_model_name_follows_sticky_routing(self):
        e1 = DummyEngine("alpha", fail=True)
        e2 = DummyEngine("beta")
        fo = FailoverEngine([e1, e2])
        fo.generate("x")
        assert fo.model_name == "beta"


class TestThreeEngineFailover:
    def test_first_two_fail_third_succeeds(self):
        e1 = DummyEngine("a", fail=True)
        e2 = DummyEngine("b", fail=True)
        e3 = DummyEngine("c")
        fo = FailoverEngine([e1, e2, e3])
        result = fo.generate("q")
        assert result.text == "c:gen:q"
        assert fo._get_current_index() == 2
