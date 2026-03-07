# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for unified prompt building."""

from unittest.mock import MagicMock

import pytest

from hfl.engine.base import ChatMessage
from hfl.engine.prompt_builder import (
    PromptBuilder,
    PromptFormat,
    detect_format_from_tokenizer,
)


@pytest.fixture
def sample_messages():
    """Create sample chat messages for testing."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="How are you?"),
    ]


@pytest.fixture
def simple_messages():
    """Create simple messages without system."""
    return [
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]


class TestPromptFormat:
    """Tests for PromptFormat enum."""

    def test_format_values(self):
        """Should have expected format values."""
        assert PromptFormat.CHATML.value == "chatml"
        assert PromptFormat.LLAMA2.value == "llama2"
        assert PromptFormat.LLAMA3.value == "llama3"
        assert PromptFormat.ALPACA.value == "alpaca"
        assert PromptFormat.VICUNA.value == "vicuna"
        assert PromptFormat.GENERIC.value == "generic"


class TestMessagesToDicts:
    """Tests for messages_to_dicts method."""

    def test_converts_messages(self, sample_messages):
        """Should convert ChatMessage to dict."""
        result = PromptBuilder.messages_to_dicts(sample_messages)

        assert len(result) == 4
        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1] == {"role": "user", "content": "Hello!"}

    def test_empty_list(self):
        """Should handle empty list."""
        result = PromptBuilder.messages_to_dicts([])
        assert result == []


class TestBuildChatML:
    """Tests for ChatML format building."""

    def test_basic_format(self, simple_messages):
        """Should format messages in ChatML."""
        result = PromptBuilder.build_chatml(simple_messages)

        assert "<|im_start|>user" in result
        assert "Hello!" in result
        assert "<|im_end|>" in result
        assert "<|im_start|>assistant" in result

    def test_with_system(self, sample_messages):
        """Should include system message."""
        result = PromptBuilder.build_chatml(sample_messages)

        assert "<|im_start|>system" in result
        assert "helpful assistant" in result

    def test_generation_prompt(self, simple_messages):
        """Should add generation prompt by default."""
        result = PromptBuilder.build_chatml(simple_messages)
        assert result.endswith("<|im_start|>assistant\n")

    def test_no_generation_prompt(self, simple_messages):
        """Should skip generation prompt when disabled."""
        result = PromptBuilder.build_chatml(simple_messages, add_generation_prompt=False)
        assert not result.endswith("<|im_start|>assistant\n")


class TestBuildLlama2:
    """Tests for Llama 2 format building."""

    def test_basic_format(self, simple_messages):
        """Should format messages in Llama 2 style."""
        result = PromptBuilder.build_llama2(simple_messages)

        assert "[INST]" in result
        assert "[/INST]" in result
        assert "Hello!" in result

    def test_with_system(self, sample_messages):
        """Should include system in first instruction."""
        result = PromptBuilder.build_llama2(sample_messages)

        assert "<<SYS>>" in result
        assert "<</SYS>>" in result
        assert "helpful assistant" in result

    def test_system_only_in_first(self):
        """System should only appear in first instruction."""
        messages = [
            ChatMessage(role="system", content="Be helpful."),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
        ]
        result = PromptBuilder.build_llama2(messages)

        # System should appear once
        assert result.count("<<SYS>>") == 1


class TestBuildLlama3:
    """Tests for Llama 3 format building."""

    def test_basic_format(self, simple_messages):
        """Should format messages in Llama 3 style."""
        result = PromptBuilder.build_llama3(simple_messages)

        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|eot_id|>" in result

    def test_generation_prompt(self, simple_messages):
        """Should add assistant header for generation."""
        result = PromptBuilder.build_llama3(simple_messages)
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_no_generation_prompt(self, simple_messages):
        """Should not add assistant header when disabled."""
        result = PromptBuilder.build_llama3(simple_messages, add_generation_prompt=False)
        # Should end with eot_id from last message
        assert result.strip().endswith("<|eot_id|>")


class TestBuildAlpaca:
    """Tests for Alpaca format building."""

    def test_basic_format(self, simple_messages):
        """Should format messages in Alpaca style."""
        result = PromptBuilder.build_alpaca(simple_messages)

        assert "### Instruction:" in result
        assert "### Response:" in result

    def test_with_system(self, sample_messages):
        """Should include system section."""
        result = PromptBuilder.build_alpaca(sample_messages)
        assert "### System:" in result


class TestBuildVicuna:
    """Tests for Vicuna format building."""

    def test_basic_format(self, simple_messages):
        """Should format messages in Vicuna style."""
        result = PromptBuilder.build_vicuna(simple_messages)

        assert "USER:" in result
        assert "ASSISTANT:" in result

    def test_generation_prompt(self, simple_messages):
        """Should add ASSISTANT: for generation."""
        result = PromptBuilder.build_vicuna(simple_messages)
        assert result.strip().endswith("ASSISTANT:")


class TestBuildGeneric:
    """Tests for generic format building."""

    def test_basic_format(self, simple_messages):
        """Should format with role: content style."""
        result = PromptBuilder.build_generic(simple_messages)

        assert "User:" in result
        assert "Hello!" in result
        assert "Assistant:" in result

    def test_capitalizes_roles(self, simple_messages):
        """Should capitalize role names."""
        result = PromptBuilder.build_generic(simple_messages)
        assert "User:" in result
        assert "user:" not in result


class TestBuildDispatch:
    """Tests for build() dispatch method."""

    def test_dispatches_to_chatml(self, simple_messages):
        """Should dispatch to ChatML builder."""
        result = PromptBuilder.build(simple_messages, PromptFormat.CHATML)
        assert "<|im_start|>" in result

    def test_dispatches_to_llama2(self, simple_messages):
        """Should dispatch to Llama 2 builder."""
        result = PromptBuilder.build(simple_messages, PromptFormat.LLAMA2)
        assert "[INST]" in result

    def test_dispatches_to_llama3(self, simple_messages):
        """Should dispatch to Llama 3 builder."""
        result = PromptBuilder.build(simple_messages, PromptFormat.LLAMA3)
        assert "<|begin_of_text|>" in result

    def test_dispatches_to_alpaca(self, simple_messages):
        """Should dispatch to Alpaca builder."""
        result = PromptBuilder.build(simple_messages, PromptFormat.ALPACA)
        assert "### Instruction:" in result

    def test_dispatches_to_vicuna(self, simple_messages):
        """Should dispatch to Vicuna builder."""
        result = PromptBuilder.build(simple_messages, PromptFormat.VICUNA)
        assert "USER:" in result

    def test_defaults_to_generic(self, simple_messages):
        """Should default to generic format."""
        result = PromptBuilder.build(simple_messages)
        assert "User:" in result

    def test_passes_add_generation_prompt(self, simple_messages):
        """Should pass add_generation_prompt flag."""
        result = PromptBuilder.build(
            simple_messages,
            PromptFormat.CHATML,
            add_generation_prompt=False,
        )
        assert not result.endswith("<|im_start|>assistant\n")


class TestDetectFormatFromTokenizer:
    """Tests for tokenizer format detection."""

    def test_detects_chatml_from_template(self):
        """Should detect ChatML from chat_template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% if im_start %}"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.CHATML

    def test_detects_llama3_from_template(self):
        """Should detect Llama 3 from chat_template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% begin_of_text %}"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.LLAMA3

    def test_detects_llama3_from_header_id(self):
        """Should detect Llama 3 from start_header_id."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% start_header_id %}"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.LLAMA3

    def test_detects_llama2_from_template(self):
        """Should detect Llama 2 from [INST]."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% [INST] instruction [/inst] %}"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.LLAMA2

    def test_detects_llama3_from_name(self):
        """Should detect Llama 3 from model name."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "meta-llama/Llama-3-8B"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.LLAMA3

    def test_detects_llama2_from_name(self):
        """Should detect Llama 2 from model name."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "meta-llama/Llama-2-7b-chat-hf"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.LLAMA2

    def test_detects_vicuna_from_name(self):
        """Should detect Vicuna from model name."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "lmsys/vicuna-7b-v1.5"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.VICUNA

    def test_detects_alpaca_from_name(self):
        """Should detect Alpaca from model name."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "alpaca-7b-native"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.ALPACA

    def test_defaults_to_generic(self):
        """Should default to generic for unknown."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.name_or_path = "some-unknown-model"

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.GENERIC

    def test_handles_missing_attributes(self):
        """Should handle tokenizer without attributes."""
        tokenizer = MagicMock(spec=[])  # No chat_template or name_or_path

        result = detect_format_from_tokenizer(tokenizer)
        assert result == PromptFormat.GENERIC


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_messages(self):
        """Should handle empty message list."""
        result = PromptBuilder.build_chatml([])
        assert "<|im_start|>assistant\n" in result

    def test_single_user_message(self):
        """Should handle single user message."""
        messages = [ChatMessage(role="user", content="Hello")]
        result = PromptBuilder.build_chatml(messages)

        assert "Hello" in result
        assert "<|im_start|>assistant" in result

    def test_only_system_message(self):
        """Should handle only system message."""
        messages = [ChatMessage(role="system", content="Be helpful")]
        result = PromptBuilder.build_chatml(messages)

        assert "Be helpful" in result

    def test_multiline_content(self):
        """Should handle multiline content."""
        messages = [ChatMessage(role="user", content="Line 1\nLine 2\nLine 3")]
        result = PromptBuilder.build_chatml(messages)

        assert "Line 1\nLine 2\nLine 3" in result

    def test_special_characters_in_content(self):
        """Should preserve special characters."""
        messages = [ChatMessage(role="user", content="Hello <world> & 'friends'")]
        result = PromptBuilder.build_chatml(messages)

        assert "<world>" in result
        assert "'friends'" in result
