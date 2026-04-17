# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the reasoning/thinking extractor (Phase 5, P1-1).

Every model family has its own convention for reasoning markers;
``extract_thinking`` normalises them all down to ``(content,
thinking)``. This suite pins behaviour across the five dialects
HFL recognises.
"""

from __future__ import annotations

from hfl.api.thinking import extract_thinking


class TestNoThinking:
    def test_empty_string(self):
        assert extract_thinking("") == ("", None)

    def test_text_with_no_markers(self):
        assert extract_thinking("just an answer") == ("just an answer", None)


class TestXmlStyleThink:
    def test_deepseek_r1_style(self):
        raw = "<think>I should consider...</think>The answer is 42."
        content, thinking = extract_thinking(raw)
        assert content == "The answer is 42."
        assert thinking == "I should consider..."

    def test_multiple_think_blocks_concatenated(self):
        raw = "<think>first</think>middle<think>second</think>end"
        content, thinking = extract_thinking(raw)
        # Both reasoning blocks are preserved and concatenated with
        # a blank line between them.
        assert "first" in thinking
        assert "second" in thinking
        assert "middle" in content
        assert "end" in content

    def test_case_insensitive(self):
        raw = "<THINK>reasoning</THINK>answer"
        _, thinking = extract_thinking(raw)
        assert thinking == "reasoning"


class TestThinkingTag:
    def test_thinking_tag(self):
        raw = "<thinking>step by step</thinking>done"
        content, thinking = extract_thinking(raw)
        assert content == "done"
        assert thinking == "step by step"


class TestReasoningTag:
    def test_reasoning_tag(self):
        raw = "<reasoning>logic</reasoning>final"
        content, thinking = extract_thinking(raw)
        assert content == "final"
        assert thinking == "logic"


class TestGemma4Style:
    def test_split_pipe_thought_channel(self):
        raw = "<|channel>thoughtdeep analysis<channel|>the answer"
        content, thinking = extract_thinking(raw)
        assert content == "the answer"
        assert thinking == "deep analysis"

    def test_split_pipe_think_form(self):
        raw = "<|think>contemplating...<think|>42"
        content, thinking = extract_thinking(raw)
        assert content == "42"
        assert thinking == "contemplating..."

    def test_orphan_markers_stripped_but_content_kept(self):
        """When Gemma 4 emits channel markers without a paired
        thought block (e.g. ``<|channel>final``), the marker itself
        is removed but the following content stays."""
        raw = "<|channel>final\nthe real answer<turn|>"
        content, thinking = extract_thinking(raw)
        # No thought block → no thinking payload
        assert thinking is None
        # But the markers themselves are removed
        assert "<|channel>" not in content
        assert "<turn|>" not in content
        assert "the real answer" in content


class TestMixedDialects:
    def test_xml_plus_gemma4(self):
        """A model that happens to emit both conventions: both
        reasoning blocks are captured, both sets of markers stripped."""
        raw = "<think>xml-style</think><|channel>thoughtgemma-style<channel|>combined answer"
        content, thinking = extract_thinking(raw)
        assert content == "combined answer"
        assert "xml-style" in thinking
        assert "gemma-style" in thinking


class TestWhitespaceHandling:
    def test_empty_thinking_block_returns_none(self):
        """A thinking block with only whitespace doesn't produce
        a thinking payload."""
        raw = "<think>   </think>answer"
        _, thinking = extract_thinking(raw)
        assert thinking is None

    def test_leading_trailing_whitespace_trimmed(self):
        raw = "<think>  padded  </think>x"
        _, thinking = extract_thinking(raw)
        assert thinking == "padded"


class TestIdempotentOnCleanText:
    def test_clean_text_roundtrips_unchanged(self):
        """Text without reasoning markers comes back identical."""
        raw = "The meaning of life is 42. <b>html</b> passes through."
        content, thinking = extract_thinking(raw)
        assert content == raw
        assert thinking is None
