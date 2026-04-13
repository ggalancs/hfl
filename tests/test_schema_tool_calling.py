# SPDX-License-Identifier: HRUL-1.0
"""Schema tests for Ollama-compatible tool-calling fields.

Covers the additions made to ``hfl.api.schemas.ollama`` so the wire contract
from ``hfl-tool-calling-spec.md`` §2 can be parsed end to end:

- ``ChatRequest.tools`` / ``tool_choice``
- ``OllamaChatMessage`` with ``role=tool`` + ``name``
- Assistant messages carrying ``tool_calls`` and (optional) empty content
- Canonical ``{"function": {"name", "arguments": dict}}`` tool-call shape
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hfl.api.schemas.ollama import (
    ChatRequest,
    OllamaChatMessage,
    OllamaTool,
    OllamaToolCall,
)


class TestChatRequestTools:
    def test_empty_tools_list_is_accepted(self):
        req = ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )
        assert req.tools == []

    def test_tool_list_parses(self):
        req = ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "write_wiki",
                        "description": "Create or overwrite a wiki article.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        )
        assert req.tools is not None and len(req.tools) == 1
        assert isinstance(req.tools[0], OllamaTool)
        assert req.tools[0].function.name == "write_wiki"
        assert req.tools[0].function.parameters["required"] == ["path", "content"]

    def test_tools_absent_is_allowed(self):
        req = ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.tools is None

    def test_tool_choice_string(self):
        req = ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice="auto",
        )
        assert req.tool_choice == "auto"

    def test_tool_choice_dict(self):
        req = ChatRequest(
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
            tool_choice={"type": "function", "function": {"name": "write_wiki"}},
        )
        assert isinstance(req.tool_choice, dict)


class TestToolRoleMessage:
    def test_tool_message_roundtrip(self):
        msg = OllamaChatMessage(role="tool", name="get_weather", content="22C sunny")
        assert msg.role == "tool"
        assert msg.name == "get_weather"
        assert msg.content == "22C sunny"

    def test_tool_message_requires_name(self):
        with pytest.raises(ValidationError):
            OllamaChatMessage(role="tool", content="result")

    def test_tool_message_requires_content(self):
        with pytest.raises(ValidationError):
            OllamaChatMessage(role="tool", name="get_weather")


class TestAssistantToolCalls:
    def test_assistant_with_tool_calls_and_empty_content(self):
        msg = OllamaChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "function": {
                        "name": "write_wiki",
                        "arguments": {
                            "path": "topics/hello.md",
                            "content": "Hello",
                        },
                    }
                }
            ],
        )
        assert msg.content == ""
        assert msg.tool_calls is not None
        tc = msg.tool_calls[0]
        assert isinstance(tc, OllamaToolCall)
        assert tc.function.name == "write_wiki"
        assert tc.function.arguments == {
            "path": "topics/hello.md",
            "content": "Hello",
        }

    def test_assistant_with_only_tool_calls_fills_content(self):
        msg = OllamaChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Madrid"},
                    }
                }
            ],
        )
        # Canonical wire shape: content defaults to "" when only tool_calls.
        assert msg.content == ""

    def test_assistant_without_content_or_tool_calls_is_invalid(self):
        with pytest.raises(ValidationError):
            OllamaChatMessage(role="assistant")

    def test_tool_call_arguments_must_be_object(self):
        with pytest.raises(ValidationError):
            OllamaChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "function": {
                            "name": "write_wiki",
                            "arguments": "not-a-dict",
                        }
                    }
                ],
            )


class TestMultiTurnRequest:
    def test_full_multi_turn_payload(self):
        """Mirrors T3 acceptance test shape from the spec."""
        req = ChatRequest(
            model="qwen3",
            messages=[
                {"role": "user", "content": "What is the weather in Madrid?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Madrid"},
                            }
                        }
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": "22C sunny"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )
        assert len(req.messages) == 3
        assert req.messages[1].tool_calls[0].function.arguments == {"city": "Madrid"}
        assert req.messages[2].role == "tool"
        assert req.messages[2].name == "get_weather"
