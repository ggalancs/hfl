# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tool calls of gpt-oss (Harmony), DeepSeek, GLM and Hermes.

Every text below is either what the family's own chat template renders for a
call (the format the model was trained on) or what a real GGUF of it wrote
on HFL's default engine: unsloth/gpt-oss-20b, DeepSeek-R1-0528-Qwen3-8B,
THUDM GLM-4-9B-0414, unsloth GLM-4.7-Flash, Hermes-3-Llama-3.2-3B.
"""

from __future__ import annotations

import pytest

from hfl.api.thinking import extract_thinking
from hfl.api.tool_parsers import _detect_family, dispatch

WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]
PARIS = {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}


@pytest.mark.parametrize(
    ("model", "text", "content"),
    [
        # gpt-oss as it wrote it (reasoning first), and as its template
        # renders a call (recipient before the channel).
        (
            "gpt-oss-20b",
            "<|channel|>analysis<|message|>We need to use the get_weather function.<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city":"Paris"}',
            "",
        ),
        (
            "gpt-oss-20b",
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json"
            '<|message|>{"city": "Paris"}<|call|>',
            "",
        ),
        # DeepSeek V3.1's spelling, and V3 / R1's.
        (
            "deepseek-v3.1",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
            '{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            "",
        ),
        (
            "deepseek-r1",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            "",
        ),
        # DeepSeek-R1-0528 8B answering HFL's written-in tool prompt.
        (
            "deepseek-r1-0528-qwen3-8b",
            "<think>\nI should call get_weather.\n</think>\n"
            '```json\n{\n  "arguments": {\n    "city": "Paris"\n  },\n'
            '  "name": "get_weather"\n}\n```',
            "",
        ),
        # GLM-4.7-Flash: text before the call stays, the reasoning goes.
        (
            "glm-4.7-flash",
            "I have all the parameters.</think>I'll get the weather in Paris."
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            "I'll get the weather in Paris.",
        ),
        # GLM-4-0414: the name on its own line, then the JSON.
        ("glm-4-9b-0414", 'get_weather\n{"city": "Paris"}', ""),
        # Hermes-3 stops before the closing tag.
        (
            "hermes-3-llama-3.2-3b",
            '<tool_call>\n{ "name": "get_weather", "arguments": { "city": "Paris" }}\n',
            "",
        ),
    ],
)
def test_the_call_each_family_writes(model, text, content):
    assert dispatch(text, model, WEATHER) == (content, [PARIS])


def test_the_family_is_recognised_under_an_alias():
    """With tools declared every native marker is tried, whatever the name."""
    text = '<|channel|>commentary to=functions.get_weather <|message|>{"city": "Paris"}'
    assert dispatch(text, "my-assistant", WEATHER)[1] == [PARIS]


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("gpt-oss-20b", "harmony"),
        ("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek"),  # DeepSeek-templated
        ("deepseek-r1-0528-qwen3-8b", "deepseek"),
        ("GLM-4.7-Flash", "glm"),
        ("Hermes-3-Llama-3.1-8B", "qwen"),  # Hermes' <tool_call> is Qwen's
        ("qwen3-8b", "qwen"),
    ],
)
def test_family_from_the_name(name, family):
    assert _detect_family(name) == family


def test_glm_values_follow_the_schema():
    text = (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>2046</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
    )
    call = dispatch(text, "glm-4.5", WEATHER)[1][0]["function"]
    assert call["arguments"] == {"city": "2046", "days": 3}  # a string stays a string


def test_glm_values_without_a_schema_are_read_as_json():
    """The template writes non-string values as JSON."""
    text = (
        "<tool_call>get_weather<arg_key>hours</arg_key><arg_value>[9, 12]</arg_value>"
        "<arg_key>note</arg_key><arg_value>bring an umbrella</arg_value></tool_call>"
    )
    call = dispatch(text, "glm-4.5", WEATHER)[1][0]["function"]
    assert call["arguments"] == {"hours": [9, 12], "note": "bring an umbrella"}


def test_glm4_what_follows_a_call_is_not_the_model_s():
    """GLM-4-0414 went on past ``<|observation|>`` to invent the tool's
    result and answer from it; on llama-server the marker is not even in
    the text. Only the call counts."""
    invented = (
        'get_weather\n{"city": "Paris"}'
        'get_weather\n{"city": "Paris", "weather": "Sunny"}\nIt is sunny.'
    )
    assert dispatch(invented, "glm-4-9b-0414", WEATHER) == ("", [PARIS])
    observed = 'get_weather\n{"city": "Paris"}<|observation|>get_weather\n{"weather": "Sunny"}'
    assert dispatch(observed, "glm-4-9b-0414", WEATHER) == ("", [PARIS])


@pytest.mark.parametrize("text", ['result\n{"a": 1}', 'Here is the data\n{"a": 1}'])
def test_glm4_a_line_that_is_not_a_tool_is_text(text):
    assert dispatch(text, "glm-4-9b-0414", WEATHER) == (text, [])


def test_glm_after_observation_nothing_is_the_model_s():
    text = (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        "<|observation|><tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Rome</arg_value></tool_call>"
    )
    assert dispatch(text, "glm-4.5-air", WEATHER) == ("", [PARIS])


def test_gpt_oss_answer_is_the_final_channel():
    text = (
        "<|channel|>analysis<|message|>Just greet.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello there!"
    )
    assert dispatch(text, "gpt-oss-20b", WEATHER) == ("Hello there!", [])


def test_a_think_block_opened_by_the_template_is_dropped():
    """Templates that open ``<think>`` in the prompt leave only its close in
    the reply: what precedes it is reasoning."""
    text = "Let me think about Paris.</think>It is 31C."
    assert dispatch(text, "glm-4.7-flash", WEATHER) == ("It is 31C.", [])


def test_gpt_oss_reasoning_is_the_thinking():
    text = (
        "<|channel|>analysis<|message|>They want a greeting.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hi!<|return|>"
    )
    assert extract_thinking(text) == ("Hi!", "They want a greeting.")


STREAMED = [
    "<|channel|>analysis<|message|>They want hi.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Hey there buddy",
    "<think>\nLet me think.\n</think>\n\nHello!",
    "<think>a\n\nb</think>x\n\ny\n",
    "<|channel>thoughtHmm<channel|>Answer",
    "No markers at all < here, <b>bold</b>\n",
]


@pytest.mark.parametrize("text", STREAMED)
@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_a_stream_splits_as_a_finished_reply_does(text, size):
    from hfl.api.thinking import ThinkingSplitter

    splitter = ThinkingSplitter()
    answer = thinking = ""
    for i in range(0, len(text), size):
        a, t = splitter.feed(text[i : i + size])
        answer, thinking = answer + a, thinking + t
    a, t = splitter.flush()
    expected_answer, expected_thinking = extract_thinking(text)
    assert (answer + a, thinking + t) == (expected_answer, expected_thinking or "")
