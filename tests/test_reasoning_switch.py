# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Each API's reasoning switch reaches ``GenerationConfig.reasoning``.

``think: false`` used to hide the reasoning only: the model still spent the
tokens (Qwen3-1.7B: 1500 either way). Now it reaches the template, and
the same switch works from OpenAI's ``reasoning_effort`` and Anthropic's
``thinking``. Measured after: 23 tokens instead of 1500.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.server import app
from hfl.api.state import get_state
from hfl.engine.base import GenerationResult, reasoning_template_vars


@pytest.fixture
def seen(temp_config):
    state = get_state()
    engine = MagicMock(is_loaded=True)
    configs: list = []

    def _chat(messages, config=None, tools=None):
        configs.append(config)
        return GenerationResult(text="391", tokens_generated=1, tokens_prompt=1)

    engine.chat = MagicMock(side_effect=_chat)
    manifest = MagicMock()
    manifest.name = "m"
    state.engine, state.current_model = engine, manifest
    yield configs
    state.engine = state.current_model = None


def _post(path, body):
    return TestClient(app).post(path, json={"model": "m", **body})


MSG = [{"role": "user", "content": "17*23?"}]


@pytest.mark.parametrize(
    ("think", "expected"), [(None, None), (False, "off"), (True, "medium"), ("high", "high")]
)
def test_ollama_think(seen, think, expected):
    body = {"messages": MSG, "stream": False}
    if think is not None:
        body["think"] = think
    assert _post("/api/chat", body).status_code == 200
    assert seen[-1].reasoning == expected


@pytest.mark.parametrize(
    ("effort", "expected"), [(None, None), ("none", "off"), ("minimal", "low"), ("high", "high")]
)
def test_openai_reasoning_effort(seen, effort, expected):
    body = {"messages": MSG}
    if effort is not None:
        body["reasoning_effort"] = effort
    assert _post("/v1/chat/completions", body).status_code == 200
    assert seen[-1].reasoning == expected


@pytest.mark.parametrize(
    ("thinking", "expected"),
    [
        (None, None),
        ({"type": "disabled"}, "off"),
        ({"type": "enabled", "budget_tokens": 2048}, "medium"),
    ],
)
def test_anthropic_thinking(seen, thinking, expected):
    body = {"messages": MSG, "max_tokens": 64}
    if thinking is not None:
        body["thinking"] = thinking
    assert _post("/v1/messages", body).status_code == 200
    assert seen[-1].reasoning == expected


def test_the_template_variables_each_family_reads():
    """Checked against the real templates of Qwen3, GLM-4.7, DeepSeek V3.1
    and gpt-oss (gpt-oss cannot reason less than "low")."""
    assert reasoning_template_vars(None) == {}
    assert reasoning_template_vars("off") == {
        "enable_thinking": False,
        "thinking": False,
        "reasoning_effort": "low",
    }
    assert reasoning_template_vars("high")["reasoning_effort"] == "high"
