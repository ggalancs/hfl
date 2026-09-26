# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A created model's Modelfile applies to every request (local audit A8/B8).

``create`` stored SYSTEM and PARAMETER and ``show`` displayed them, but no
inference path used them: the model answered with no system prompt and the
default sampling. These pin the helper and each API that goes through it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hfl.api.modelfile_defaults import (
    apply_parameters,
    apply_to_chat,
    default_system,
    default_template,
    explicit_fields,
)
from hfl.api.server import app
from hfl.api.state import get_state, reset_state
from hfl.engine.base import ChatMessage, GenerationConfig, GenerationResult
from hfl.models.manifest import ModelManifest

SYSTEM = "Answer only in French."


def _manifest(**extra) -> ModelManifest:
    return ModelManifest(
        name="french",
        repo_id="org/french",
        local_path="/tmp/fake.gguf",
        format="gguf",
        system=SYSTEM,
        default_parameters={"temperature": 0, "num_predict": 7, "stop": "<END>"},
        **extra,
    )


# -- the helper ----------------------------------------------------------------


def test_parameters_fill_what_the_request_left_unset() -> None:
    config = GenerationConfig()
    apply_parameters(_manifest(), config, set())
    assert (config.temperature, config.max_tokens, config.stop) == (0.0, 7, ["<END>"])


def test_an_option_the_request_set_wins() -> None:
    config = GenerationConfig(temperature=1.2)
    apply_parameters(_manifest(), config, {"temperature"})
    assert config.temperature == 1.2 and config.max_tokens == 7


def test_repeat_penalty_from_the_modelfile_counts_as_chosen() -> None:
    manifest = _manifest()
    manifest.default_parameters = {"repeat_penalty": 1.3}
    config = GenerationConfig()
    apply_parameters(manifest, config, set())
    assert config.repeat_penalty == 1.3 and config.repeat_penalty_chosen


def test_an_unusable_value_is_skipped_not_raised() -> None:
    manifest = _manifest()
    manifest.default_parameters = {"temperature": "hot", "top_k": 5}
    config = GenerationConfig()
    apply_parameters(manifest, config, set())
    assert config.temperature == GenerationConfig().temperature and config.top_k == 5


def test_system_only_when_the_request_has_none() -> None:
    out = apply_to_chat(
        _manifest(), [ChatMessage(role="user", content="hi")], GenerationConfig(), ()
    )
    assert [m.role for m in out] == ["system", "user"] and out[0].content == SYSTEM
    own = [ChatMessage(role="system", content="mine"), ChatMessage(role="user", content="hi")]
    assert apply_to_chat(_manifest(), own, GenerationConfig(), ())[0].content == "mine"


def test_messages_sit_between_system_and_conversation() -> None:
    manifest = _manifest(
        messages=[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    )
    out = apply_to_chat(
        manifest, [ChatMessage(role="user", content="live")], GenerationConfig(), ()
    )
    assert [m.content for m in out] == [SYSTEM, "Q", "A", "live"]


def test_a_manifest_like_mock_contributes_nothing() -> None:
    messages = [ChatMessage(role="user", content="hi")]
    config = GenerationConfig()
    assert apply_to_chat(MagicMock(), messages, config, ()) == messages
    assert config == GenerationConfig()


def test_completion_defaults() -> None:
    assert default_system(_manifest(), None) == SYSTEM
    assert default_system(_manifest(), "mine") == "mine"
    assert default_template(_manifest(chat_template="{{ .Prompt }}"), None) == "{{ .Prompt }}"
    assert default_template(_manifest(chat_template="{{ .Prompt }}"), "x") == "x"


def test_only_fields_the_client_sent_count() -> None:
    from hfl.api.schemas.openai import ChatCompletionRequest

    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], top_p=0.5)
    assert explicit_fields(req, {"temperature": "temperature", "top_p": "top_p"}) == {"top_p"}


# -- every API -----------------------------------------------------------------


@pytest.fixture
def engine(temp_config):
    reset_state()
    state = get_state()
    fake = MagicMock(is_loaded=True)
    result = GenerationResult(text="Bonjour", tokens_generated=1, tokens_prompt=1)
    fake.chat = MagicMock(return_value=result)
    fake.generate = MagicMock(return_value=result)
    state.engine = fake
    state.current_model = _manifest()
    yield fake
    reset_state()


def _chat_call(fake: MagicMock) -> tuple[list[ChatMessage], GenerationConfig]:
    args, kwargs = fake.chat.call_args
    messages = args[0] if args else kwargs["messages"]
    config = args[1] if len(args) > 1 else kwargs.get("config")
    return messages, config


USER = [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/chat", {"model": "french", "messages": USER, "stream": False}),
        ("/v1/chat/completions", {"model": "french", "messages": USER}),
        ("/v1/messages", {"model": "french", "messages": USER, "max_tokens": 50}),
        ("/v1/responses", {"model": "french", "input": "Hello"}),
    ],
)
def test_chat_apis_apply_system_and_parameters(engine, path, body) -> None:
    response = TestClient(app).post(path, json=body)
    assert response.status_code == 200, response.text
    messages, config = _chat_call(engine)
    assert messages[0].role == "system" and messages[0].content == SYSTEM
    assert config.temperature == 0.0 and config.stop == ["<END>"]


def test_a_request_option_overrides_the_modelfile(engine) -> None:
    body = {"model": "french", "messages": USER, "stream": False, "options": {"temperature": 0.9}}
    assert TestClient(app).post("/api/chat", json=body).status_code == 200
    _, config = _chat_call(engine)
    assert config.temperature == 0.9 and config.max_tokens == 7


def test_generate_applies_system_and_parameters(engine) -> None:
    body = {"model": "french", "prompt": "Hello", "stream": False}
    assert TestClient(app).post("/api/generate", json=body).status_code == 200
    args, kwargs = engine.generate.call_args
    prompt, config = args[0], args[1] if len(args) > 1 else kwargs.get("config")
    assert prompt.startswith(SYSTEM) and config.max_tokens == 7
