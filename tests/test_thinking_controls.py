# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``/api/show`` reports a model's reasoning controls as Ollama 0.34.3 does
(``"thinking": {"values", "default"}``), found by rendering its template:
the default is what a request without ``think`` gets (plan 0.22 P1-10)."""

from __future__ import annotations

import pytest

pytest.importorskip("jinja2")

from hfl.models.manifest import ModelManifest  # noqa: E402
from hfl.models.thinking import thinking_controls  # noqa: E402

# Shaped like the real ones (Qwen3 / Gemma 4 / gpt-oss), cut to the part
# that decides.
QWEN3 = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)
GEMMA4 = (
    "{% if enable_thinking is defined and enable_thinking %}<|think|>\n{% endif %}"
    "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n"
    "{% if not enable_thinking | default(false) %}<|channel>thought\n<channel|>{% endif %}"
    "{% endif %}"
)
GPT_OSS = (
    "{% if not reasoning_effort is defined %}{% set reasoning_effort = 'medium' %}{% endif %}"
    "<|start|>system<|message|>Reasoning: {{ reasoning_effort }}<|end|>"
    "{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}"
)
DEEPSEEK_V31 = (
    "{% for m in messages %}<｜User｜>{{ m.content }}{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜>{% if thinking %}<think>{% else %}</think>"
    "{% endif %}{% endif %}"
)
PLAIN = "{% for m in messages %}{{ m.content }}{% endfor %}"


def _model(template: str, name: str = "m") -> ModelManifest:
    return ModelManifest(
        name=name,
        repo_id=f"org/{name}",
        local_path="/nowhere",
        format="gguf",
        chat_template=template,
    )


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (QWEN3, {"values": [False, True], "default": True}),
        (GEMMA4, {"values": [False, True], "default": False}),
        (GPT_OSS, {"values": ["low", "medium", "high"], "default": "medium"}),
        (DEEPSEEK_V31, {"values": [False, True], "default": False}),
    ],
    ids=["qwen3-on", "gemma4-off", "gpt-oss-levels", "deepseek-v3.1-off"],
)
def test_found_by_rendering_the_template(template, expected) -> None:
    assert thinking_controls(_model(template)) == expected


def test_a_model_that_always_reasons() -> None:
    assert thinking_controls(_model(PLAIN, "deepseek-r1-distill")) == {
        "values": [True],
        "default": True,
    }


def test_a_model_that_does_not_reason_has_none() -> None:
    assert thinking_controls(_model(PLAIN, "smollm2")) is None


def test_an_unrenderable_template_is_none_not_an_error() -> None:
    assert thinking_controls(_model("{% for %}", "smollm2")) is None


def test_api_show(temp_config) -> None:
    from fastapi.testclient import TestClient

    from hfl.api.server import app
    from hfl.models.registry import get_registry, reset_registry

    reset_registry()
    get_registry().add(_model(QWEN3, "q"))
    get_registry().add(_model(PLAIN, "plain"))
    client = TestClient(app)
    shown = client.post("/api/show", json={"model": "q"}).json()
    assert shown["thinking"] == {"values": [False, True], "default": True}
    assert "thinking" not in client.post("/api/show", json={"model": "plain"}).json()


def test_max_is_the_highest_level() -> None:
    from hfl.api.routes_native import _resolve_thinking_level

    assert _resolve_thinking_level("max") == "high"
