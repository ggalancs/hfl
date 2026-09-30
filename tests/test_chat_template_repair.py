# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Known mistakes in shipped chat templates are corrected before use.

Qwen2.5-Coder's template shows the tool call with doubled braces, the model
copies them, and its calls are not JSON (4 of 10 replies needed rescuing)."""

from __future__ import annotations

import pytest

from hfl.models.chat_template import repair_chat_template
from tests.chat_templates import DOUBLED_HINT, QWEN_CODER_TOOL_LINE, TOOL_CALL_HINT

# Rendering needs jinja2, which the lean CI venv does not install.
jinja2 = pytest.importorskip("jinja2")


def _render(template: str) -> str:
    return jinja2.Environment().from_string(template).render()


def test_as_shipped_the_model_is_shown_doubled_braces():
    assert DOUBLED_HINT in _render(QWEN_CODER_TOOL_LINE)


def test_corrected_it_is_shown_the_call_as_json():
    rendered = _render(repair_chat_template(QWEN_CODER_TOOL_LINE))
    assert TOOL_CALL_HINT in rendered and DOUBLED_HINT not in rendered


def test_a_correct_template_is_left_alone():
    fixed = repair_chat_template(QWEN_CODER_TOOL_LINE)
    assert repair_chat_template(fixed) == fixed
    assert repair_chat_template("{{ messages }}") == "{{ messages }}"
