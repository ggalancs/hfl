# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A model's reasoning controls, as ``/api/show`` reports them.

Ollama (0.34.3) advertises ``"thinking": {"values": [...], "default": ...}``:
what ``think`` may be for this model, and what a request that leaves it out
gets. In HFL an omitted ``think`` sets no template variable, so the model's
own chat template decides (``hfl.engine.base.reasoning_template_vars``).
Both are therefore found by rendering that template — reading it was wrong:
Qwen3 thinks unless ``enable_thinking`` is false, Gemma 4 only when it is
true (``not enable_thinking | default(false)``), the same variable with
opposite defaults.

- ``reasoning_effort`` (gpt-oss): the levels; it cannot be switched off.
- ``enable_thinking`` (Qwen3, GLM, Gemma 4) or ``thinking`` (DeepSeek
  V3.1): ``[false, true]``.
- A thinking model whose template reads none of them (DeepSeek-R1,
  Qwen3-Thinking-2507) always reasons: ``[true]``.
"""

from __future__ import annotations

import functools
from typing import Any

from hfl.models.capabilities import detect_capabilities
from hfl.models.chat_template import model_template, template_env

LEVELS = ("low", "medium", "high")
_USER = [{"role": "user", "content": "hi"}]


@functools.lru_cache(maxsize=64)
def _from_template(template: str) -> dict[str, Any] | None:
    env = template_env()
    if env is None or not template:
        return None
    try:
        compiled = env.from_string(template)
    except Exception:
        return None

    def render(**variables: Any) -> str | None:
        try:
            return str(
                compiled.render(
                    messages=_USER,
                    add_generation_prompt=True,
                    bos_token="",
                    eos_token="",
                    **variables,
                )
            )
        except Exception:
            return None

    unset = render()
    if unset is None:
        return None
    levels = {level: render(reasoning_effort=level) for level in LEVELS}
    if len(set(levels.values())) == len(LEVELS):
        default = next((lvl for lvl, out in levels.items() if out == unset), None)
        if default is not None:
            return {"values": list(LEVELS), "default": default}
    for variable in ("enable_thinking", "thinking"):
        on, off = render(**{variable: True}), render(**{variable: False})
        if on is not None and off is not None and on != off and unset in (on, off):
            return {"values": [False, True], "default": unset == on}
    return None


def thinking_controls(manifest: Any) -> dict[str, Any] | None:
    """``{"values": [...], "default": ...}``, or None for a model that does
    not reason or whose template cannot be rendered here."""
    found = _from_template(model_template(manifest))
    if found is not None:
        return found
    try:
        reasons = "thinking" in detect_capabilities(manifest)
    except Exception:
        reasons = False
    return {"values": [True], "default": True} if reasons and template_env() else None
