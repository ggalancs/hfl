# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A model's chat template, as ``show`` reports it.

``/api/show`` and ``hfl show --template`` returned only a Modelfile TEMPLATE,
so every pulled model showed an empty template (local audit A34/B25). The
model's own template lives in its files: a GGUF's ``tokenizer.chat_template``
header key, or a safetensors folder's ``chat_template.jinja`` /
``tokenizer_config.json``. Read without loading the model or any extra.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def model_template(manifest: Any) -> str:
    """The template requests to ``manifest`` are formatted with: a created
    model's TEMPLATE, else the one its files carry; "" when there is none."""
    own = getattr(manifest, "chat_template", None)
    if isinstance(own, str) and own:
        return own
    path = Path(str(getattr(manifest, "local_path", "") or ""))
    try:
        if path.is_file():
            from hfl.converter.gguf_header import read_fields

            found = read_fields(path, {"tokenizer.chat_template"})
            value = found.get("tokenizer.chat_template")
            return value if isinstance(value, str) else ""
        if path.is_dir():
            jinja = path / "chat_template.jinja"
            if jinja.is_file():
                return jinja.read_text(encoding="utf-8")
            config = path / "tokenizer_config.json"
            if config.is_file():
                value = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
                if isinstance(value, str):
                    return value
                if isinstance(value, list):  # named templates: the default one
                    named = {t.get("name"): t.get("template") for t in value if isinstance(t, dict)}
                    default = named.get("default")
                    return default if isinstance(default, str) else ""
    except (OSError, ValueError) as exc:
        logger.debug("no chat template readable from %s: %s", path, exc)
    return ""
