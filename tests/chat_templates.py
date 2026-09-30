# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Chat template text as models ship it, for tests."""

# Qwen2.5-Coder-1.5B-Instruct's tool-call line, as in its GGUF: doubled
# braces inside a Jinja string (backslashes are the template's own).
QWEN_CODER_TOOL_LINE = (
    r'{{- "<tool_call>\n'
    r"{{\"name\": <function-name>, \"arguments\": <args-json-object>}}"
    r'\n</tool_call>" }}'
)
# What the model should be shown instead: the call as JSON.
TOOL_CALL_HINT = '{"name": <function-name>, "arguments": <args-json-object>}'
DOUBLED_HINT = "{" + TOOL_CALL_HINT + "}"
