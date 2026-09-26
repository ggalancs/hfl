# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Per-model-family tool-call parsers for Ollama-compatible output.

The Ollama wire protocol requires that the server return structured
``tool_calls`` on the assistant message. When an inference backend does not
do that itself (e.g. llama-cpp-python older than 0.3.0, or any backend
wired through raw completion APIs), HFL must parse the model's native
tool-call markers out of the generated text.

This module implements the mapping described in ``hfl-tool-calling-spec.md``
§4:

- **Qwen 2.5 / Qwen 3**: ``<tool_call>{json}</tool_call>``
- **Llama 3.x**: ``<|python_tag|>{json}<|eom_id|>`` or
  ``<function=name>{json}</function>``
- **Mistral / Mixtral**: ``[TOOL_CALLS][{json array}]``
- **gpt-oss** (Harmony): ``<|channel|>commentary to=functions.NAME
  <|constrain|>json<|message|>{json}<|call|>``
- **DeepSeek**: ``<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{json}<｜tool▁call▁end｜>``
  (V3.1) or ``...begin｜>function<｜tool▁sep｜>NAME\n```json {json}```...``
  (V3 / R1)
- **GLM**: ``<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>
  </tool_call>`` (4.5+) or ``NAME\n{json}`` (GLM-4-0414)
- **Hermes**: Qwen's ``<tool_call>{json}</tool_call>``
- **Gemma 4**: split-pipe DSL
  ``<|tool_call>call:NAME{key:<|"|>val<|"|>,k2:42}<tool_call|>``
  (not JSON — keys are bare, strings wrapped in Gemma 4's dedicated
  ``<|"|>`` delimiter token).
- **Fallback**: bare ``{"name": ..., "arguments": {...}}`` or
  ``{"tool_call": {"name": ..., "args|arguments": {...}}}`` envelopes.

Every parser returns a ``(clean_content, tool_calls)`` tuple where
``clean_content`` is the original text with all recognized tool-call
markers stripped out, and ``tool_calls`` is a list of canonical dicts of
shape::

    {"function": {"name": str, "arguments": dict}}

``arguments`` is always a parsed JSON object (never a string) — this is
rule C3 of the spec.
"""

from __future__ import annotations

import json
import re
from typing import Any

ToolCall = dict[str, Any]
ParseResult = tuple[str, list[ToolCall]]


# --- Canonical shape helpers --------------------------------------------------


def _wrap(name: str, arguments: Any) -> ToolCall:
    """Build a canonical ``{"function": {"name", "arguments"}}`` entry.

    ``arguments`` is coerced to a dict: strings are JSON-parsed; anything
    else becomes ``{}``. This enforces rule C3 (arguments is always an
    object, never a string).
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"function": {"name": str(name), "arguments": arguments}}


def _safe_json_load(payload: str) -> Any:
    """Attempt a json.loads, returning ``None`` on failure."""
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return None


# --- Qwen 2.5 / Qwen 3 --------------------------------------------------------


# The closing tag is optional at the very end: Hermes-3 stops right after
# the JSON (its end-of-turn token comes first).
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)",
    re.DOTALL,
)


_DANGLING_CALL_RE = re.compile(r"\s*<tool_call>\s*$")


def parse_qwen(text: str, tools: list[dict] | None = None) -> ParseResult:
    """Parse qwen-family ``<tool_call>...</tool_call>`` markers.

    Both forms: the JSON one (Qwen 2.5 / Qwen 3) and Qwen3-Coder's XML
    ``<function=NAME><parameter=KEY>value</parameter></function>``, whose
    values are typed from the tool's JSON schema in ``tools``.
    """
    calls: list[ToolCall] = []

    def _sub(match: re.Match) -> str:
        raw = match.group(1)
        payload = _safe_json_load(raw)
        if payload is None and raw.startswith("{{") and raw.endswith("}}"):
            # Qwen2.5-7B's GGUF template shows the call format with doubled
            # braces, and the model copies them (found by the matrix).
            payload = _safe_json_load(raw[1:-1])
        if isinstance(payload, dict) and "name" in payload:
            calls.append(
                _wrap(
                    payload["name"],
                    payload.get("arguments", payload.get("parameters", {})),
                )
            )
        return ""

    def _sub_xml(match: re.Match[str]) -> str:
        name, body = match.group(1), match.group(2)
        if body.lstrip().startswith("{"):
            return match.group(0)  # Llama 3's ``<function=name>{json}``
        schema = _parameter_schemas(tools, name)
        arguments = {
            key: _typed_value(_strip_template_newlines(raw), schema.get(key))
            for key, raw in _QWEN_XML_PARAM_RE.findall(body)
        }
        calls.append(_wrap(name, arguments))
        return ""

    cleaned = _QWEN_TOOL_CALL_RE.sub(_sub, text)
    cleaned = _QWEN_XML_FUNCTION_RE.sub(_sub_xml, cleaned)
    # A call opened and never written: Qwen3-Coder ended its last answer to
    # Claude Code and Codex with a bare ``<tool_call>`` (measured). Nothing
    # follows it, so it is no text of the answer's.
    cleaned = _DANGLING_CALL_RE.sub("", cleaned)
    return _strip_thinking(cleaned).strip(), calls


_QWEN_XML_FUNCTION_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>(?:\s*</tool_call>)?",
    re.DOTALL,
)
# A value ends at its closing tag, or — when the model forgets it — at the
# next parameter or the end of the function body.
_QWEN_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|\Z)",
    re.DOTALL,
)


def _strip_template_newlines(value: str) -> str:
    """Drop the one newline the template puts on each side of a value, and
    nothing else: code in an ``Edit`` must match the file byte for byte."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _parameter_schemas(tools: list[dict] | None, name: str) -> dict[str, Any]:
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters") or fn.get("input_schema") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            return props if isinstance(props, dict) else {}
    return {}


def _typed_value(value: str, schema: Any) -> Any:
    """``value`` as the type its schema declares; the text itself when there
    is no schema, the type is ``string``, or the value does not fit."""
    declared = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(declared, list):
        declared = next((t for t in declared if t != "null"), None)
    text = value.strip()
    if declared == "integer" and re.fullmatch(r"-?\d+", text):
        return int(text)
    if declared == "number":
        number = _safe_json_load(text)
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            return number
    if declared == "boolean" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if declared == "array":
        parsed = _safe_json_load(text)
        if isinstance(parsed, list):
            return parsed
    if declared == "object":
        parsed = _safe_json_load(text)
        if isinstance(parsed, dict):
            return parsed
    return value


# --- Llama 3.x ----------------------------------------------------------------


_LLAMA3_PYTHON_TAG_RE = re.compile(
    r"<\|python_tag\|>\s*(\{.*?\})(?:\s*<\|eom_id\|>|\s*$)",
    re.DOTALL,
)

_LLAMA3_FUNCTION_RE = re.compile(
    r"<function=([\w.\-]+)>\s*(\{.*?\})\s*</function>",
    re.DOTALL,
)


def parse_llama3(text: str) -> ParseResult:
    """Parse llama3 native tool-call markers.

    Llama 3 uses ``parameters`` rather than ``arguments`` — we normalise
    that to ``arguments`` so downstream code always sees the canonical
    field.
    """
    calls: list[ToolCall] = []

    def _sub_python_tag(match: re.Match) -> str:
        payload = _safe_json_load(match.group(1))
        if isinstance(payload, dict) and "name" in payload:
            args = payload.get("arguments", payload.get("parameters", {}))
            calls.append(_wrap(payload["name"], args))
        return ""

    cleaned = _LLAMA3_PYTHON_TAG_RE.sub(_sub_python_tag, text)

    def _sub_function(match: re.Match) -> str:
        name = match.group(1)
        args = _safe_json_load(match.group(2)) or {}
        calls.append(_wrap(name, args))
        return ""

    cleaned = _LLAMA3_FUNCTION_RE.sub(_sub_function, cleaned)
    return cleaned.strip(), calls


# --- Mistral / Mixtral --------------------------------------------------------


_MISTRAL_TOOL_CALLS_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*?\])",
    re.DOTALL,
)


def parse_mistral(text: str) -> ParseResult:
    """Parse mistral's ``[TOOL_CALLS][...json array...]`` envelope."""
    calls: list[ToolCall] = []

    def _sub(match: re.Match) -> str:
        payload = _safe_json_load(match.group(1))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "name" in item:
                    args = item.get("arguments", item.get("parameters", {}))
                    calls.append(_wrap(item["name"], args))
        return ""

    cleaned = _MISTRAL_TOOL_CALLS_RE.sub(_sub, text)
    return cleaned.strip(), calls


# --- Gemma 4 ------------------------------------------------------------------


# Gemma 4's dedicated string-delimiter token (ID 110 in the vocab).
# Opens AND closes string values in the argument DSL — the same token
# is used on both ends, so it's not a balanced pair like regular
# quotes.
_GEMMA4_STR_DELIM = '<|"|>'

# Tool-call envelope. Body is matched non-greedily so that the
# ``}<tool_call|>`` / ``}$`` anchor forces the minimal balanced
# ``{...}``. The closing marker is optional because callers may set
# ``stop=["<tool_call|>"]`` on the underlying ``create_chat_completion``
# call, which consumes the stop string before it reaches the output.
_GEMMA4_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([\w.\-]+)\{(.*?)\}(?:<tool_call\|>|(?=<\|)|$)",
    re.DOTALL,
)


def _gemma4_dsl_to_dict(body: str) -> dict:
    """Convert a Gemma 4 argument DSL body into a Python dict.

    The body syntax is::

        key:<|"|>string<|"|>,key2:42,key3:true,key4:{nested:<|"|>v<|"|>}

    Strings are delimited by the ``<|"|>`` token on *both* sides, keys
    are bare identifiers, numbers / booleans / null are bare, and
    nested objects use ``{...}``. We transform the DSL to equivalent
    JSON and delegate to :func:`json.loads` for robustness.

    Returns ``{}`` on any decoding error so a malformed payload does
    not drop the surrounding tool call itself — the caller can still
    see which function was invoked and respond appropriately.
    """
    if not body.strip():
        return {}

    # Split alternating outside / inside string-delimiter segments.
    # Well-formed input always has an odd number of parts (each
    # string opens and closes with the same delimiter). An even count
    # means an unclosed string → malformed.
    parts = body.split(_GEMMA4_STR_DELIM)
    if len(parts) % 2 == 0:
        return {}

    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Outside a string: quote bare keys. Keys appear either at
            # the start of the segment, after a comma, or after an
            # opening brace (nested objects). The lookbehind
            # ``(?<![\w"])`` avoids mangling partial words and already-
            # quoted strings.
            transformed = re.sub(
                r'(?<![\w"])(\w+)(?=\s*:)',
                r'"\1"',
                part,
            )
            out.append(transformed)
        else:
            # Inside a string: re-emit as a JSON-escaped string body
            # so embedded quotes / backslashes / control chars don't
            # break the outer json.loads.
            escaped = json.dumps(part)  # includes outer quotes
            out.append(escaped)

    wrapped = "{" + "".join(out) + "}"
    parsed = _safe_json_load(wrapped)
    if isinstance(parsed, dict):
        return parsed
    return {}


def parse_gemma4(text: str) -> ParseResult:
    """Parse Gemma 4's split-pipe ``<|tool_call>...<tool_call|>`` markers.

    Extracts each ``<|tool_call>call:NAME{...}<tool_call|>`` block,
    decodes the argument DSL, and returns ``(cleaned_text, calls)``
    where ``cleaned_text`` has the blocks stripped and ``calls`` is
    the list of canonical tool-call dicts.

    Thought / turn / response markers that may surround the call are
    left alone so they can be stripped by the separate channel-marker
    filter (``hfl.engine.llama_cpp._strip_gemma4_channel_markers``).
    Orphan tool markers without a matching body are also left alone —
    they're expected to be cleaned up downstream.
    """
    calls: list[ToolCall] = []

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        body = match.group(2)
        args = _gemma4_dsl_to_dict(body)
        calls.append(_wrap(name, args))
        return ""

    cleaned = _GEMMA4_TOOL_CALL_RE.sub(_sub, text)
    return cleaned, calls


# --- gpt-oss (Harmony) --------------------------------------------------------


# The recipient comes after the channel when the model writes a call and
# before it in the template's own rendering; the JSON starts after
# ``<|message|>`` and runs to ``<|call|>`` (dropped as the stop token, so
# often absent).
_HARMONY_CALL_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?(?:<\|channel\|>\w+\s*)?to=functions\.([^\s<]+)"
    r"(?:(?!<\|message\|>).)*<\|message\|>",
    re.DOTALL,
)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", re.DOTALL
)
_HARMONY_ANY_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|$)|<\|start\|>assistant|"
    r"<\|channel\|>\w*|<\|message\|>|<\|end\|>|<\|return\|>|<\|call\|>|<\|constrain\|>\w*",
    re.DOTALL,
)


def parse_harmony(text: str) -> ParseResult:
    """Parse gpt-oss's Harmony tool calls; the content is the ``final``
    channel (reasoning and channel markers dropped)."""
    calls: list[ToolCall] = []
    consumed: list[tuple[int, int]] = []
    for match in _HARMONY_CALL_RE.finditer(text):
        found = _extract_first_json_object(text[match.end() :])
        if found is None:
            continue
        raw, (start, end) = found
        if text[match.end() : match.end() + start].strip():
            continue  # the JSON must start the message
        calls.append(_wrap(match.group(1), _safe_json_load(raw) or {}))
        consumed.append((match.start(), match.end() + end))
    if not calls and "<|channel|>" not in text:
        return text, calls
    final = _HARMONY_FINAL_RE.search(text)
    if final:
        return final.group(1).strip(), calls
    for start, end in reversed(consumed):
        text = text[:start] + text[end:]
    return _HARMONY_ANY_RE.sub("", text).strip(), calls


# --- DeepSeek -----------------------------------------------------------------


_DEEPSEEK_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)(?:<｜tool▁call▁end｜>|$)",
    re.DOTALL,
)
_DEEPSEEK_WRAPPERS_RE = re.compile(r"<｜tool▁calls▁(?:begin|end)｜>")
_DEEPSEEK_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_deepseek(text: str) -> ParseResult:
    """Parse DeepSeek's tool-call tokens, both spellings: V3.1's
    ``NAME<｜tool▁sep｜>{json}`` and V3 / R1's
    ``function<｜tool▁sep｜>NAME`` followed by a fenced JSON block."""
    calls: list[ToolCall] = []

    def _sub(match: re.Match[str]) -> str:
        head, body = match.group(1).strip(), match.group(2)
        if head == "function":
            name, _, body = body.partition("\n")
            fenced = _DEEPSEEK_FENCE_RE.search(body)
            body = fenced.group(1) if fenced else body
        else:
            name = head
        calls.append(_wrap(name.strip(), _safe_json_load(body.strip()) or {}))
        return ""

    cleaned = _DEEPSEEK_CALL_RE.sub(_sub, text)
    if calls:
        cleaned = _DEEPSEEK_WRAPPERS_RE.sub("", cleaned)
    return _strip_thinking(cleaned).strip(), calls


# --- GLM ----------------------------------------------------------------------


# The body is taken whole and its pairs read by ``_GLM_ARG_RE``: matching
# the pairs inside this pattern (a repeated group of lazy parts) could
# backtrack exponentially on a crafted reply, and a reply can be steered by
# whoever writes the prompt (CodeQL py/redos).
_GLM_CALL_RE = re.compile(
    r"<tool_call>\s*([^\s<{]+)\s*(.*?)(?:</tool_call>|$)",
    re.DOTALL,
)
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
# GLM-4-0414: a call is the function's name on a line of its own, then its
# JSON arguments, each call in an assistant turn of its own.
_GLM4_HEAD_RE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\n\s*(?=\{)")
_GLM4_TURN_RE = re.compile(r"<\|assistant\|>")


def _tool_names(tools: list[dict] | None) -> set[str]:
    names = set()
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def parse_glm(text: str, tools: list[dict] | None = None) -> ParseResult:
    """Parse GLM's tool calls: GLM-4.5+'s ``<arg_key>``/``<arg_value>``
    pairs (values typed from the tool's schema; the template writes
    non-strings as JSON) and GLM-4-0414's ``NAME\n{json}`` — the latter
    only for a name among ``tools``, since it has no marker of its own."""
    calls: list[ToolCall] = []
    # ``<|observation|>`` hands the turn to the tool; anything after it is
    # the model imagining the tool's answer (GLM-4-0414 did, measured).
    text = text.split("<|observation|>", 1)[0]

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        schema = _parameter_schemas(tools, name)
        arguments: dict[str, Any] = {}
        for key, raw in _GLM_ARG_RE.findall(match.group(2)):
            key = key.strip()
            typed = _typed_value(raw, schema.get(key))
            if typed is raw and (schema.get(key) or {}).get("type") != "string":
                parsed = _safe_json_load(raw.strip())
                typed = raw if parsed is None else parsed
            arguments[key] = typed
        calls.append(_wrap(name, arguments))
        return ""

    cleaned = _GLM_CALL_RE.sub(_sub, text)
    if calls:
        return _strip_thinking(cleaned).strip(), calls

    names = _tool_names(tools)
    if names:
        rest: list[str] = []
        for turn in _GLM4_TURN_RE.split(text):
            head = _GLM4_HEAD_RE.match(turn)
            found = (
                _extract_first_json_object(turn[head.end() :])
                if head and head.group(1) in names
                else None
            )
            args = _safe_json_load(found[0]) if found else None
            if head and isinstance(args, dict):
                # The call is the whole turn: whatever follows it is not the
                # model's to say. (On llama-server, where ``<|observation|>``
                # cannot stop it, GLM-4-0414 went on to invent the result.)
                calls.append(_wrap(head.group(1), args))
            else:
                rest.append(turn)
        if calls:
            return "".join(rest).strip(), calls
    return text, calls


# --- Generic JSON envelope fallback -------------------------------------------


_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove qwen/DeepSeek-style ``<think>...</think>`` blocks — and, when
    the template opened the block in the prompt (so only ``</think>``
    reaches the text), everything up to that close."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[1]
    return text


def parse_fallback(text: str) -> ParseResult:
    """Best-effort parser for unstructured JSON tool-call envelopes.

    Recognizes, in order:

    1. A fenced ``json`` code block whose body is a tool-call dict.
    2. A top-level ``{"tool_call": {...}}`` wrapper (non-standard but
       observed from qwen3 when the chat template was not applied — see
       spec §1 Test 2).
    3. A top-level ``{"name": "...", "arguments": {...}}`` dict.

    Only the first recognized envelope is consumed; additional tool calls
    in the same text should have been emitted via the native per-family
    markers already (and therefore parsed earlier in ``dispatch``).
    """
    calls: list[ToolCall] = []
    # Use a list cell so nested helpers can mutate without ``nonlocal``.
    state = {"cleaned": _strip_thinking(text)}

    def _consume(raw: str, full_match: str) -> bool:
        payload = _safe_json_load(raw)
        if not isinstance(payload, dict):
            return False
        # Form 2: {"tool_call": {...}}
        if "tool_call" in payload and isinstance(payload["tool_call"], dict):
            inner = payload["tool_call"]
            if "name" in inner:
                args = inner.get(
                    "arguments",
                    inner.get("args", inner.get("parameters", {})),
                )
                calls.append(_wrap(inner["name"], args))
                state["cleaned"] = state["cleaned"].replace(full_match, "", 1)
                return True
        # Form 4: the tool named under "function" — a string
        # (DeepSeek-R1-Distill 1.5B wrote {"function": "get_weather",
        # "arguments": {...}}) or OpenAI's {"function": {"name", ...}}.
        fn = payload.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            args = fn.get("arguments", payload.get("arguments", {}))
            calls.append(_wrap(fn["name"], args))
            state["cleaned"] = state["cleaned"].replace(full_match, "", 1)
            return True
        if isinstance(fn, str) and fn and "arguments" in payload:
            calls.append(_wrap(fn, payload["arguments"]))
            state["cleaned"] = state["cleaned"].replace(full_match, "", 1)
            return True
        # Form 3: {"name": "...", "arguments": {...}}
        if "name" in payload and (
            "arguments" in payload or "parameters" in payload or "args" in payload
        ):
            args = payload.get(
                "arguments",
                payload.get("args", payload.get("parameters", {})),
            )
            calls.append(_wrap(payload["name"], args))
            state["cleaned"] = state["cleaned"].replace(full_match, "", 1)
            return True
        return False

    # 1. Fenced json block
    m = _FENCED_JSON_RE.search(state["cleaned"])
    if m and _consume(m.group(1), m.group(0)):
        return state["cleaned"].strip(), calls

    # 2/3. Balanced top-level JSON object. We scan for the first "{"
    # and try to parse the maximal balanced substring that starts there.
    candidate = _extract_first_json_object(state["cleaned"])
    if candidate is not None:
        raw, span = candidate
        if _consume(raw, state["cleaned"][span[0] : span[1]]):
            return state["cleaned"].strip(), calls

    return state["cleaned"].strip(), calls


def _extract_first_json_object(text: str) -> tuple[str, tuple[int, int]] | None:
    """Return the first balanced JSON object substring and its span.

    Walks a brace counter while respecting JSON string escaping. Returns
    ``None`` if no balanced object is found.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1], (start, i + 1)
        start = text.find("{", start + 1)
    return None


# --- Dispatch -----------------------------------------------------------------


def _detect_family(model_name: str) -> str:
    """Return a short family tag from a model name.

    We match on common substrings: ``qwen``, ``llama`` (→ llama3 for
    versions >= 3, llama2 otherwise), ``mistral``/``mixtral``,
    ``gemma-4`` / ``gemma4``.

    Note: earlier Gemma versions (2 and 3) do not share the split-
    pipe tool-call DSL that the Gemma 4 family uses, so they are
    *not* routed to ``parse_gemma4``.
    """
    name = (model_name or "").lower()
    if "gpt-oss" in name or "gpt_oss" in name:
        return "harmony"
    # Before "qwen" and "llama": R1's distills (DeepSeek-R1-Distill-Qwen,
    # DeepSeek-R1-0528-Qwen3) are DeepSeek-templated.
    if "deepseek" in name:
        return "deepseek"
    if "glm" in name:
        return "glm"
    if "hermes" in name:
        return "qwen"  # Hermes' <tool_call> is Qwen's
    if "qwen" in name:
        return "qwen"
    if "llama-3" in name or "llama3" in name or "llama 3" in name:
        return "llama3"
    if "llama" in name:
        # llama2 and friends share the llama3-style function tag often
        return "llama3"
    if "mistral" in name or "mixtral" in name:
        return "mistral"
    if "gemma-4" in name or "gemma4" in name or "gemma 4" in name:
        return "gemma4"
    return "generic"


def _answer_text(text: str) -> str:
    """The reply without reasoning, when no native call was found: a Harmony
    reply's ``final`` channel, or the text without ``<think>`` blocks."""
    if "<|channel|>" in text:
        return parse_harmony(text)[0]
    return _DANGLING_CALL_RE.sub("", _strip_thinking(text)).strip()


def dispatch(
    text: str,
    model_name: str | None = None,
    tools: list[dict] | None = None,
) -> ParseResult:
    """Route ``text`` to the appropriate family parser.

    The per-family parser runs first; if it emits at least one tool call,
    the result is returned immediately. Otherwise we fall through to the
    generic fallback parser so that non-template JSON envelopes (like the
    ``{"tool_call": {...}}`` shape documented in spec §1) are still
    recognized.

    ``tools`` is used only as a guard: the fallback parser is **not**
    applied when ``tools`` is empty or ``None``, so ordinary chat replies
    that happen to contain JSON are not misinterpreted as tool calls.
    """
    if not text:
        return "", []

    family = _detect_family(model_name or "")
    parsers_by_family = {
        "qwen": lambda t: parse_qwen(t, tools),
        "llama3": parse_llama3,
        "mistral": parse_mistral,
        "gemma4": parse_gemma4,
        "harmony": parse_harmony,
        "deepseek": parse_deepseek,
        "glm": lambda t: parse_glm(t, tools),
    }
    parser = parsers_by_family.get(family)

    if parser is not None:
        cleaned, calls = parser(text)
        if calls:
            return cleaned, calls

    if not tools:
        return _answer_text(text), []

    # The client sent tools and the family's own parser found nothing: the
    # name may hide the family (an alias such as "coder"), or the text may
    # carry another family's markers (llama-server hands structured calls
    # back and HFL writes them as ``<tool_call>``). Every native marker is
    # unambiguous, so try them all before the loose JSON fallback.
    for native in parsers_by_family.values():
        if native is not parser:
            cleaned, calls = native(text)
            if calls:
                return cleaned, calls

    return parse_fallback(_answer_text(text))
