# SPDX-License-Identifier: HRUL-1.0
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


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_qwen(text: str) -> ParseResult:
    """Parse qwen-family ``<tool_call>...</tool_call>`` markers."""
    calls: list[ToolCall] = []

    def _sub(match: re.Match) -> str:
        payload = _safe_json_load(match.group(1))
        if isinstance(payload, dict) and "name" in payload:
            calls.append(
                _wrap(
                    payload["name"],
                    payload.get("arguments", payload.get("parameters", {})),
                )
            )
        return ""

    cleaned = _QWEN_TOOL_CALL_RE.sub(_sub, text)
    return _strip_thinking(cleaned).strip(), calls


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


# --- Generic JSON envelope fallback -------------------------------------------


_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove qwen/DeepSeek-style ``<think>...</think>`` blocks."""
    return _THINK_RE.sub("", text)


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
        "qwen": parse_qwen,
        "llama3": parse_llama3,
        "mistral": parse_mistral,
        "gemma4": parse_gemma4,
    }
    parser = parsers_by_family.get(family)

    if parser is not None:
        cleaned, calls = parser(text)
        if calls:
            return cleaned, calls

    if not tools:
        return _strip_thinking(text).strip(), []

    return parse_fallback(text)
