# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Parse an Ollama Modelfile into a structured ModelfileDocument.

Renderer in hfl.converter.modelfile goes the other way (ModelManifest
to Modelfile text). This module closes the loop so POST /api/create
can accept a Modelfile body and materialise a manifest from it.

Grammar (from https://docs.ollama.com/modelfile): instructions FROM,
PARAMETER, TEMPLATE, SYSTEM, ADAPTER, LICENSE, MESSAGE, REQUIRES. TEMPLATE
/ SYSTEM / LICENSE / MESSAGE values may be wrapped in a single-line
double-quoted string or a triple-quoted block spanning many lines.
Both honour backslash-escapes for newline, tab, quote, backslash.

Strict on structural errors (unknown instructions, missing FROM,
unterminated quotes); lenient on formatting (case-insensitive
keywords, any whitespace between tokens, comments at line start).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ModelfileDocument",
    "ModelfileParseError",
    "ModelfileMessage",
    "parse_modelfile",
    "render_modelfile_document",
    "KNOWN_PARAMETERS",
    "INT_PARAMETERS",
    "FLOAT_PARAMETERS",
    "BOOL_PARAMETERS",
]


# Parameters whose values are parsed as int/float/bool. Everything
# else (notably ``stop``) stays as a string to preserve the user's
# quoting. The taxonomy mirrors Ollama's Modelfile reference.
INT_PARAMETERS: frozenset[str] = frozenset(
    {
        "num_ctx",
        "num_keep",
        "num_predict",
        "num_batch",
        "num_gpu",
        "num_gqa",
        "num_thread",
        "repeat_last_n",
        "mirostat",
        "seed",
        "top_k",
    }
)

FLOAT_PARAMETERS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "min_p",
        "repeat_penalty",
        "mirostat_eta",
        "mirostat_tau",
        "tfs_z",
        "typical_p",
        "presence_penalty",
        "frequency_penalty",
    }
)

BOOL_PARAMETERS: frozenset[str] = frozenset(
    {
        "numa",
        "penalize_newline",
        "f16_kv",
        "low_vram",
        "vocab_only",
        "use_mmap",
        "use_mlock",
    }
)

KNOWN_PARAMETERS: frozenset[str] = (
    INT_PARAMETERS | FLOAT_PARAMETERS | BOOL_PARAMETERS | frozenset({"stop"})
)

_VALID_MESSAGE_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


class ModelfileParseError(ValueError):
    """Raised when a Modelfile cannot be parsed.

    Carries a 1-based ``line`` attribute so callers can render useful
    diagnostics (``/api/create`` surfaces this as a 400 response).
    """

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        if line is not None:
            message = f"line {line}: {message}"
        super().__init__(message)


@dataclass
class ModelfileMessage:
    """A single ``MESSAGE`` instruction from a Modelfile."""

    role: str
    content: str


@dataclass
class ModelfileDocument:
    """Parsed, validated Modelfile.

    The ``from_`` field is required (a Modelfile without ``FROM`` is
    rejected at parse time). Everything else is optional and defaults
    to an empty container so callers don't need ``is None`` guards.
    """

    from_: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    stop_sequences: list[str] = field(default_factory=list)
    template: str | None = None
    system: str | None = None
    adapters: list[str] = field(default_factory=list)
    license: str | None = None
    messages: list[ModelfileMessage] = field(default_factory=list)
    requires: str | None = None
    # Phase 14 — V2 rows 19 / 22.
    # ``env`` captures ``ENV KEY=VALUE`` lines; the engine applies them
    # in a scoped environment during load. ``capabilities`` captures
    # an explicit ``CAPABILITIES completion,tools,...`` declaration
    # that overrides auto-detection.
    env: dict[str, str] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)

    def to_manifest_fields(self) -> dict[str, Any]:
        """Flatten the document into kwargs for ``ModelManifest``.

        Returns a dict that can be ``**``-unpacked into the manifest
        constructor or merged into an existing manifest's ``__dict__``.
        Only fields that the manifest actually supports are included.
        """
        result: dict[str, Any] = {}
        if self.system is not None:
            result["system"] = self.system
        if self.template is not None:
            result["chat_template"] = self.template
        # ``num_ctx`` on the Modelfile maps to ``context_length`` on
        # the manifest so the engine's existing context-sizing logic
        # picks it up without a translation layer.
        num_ctx = self.parameters.get("num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0:
            result["context_length"] = num_ctx
        # Other PARAMETER values go under ``default_parameters`` for
        # the engine to apply on every request.
        defaults: dict[str, Any] = {}
        for key, value in self.parameters.items():
            if key == "num_ctx":
                continue
            defaults[key] = value
        if self.stop_sequences:
            defaults["stop"] = list(self.stop_sequences)
        if defaults:
            result["default_parameters"] = defaults
        if self.adapters:
            result["adapter_paths"] = list(self.adapters)
        if self.license is not None:
            result["license_name"] = self.license
        if self.env:
            result["env_vars"] = dict(self.env)
        if self.capabilities:
            # Preserved on the manifest under an explicit list so
            # downstream consumers can skip auto-detection.
            result["declared_capabilities"] = list(self.capabilities)
        return result


# ---------------------------------------------------------------------
# Tokeniser / value reader
# ---------------------------------------------------------------------


def _decode_quoted(raw: str) -> str:
    """Decode ``\\n``, ``\\t``, ``\\"`` and ``\\\\`` inside a quoted value.

    Ollama's parser is conservative about escapes — it recognises the
    four sequences above and leaves everything else as-is (so a stray
    ``\\x`` stays ``\\x``). We match that behaviour.
    """
    out: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt == "r":
                out.append("\r")
                i += 2
                continue
            if nxt == '"':
                out.append('"')
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _read_value(
    rest: str,
    lines: list[str],
    start_idx: int,
) -> tuple[str, int]:
    """Extract the value from an instruction with a quoted argument.

    Returns (decoded_value, lines_consumed). lines_consumed is 1 when
    the value fits on the same physical line, more when a triple-quoted
    block spans multiple lines. Accepts triple-quoted blocks (single- or
    multi-line), single-line double-quoted strings, or bare unquoted
    values.
    """
    rest = rest.strip()

    # ---- triple-quoted -------------------------------------------
    if rest.startswith('"""'):
        body = rest[3:]
        # Same-line: ``"""foo"""``
        if body.endswith('"""') and len(body) >= 3:
            return _decode_quoted(body[:-3]), 1
        # Multi-line: collect until a line ends with ``"""``
        collected: list[str] = [body]
        j = start_idx + 1
        while j < len(lines):
            cur = lines[j]
            if cur.rstrip().endswith('"""'):
                ending = cur.rstrip()
                collected.append(ending[:-3].rstrip())
                # Drop the leading fragment if it was empty (i.e.
                # the opener was on its own line).
                if collected and collected[0] == "":
                    collected = collected[1:]
                return (
                    _decode_quoted("\n".join(collected)),
                    j - start_idx + 1,
                )
            collected.append(cur)
            j += 1
        raise ModelfileParseError(
            "unterminated triple-quoted string",
            line=start_idx + 1,
        )

    # ---- single-line double-quoted -------------------------------
    if rest.startswith('"'):
        # Walk the string honouring backslash escapes.
        i = 1
        while i < len(rest):
            c = rest[i]
            if c == "\\" and i + 1 < len(rest):
                i += 2
                continue
            if c == '"':
                inside = rest[1:i]
                trailing = rest[i + 1 :].strip()
                # Accept trailing comment ``# ...`` after the close.
                if trailing and not trailing.startswith("#"):
                    raise ModelfileParseError(
                        "unexpected tokens after quoted value",
                        line=start_idx + 1,
                    )
                return _decode_quoted(inside), 1
            i += 1
        raise ModelfileParseError(
            "unterminated double-quoted string",
            line=start_idx + 1,
        )

    # ---- bare value ---------------------------------------------
    # Strip an inline ``# comment`` tail but only if preceded by
    # whitespace; ``sha256:#deadbeef`` must survive.
    comment_at = _find_inline_comment(rest)
    if comment_at is not None:
        rest = rest[:comment_at].rstrip()
    return rest, 1


def _find_inline_comment(line: str) -> int | None:
    """Return index of an inline ``# comment`` (preceded by space), else None."""
    for i, c in enumerate(line):
        if c == "#" and (i == 0 or line[i - 1] in (" ", "\t")):
            return i
    return None


# ---------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------


def _coerce_bool(value: str, *, line: int) -> bool:
    lower = value.lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    raise ModelfileParseError(
        f"PARAMETER expects bool (true/false), got {value!r}",
        line=line,
    )


def _parse_parameter(
    rest: str,
    doc: ModelfileDocument,
    lines: list[str],
    idx: int,
) -> int:
    """Parse a ``PARAMETER`` line into ``doc``.

    Returns lines-consumed (1 for everything except quoted values that
    span multiple lines — currently only ``stop`` can do that).
    """
    parts = rest.split(None, 1)
    if len(parts) != 2:
        raise ModelfileParseError(
            "PARAMETER requires a key and a value",
            line=idx + 1,
        )
    key, raw_value = parts[0], parts[1]
    key_lower = key.lower()

    if key_lower == "stop":
        value, consumed = _read_value(raw_value, lines, idx)
        doc.stop_sequences.append(value)
        return consumed

    # For other params, the value is bare (or single-quoted).
    value_text, consumed = _read_value(raw_value, lines, idx)

    if key_lower in INT_PARAMETERS:
        try:
            doc.parameters[key_lower] = int(value_text)
        except ValueError as exc:
            raise ModelfileParseError(
                f"PARAMETER {key} expects int, got {value_text!r}",
                line=idx + 1,
            ) from exc
    elif key_lower in FLOAT_PARAMETERS:
        try:
            doc.parameters[key_lower] = float(value_text)
        except ValueError as exc:
            raise ModelfileParseError(
                f"PARAMETER {key} expects float, got {value_text!r}",
                line=idx + 1,
            ) from exc
    elif key_lower in BOOL_PARAMETERS:
        doc.parameters[key_lower] = _coerce_bool(value_text, line=idx + 1)
    else:
        # Unknown parameter — accept as string. Ollama is permissive
        # here and forwards unknown params to the engine; preserving
        # the raw text lets clients do the same.
        doc.parameters[key_lower] = value_text

    return consumed


# ---------------------------------------------------------------------
# MESSAGE parsing
# ---------------------------------------------------------------------


def _parse_message(
    rest: str,
    lines: list[str],
    idx: int,
) -> tuple[ModelfileMessage, int]:
    """Parse a ``MESSAGE role "content"`` line."""
    parts = rest.split(None, 1)
    if len(parts) != 2:
        raise ModelfileParseError(
            "MESSAGE requires a role and content",
            line=idx + 1,
        )
    role, raw_value = parts[0].lower(), parts[1]
    if role not in _VALID_MESSAGE_ROLES:
        raise ModelfileParseError(
            f"MESSAGE role must be one of {sorted(_VALID_MESSAGE_ROLES)}, got {role!r}",
            line=idx + 1,
        )
    content, consumed = _read_value(raw_value, lines, idx)
    return ModelfileMessage(role=role, content=content), consumed


# ---------------------------------------------------------------------
# Public parse entrypoint
# ---------------------------------------------------------------------


_INCLUDE_RE = re.compile(r"^\s*INCLUDE\s+(.+?)\s*(?:#.*)?$", re.IGNORECASE)


def _expand_includes(
    text: str,
    base_path: Path | None,
    seen: set[Path] | None = None,
    depth: int = 0,
) -> str:
    """Inline ``INCLUDE <path>`` lines before parsing (Phase 14 — V2 row 21).

    Paths resolve relative to ``base_path``. Cycle detection via the
    ``seen`` set; ``depth`` caps runaway chains.
    """
    if base_path is None or "INCLUDE" not in text.upper():
        return text
    if depth > 16:
        raise ModelfileParseError("INCLUDE chain exceeds depth 16")
    seen = set(seen or ())
    out_lines: list[str] = []
    for raw_line in text.splitlines():
        match = _INCLUDE_RE.match(raw_line)
        if not match:
            out_lines.append(raw_line)
            continue
        rel = match.group(1).strip().strip('"').strip("'")
        target = (base_path / rel).resolve()
        if target in seen:
            raise ModelfileParseError(f"INCLUDE cycle: {target}")
        if not target.exists() or not target.is_file():
            raise ModelfileParseError(f"INCLUDE target missing: {target}")
        try:
            included = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelfileParseError(f"INCLUDE {target} unreadable: {exc}") from exc
        next_seen = seen | {target}
        expanded = _expand_includes(
            included,
            target.parent,
            seen=next_seen,
            depth=depth + 1,
        )
        out_lines.append(f"# <<< INCLUDE {rel}")
        out_lines.append(expanded)
        out_lines.append(f"# >>> INCLUDE {rel}")
    return "\n".join(out_lines)


def parse_modelfile(text: str, base_path: str | Path | None = None) -> ModelfileDocument:
    """Parse ``text`` and return a validated ``ModelfileDocument``.

    Raises ``ModelfileParseError`` with ``.line`` pointing at the
    first failure. A Modelfile without ``FROM`` is always rejected.

    ``base_path`` enables ``INCLUDE`` resolution (Phase 14 — V2 row
    21). When ``None`` (default), ``INCLUDE`` lines parse as the
    unknown instruction they are — inlining only happens when the
    caller supplies a filesystem anchor.
    """
    base = Path(base_path) if base_path is not None else None
    if base is not None:
        text = _expand_includes(text, base)
    doc = ModelfileDocument()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Blank or full-line comment → skip.
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        parts = stripped.split(None, 1)
        instruction = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""

        if not rest and instruction not in {"FROM"}:
            # FROM without a value also fails below; other instructions
            # are caught here so the error points at the right line.
            raise ModelfileParseError(
                f"instruction {instruction!r} requires a value",
                line=i + 1,
            )

        if instruction == "FROM":
            if doc.from_:
                raise ModelfileParseError(
                    "duplicate FROM instruction",
                    line=i + 1,
                )
            if not rest:
                raise ModelfileParseError(
                    "FROM requires a value",
                    line=i + 1,
                )
            # Strip inline comments (same heuristic as bare values).
            cleaned = rest
            comment_at = _find_inline_comment(cleaned)
            if comment_at is not None:
                cleaned = cleaned[:comment_at].rstrip()
            doc.from_ = cleaned
            i += 1
            continue

        if instruction == "TEMPLATE":
            value, consumed = _read_value(rest, lines, i)
            doc.template = value
            i += consumed
            continue

        if instruction == "SYSTEM":
            value, consumed = _read_value(rest, lines, i)
            doc.system = value
            i += consumed
            continue

        if instruction == "LICENSE":
            value, consumed = _read_value(rest, lines, i)
            doc.license = value
            i += consumed
            continue

        if instruction == "PARAMETER":
            consumed = _parse_parameter(rest, doc, lines, i)
            i += consumed
            continue

        if instruction == "ADAPTER":
            doc.adapters.append(rest.strip())
            i += 1
            continue

        if instruction == "MESSAGE":
            message, consumed = _parse_message(rest, lines, i)
            doc.messages.append(message)
            i += consumed
            continue

        if instruction == "REQUIRES":
            # Trim quotes if the user wrote them.
            value = rest.strip()
            if value.startswith('"') and value.endswith('"') and len(value) >= 2:
                value = value[1:-1]
            doc.requires = value
            i += 1
            continue

        if instruction == "ENV":
            # ``ENV KEY=VALUE`` — mimics Dockerfile's ENV syntax.
            # Value can be bare, ``"quoted"``, or a raw rest-of-line.
            raw = rest.strip()
            if "=" not in raw:
                raise ModelfileParseError(
                    "ENV requires KEY=VALUE",
                    line=i + 1,
                )
            key, _, rhs = raw.partition("=")
            key = key.strip()
            rhs = rhs.strip()
            if rhs.startswith('"') and rhs.endswith('"') and len(rhs) >= 2:
                rhs = _decode_quoted(rhs[1:-1])
            if not key:
                raise ModelfileParseError(
                    "ENV KEY must not be empty",
                    line=i + 1,
                )
            doc.env[key] = rhs
            i += 1
            continue

        if instruction == "CAPABILITIES":
            # ``CAPABILITIES completion,tools,vision`` — comma-separated.
            tokens_csv = [t.strip() for t in rest.replace(",", " ").split() if t.strip()]
            for token_ in tokens_csv:
                # Preserve case for downstream comparisons but dedupe.
                if token_ not in doc.capabilities:
                    doc.capabilities.append(token_)
            i += 1
            continue

        raise ModelfileParseError(
            f"unknown instruction {instruction!r}",
            line=i + 1,
        )

    if not doc.from_:
        raise ModelfileParseError("Modelfile missing FROM instruction")

    return doc


# ---------------------------------------------------------------------
# Render back to text (round-trip)
# ---------------------------------------------------------------------


def _quote_triple(value: str) -> str:
    """Render ``value`` as a triple-quoted block.

    Values containing an embedded triplet of double-quotes would break
    the shape; we defensively escape the offending triplet even though
    it is vanishingly rare in practice.
    """
    if '"""' in value:
        value = value.replace('"""', r"\"\"\"")
    if "\n" in value:
        return f'"""{value}"""'
    return f'"""{value}"""'


def _quote_single(value: str) -> str:
    """Render ``value`` as a single-line double-quoted string.

    Escapes ``\\`` and ``"``; passes other control chars through.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_modelfile_document(doc: ModelfileDocument) -> str:
    """Serialise ``doc`` back to Modelfile text.

    The output is deterministic and round-trips: ``parse_modelfile(
    render_modelfile_document(doc))`` yields a document equal to
    ``doc`` (tested in ``tests/test_modelfile_parser.py``).
    """
    lines: list[str] = []
    lines.append(f"FROM {doc.from_}")

    if doc.template is not None:
        lines.append("")
        lines.append(f"TEMPLATE {_quote_triple(doc.template)}")

    if doc.system is not None:
        lines.append("")
        lines.append(f"SYSTEM {_quote_triple(doc.system)}")

    # PARAMETER block. Sort alphabetically for stable output; ``stop``
    # goes last so it's visually grouped at the bottom of the block.
    param_keys = sorted(k for k in doc.parameters.keys() if k != "stop")
    if param_keys or doc.stop_sequences:
        lines.append("")
        for key in param_keys:
            value = doc.parameters[key]
            if isinstance(value, bool):
                # Python's str(True) is "True"; Ollama uses lowercase.
                rendered: str = "true" if value else "false"
            else:
                rendered = str(value)
            lines.append(f"PARAMETER {key} {rendered}")
        for stop in doc.stop_sequences:
            lines.append(f"PARAMETER stop {_quote_single(stop)}")

    for adapter in doc.adapters:
        lines.append("")
        lines.append(f"ADAPTER {adapter}")

    for message in doc.messages:
        lines.append("")
        lines.append(f"MESSAGE {message.role} {_quote_single(message.content)}")

    if doc.license is not None:
        lines.append("")
        lines.append(f"LICENSE {_quote_triple(doc.license)}")

    if doc.requires is not None:
        lines.append("")
        lines.append(f"REQUIRES {doc.requires}")

    return "\n".join(lines) + "\n"
