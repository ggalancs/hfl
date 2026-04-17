# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Minimal Go-template renderer for Ollama Modelfiles (Phase 11 P1 — V2 row 23).

Ollama uses Go's ``text/template`` format for Modelfile ``TEMPLATE``
blocks. The Phase 6 regex substitution ("replace ``{{ .Prompt }}``")
covered 80 % of real Modelfiles but bailed on ``{{ range }}``, ``{{
if }}`` and whitespace-trim syntax. This module implements the
subset we actually see in 2026 Modelfiles:

    {{ .Field }}            → field access (possibly nested: ``.A.B``)
    {{ range .Slice }}...{{ end }}
    {{ if .Flag }}...{{ end }}
    {{ if .X }}...{{ else }}...{{ end }}
    {{ "literal" }}          → string literal
    {{- expr -}}             → whitespace-trim on either side

No pipes, no function calls, no template definitions. Anything the
parser can't handle falls back to the original text + a warning —
never crashes, never raises on the user.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["render_go_template", "GoTemplateError"]


class GoTemplateError(ValueError):
    """Raised by the internal parser; callers see the raw template on failure."""


# ----------------------------------------------------------------------
# Tokeniser — split text into literal + action nodes.
# ----------------------------------------------------------------------


_ACTION_RE = re.compile(
    r"\{\{(?P<ltrim>-)?\s*(?P<body>.*?)\s*(?P<rtrim>-)?\}\}",
    re.DOTALL,
)


@dataclass
class _TextNode:
    text: str


@dataclass
class _ActionNode:
    body: str
    ltrim: bool
    rtrim: bool


def _tokenize(source: str) -> list[_TextNode | _ActionNode]:
    tokens: list[_TextNode | _ActionNode] = []
    pos = 0
    for match in _ACTION_RE.finditer(source):
        if match.start() > pos:
            tokens.append(_TextNode(source[pos : match.start()]))
        tokens.append(
            _ActionNode(
                body=match.group("body"),
                ltrim=bool(match.group("ltrim")),
                rtrim=bool(match.group("rtrim")),
            )
        )
        pos = match.end()
    if pos < len(source):
        tokens.append(_TextNode(source[pos:]))
    return tokens


# ----------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------


@dataclass
class _Node:
    pass


@dataclass
class _Literal(_Node):
    text: str


@dataclass
class _Field(_Node):
    path: tuple[str, ...]


@dataclass
class _StringLit(_Node):
    value: str


@dataclass
class _Block(_Node):
    children: list[_Node] = field(default_factory=list)


@dataclass
class _If(_Node):
    cond: _Field
    then_: _Block
    else_: _Block | None = None


@dataclass
class _Range(_Node):
    expr: _Field
    body: _Block


# ----------------------------------------------------------------------
# Parser (stream-based)
# ----------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_TextNode | _ActionNode]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse_block(self, end_keywords: tuple[str, ...]) -> tuple[_Block, str]:
        block = _Block()
        while self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            if isinstance(tok, _TextNode):
                block.children.append(_Literal(tok.text))
                self.pos += 1
                continue
            # Action node.
            body = tok.body.strip()
            # Apply whitespace trimming on surrounding literal nodes.
            if tok.ltrim and block.children and isinstance(block.children[-1], _Literal):
                prev = block.children[-1]
                prev.text = prev.text.rstrip()
            if tok.rtrim and self.pos + 1 < len(self.tokens):
                nxt = self.tokens[self.pos + 1]
                if isinstance(nxt, _TextNode):
                    nxt.text = nxt.text.lstrip()
            keyword = body.split()[0] if body else ""
            if keyword in end_keywords:
                self.pos += 1
                return block, keyword
            self.pos += 1
            if keyword == "if":
                rest = body[len("if") :].strip()
                cond = self._parse_field(rest)
                then_body, terminator = self.parse_block(("else", "end"))
                else_body: _Block | None = None
                if terminator == "else":
                    else_body, _ = self.parse_block(("end",))
                block.children.append(_If(cond=cond, then_=then_body, else_=else_body))
            elif keyword == "range":
                rest = body[len("range") :].strip()
                expr = self._parse_field(rest)
                body_block, _ = self.parse_block(("end",))
                block.children.append(_Range(expr=expr, body=body_block))
            elif body.startswith('"') and body.endswith('"'):
                block.children.append(_StringLit(body[1:-1]))
            else:
                block.children.append(self._parse_field(body))
        return block, ""

    @staticmethod
    def _parse_field(body: str) -> _Field:
        body = body.strip()
        if not body.startswith("."):
            raise GoTemplateError(f"expected field access, got {body!r}")
        path = tuple(segment for segment in body[1:].split(".") if segment)
        return _Field(path=path)


def _parse(source: str) -> _Block:
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    block, leftover = parser.parse_block(())
    if leftover:
        raise GoTemplateError(f"stray {{{{ {leftover} }}}}")
    return block


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


def _resolve_field(data: Any, path: tuple[str, ...]) -> Any:
    current: Any = data
    for segment in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
            continue
        current = getattr(current, segment, None)
    return current


def _truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _render(node: _Node, data: Any, out: list[str]) -> None:
    if isinstance(node, _Block):
        for child in node.children:
            _render(child, data, out)
    elif isinstance(node, _Literal):
        out.append(node.text)
    elif isinstance(node, _StringLit):
        out.append(node.value)
    elif isinstance(node, _Field):
        value = _resolve_field(data, node.path)
        if value is None or value is False:
            return
        out.append(str(value))
    elif isinstance(node, _If):
        value = _resolve_field(data, node.cond.path)
        if _truthy(value):
            _render(node.then_, data, out)
        elif node.else_ is not None:
            _render(node.else_, data, out)
    elif isinstance(node, _Range):
        value = _resolve_field(data, node.expr.path)
        if not value:
            return
        if isinstance(value, dict):
            value = list(value.values())
        try:
            iterator = iter(value)
        except TypeError:
            return
        for item in iterator:
            _render(node.body, item, out)


# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------


def render_go_template(source: str, data: Any) -> str:
    """Render a Go-template string against ``data``.

    ``data`` is typically a dict with string keys matching the
    Modelfile conventions (``Prompt``, ``System``, ``Messages``,
    ``Response``, …). Missing fields render as empty strings; falsey
    fields inside ``{{ if }}`` go down the ``else`` branch.

    On parse errors the original ``source`` is returned unmodified
    and a WARNING is logged — the Modelfile renderer is a convenience
    layer, never a gate.
    """
    try:
        tree = _parse(source)
    except GoTemplateError as exc:
        logger.warning("Go-template parse failed, falling back to literal: %s", exc)
        return source
    out: list[str] = []
    _render(tree, data, out)
    return "".join(out)
