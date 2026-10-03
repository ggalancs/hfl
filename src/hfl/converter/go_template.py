# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Go-template renderer for Ollama Modelfile ``TEMPLATE`` blocks.

Ollama uses Go's ``text/template``. The first version of this module handled
fields, ``if``/``else``, ``range`` and string literals — enough for
``/api/generate``'s ``{{ .System }} {{ .Prompt }}`` templates, not for a chat
template: Ollama's own use ``eq .Role "user"``, ``$.System``, ``range $i, $m
:= .Messages`` and ``len (slice $.Messages $i)`` to find the last turn. This
is the subset those need:

    {{ .A.B }}  {{ . }}  {{ $ }}  {{ $.A }}  {{ $x.A }}    fields and variables
    {{ "s" }}  {{ `raw` }}  {{ 3 }}  {{ true }}  {{ nil }} literals
    {{ fn arg ... }}  {{ (fn arg) }}  {{ arg | fn }}       calls, pipelines
    {{ $x := pipeline }}  {{ $x = pipeline }}              variables
    {{ if p }}..{{ else if p }}..{{ else }}..{{ end }}
    {{ range p }}  {{ range $v := p }}  {{ range $i, $v := p }} .. {{ else }} .. {{ end }}
    {{ with p }}..{{ else }}..{{ end }}
    {{- trim -}}   {{/* comment */}}

Functions: ``eq ne lt le gt ge and or not len index slice json print printf``;
``{{ break }}`` and ``{{ continue }}`` in a range.

Templates are attacker-controlled (``template`` in an /api/generate body,
``TEMPLATE`` in a Modelfile), so: field lookups never resolve a segment
starting with ``_`` and never walk into a primitive's attributes, no function
reaches Python beyond the list above, and output is capped
(``MAX_OUTPUT``) — nested ranges multiply — and so is the work done
(``MAX_STEPS``): ``{{ range 9999999999 }}{{ end }}`` prints nothing.

:func:`render_go_template` falls back to the literal source on any error
(``/api/generate``: a convenience, never a gate); :func:`render_strict`
raises, for callers that have something better to fall back to.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["render_go_template", "render_strict", "GoStruct", "GoTemplateError", "MAX_OUTPUT"]

MAX_OUTPUT = 4_000_000  # characters
# Nodes rendered plus range iterations. A real template over a long chat is a
# few dozen per message; an empty loop over an integer literal output never
# caught (``{{ range 3000000 }}`` built a 3M-item list first: 319 MB).
MAX_STEPS = 1_000_000


class GoTemplateError(ValueError):
    """A template this renderer cannot parse or evaluate."""


# ----------------------------------------------------------------------
# Tokeniser — split text into literal + action nodes.
# ----------------------------------------------------------------------


# A trim marker is a ``-`` next to the braces with whitespace on its other
# side, as in Go (``{{-3}}`` is the number -3).
_ACTION_RE = re.compile(
    r"\{\{(?P<ltrim>-(?=\s))?\s*(?P<body>.*?)\s*(?:(?<=\s)(?P<rtrim>-))?\}\}",
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
    # Trimming applies to the neighbouring text, whatever block it ends up in.
    for i, tok in enumerate(tokens):
        if not isinstance(tok, _ActionNode):
            continue
        if tok.ltrim and i > 0 and isinstance(tokens[i - 1], _TextNode):
            prev = tokens[i - 1]
            assert isinstance(prev, _TextNode)
            prev.text = prev.text.rstrip()
        if tok.rtrim and i + 1 < len(tokens) and isinstance(tokens[i + 1], _TextNode):
            nxt = tokens[i + 1]
            assert isinstance(nxt, _TextNode)
            nxt.text = nxt.text.lstrip()
    return tokens


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------

_EXPR_TOKEN = re.compile(
    r"""\s*(?:
        (?P<str>"(?:[^"\\]|\\.)*")
      | (?P<raw>`[^`]*`)
      | (?P<num>-?\d+(?:\.\d+)?)
      | (?P<lp>\()
      | (?P<rp>\))
      | (?P<pipe>\|)
      | (?P<decl>:=)
      | (?P<assign>=)
      | (?P<comma>,)
      | (?P<var>\$[A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)
      | (?P<field>\.(?:[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)?)
      | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)


def _lex(body: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    body = body.rstrip()
    while pos < len(body):
        match = _EXPR_TOKEN.match(body, pos)
        if match is None or match.end() == pos:
            raise GoTemplateError(f"cannot read {body[pos:]!r}")
        kind = match.lastgroup or ""
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


@dataclass
class _Operand:
    kind: str  # field | var | lit | ident | sub
    path: tuple[str, ...] = ()
    name: str = ""
    value: Any = None
    sub: _Pipeline | None = None


@dataclass
class _Pipeline:
    cmds: list[list[_Operand]]
    decl: list[str] = field(default_factory=list)
    assign: bool = False  # ``=`` rather than ``:=``


class _ExprParser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> tuple[str, str] | None:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def pipeline(self, allow_decl: bool = True, until_rp: bool = False) -> _Pipeline:
        decl: list[str] = []
        assign = False
        if allow_decl:
            # $a := / $a = / $i, $v :=
            save = self.pos
            names: list[str] = []
            while (tok := self.peek()) is not None and tok[0] == "var" and "." not in tok[1]:
                names.append(tok[1])
                self.pos += 1
                nxt = self.peek()
                if nxt is not None and nxt[0] == "comma":
                    self.pos += 1
                    continue
                break
            nxt = self.peek()
            if names and nxt is not None and nxt[0] in ("decl", "assign"):
                decl, assign = names, nxt[0] == "assign"
                self.pos += 1
            else:
                self.pos = save
        cmds = [self.command()]
        while (tok := self.peek()) is not None and tok[0] == "pipe":
            self.pos += 1
            cmds.append(self.command())
        if not until_rp and self.peek() is not None:
            raise GoTemplateError(f"unexpected {self.peek()}")
        return _Pipeline(cmds=cmds, decl=decl, assign=assign)

    def command(self) -> list[_Operand]:
        operands: list[_Operand] = []
        while (tok := self.peek()) is not None and tok[0] not in ("pipe", "rp"):
            operands.append(self.operand())
        if not operands:
            raise GoTemplateError("empty command")
        return operands

    def operand(self) -> _Operand:
        tok = self.peek()
        assert tok is not None
        kind, text = tok
        self.pos += 1
        if kind == "field":
            return _Operand("field", path=tuple(p for p in text[1:].split(".") if p))
        if kind == "var":
            name, *rest = text.split(".")
            return _Operand("var", name=name, path=tuple(rest))
        if kind == "str":
            return _Operand("lit", value=json.loads(text))
        if kind == "raw":
            return _Operand("lit", value=text[1:-1])
        if kind == "num":
            return _Operand("lit", value=float(text) if "." in text else int(text))
        if kind == "ident":
            if text in ("true", "false"):
                return _Operand("lit", value=text == "true")
            if text == "nil":
                return _Operand("lit", value=None)
            if text not in _FUNCS:
                raise GoTemplateError(f"function {text!r} not defined")
            return _Operand("ident", name=text)
        if kind == "lp":
            sub = self.pipeline(allow_decl=False, until_rp=True)
            end = self.peek()
            if end is None or end[0] != "rp":
                raise GoTemplateError("unclosed (")
            self.pos += 1
            return _Operand("sub", sub=sub)
        raise GoTemplateError(f"unexpected {text!r}")


def _parse_pipeline(body: str) -> _Pipeline:
    return _ExprParser(_lex(body)).pipeline()


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
class _Action(_Node):
    pipe: _Pipeline


@dataclass
class _Jump(_Node):
    kind: str  # break | continue


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass
class _Block(_Node):
    children: list[_Node] = field(default_factory=list)


@dataclass
class _If(_Node):
    branches: list[tuple[_Pipeline, _Block]]
    else_: _Block | None = None


@dataclass
class _Range(_Node):
    pipe: _Pipeline
    body: _Block
    else_: _Block | None = None


@dataclass
class _With(_Node):
    pipe: _Pipeline
    body: _Block
    else_: _Block | None = None


class _Parser:
    def __init__(self, tokens: list[_TextNode | _ActionNode]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse_block(self, end_keywords: tuple[str, ...]) -> tuple[_Block, str, str]:
        """Nodes up to one of ``end_keywords``; (block, keyword, its body)."""
        block = _Block()
        while self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            self.pos += 1
            if isinstance(tok, _TextNode):
                if tok.text:
                    block.children.append(_Literal(tok.text))
                continue
            body = tok.body.strip()
            if body.startswith("/*"):
                continue
            keyword = body.split()[0] if body else ""
            if keyword in end_keywords:
                return block, keyword, body
            if keyword in ("end", "else"):
                raise GoTemplateError(f"unexpected {{{{ {keyword} }}}}")
            rest = body[len(keyword) :].strip()
            if keyword == "if":
                block.children.append(self._if(rest))
            elif keyword in ("range", "with"):
                pipe = _parse_pipeline(rest)
                inner, term, _ = self.parse_block(("else", "end"))
                other = self.parse_block(("end",))[0] if term == "else" else None
                if term == "":
                    raise GoTemplateError(f"{keyword} without end")
                node = _Range if keyword == "range" else _With
                block.children.append(node(pipe=pipe, body=inner, else_=other))
            elif keyword in ("break", "continue") and body == keyword:
                block.children.append(_Jump(keyword))
            elif keyword in ("define", "template", "block"):
                raise GoTemplateError(f"{{{{ {keyword} }}}} is not supported")
            else:
                block.children.append(_Action(_parse_pipeline(body)))
        return block, "", ""

    def _if(self, cond: str) -> _If:
        branches: list[tuple[_Pipeline, _Block]] = []
        pipe = _parse_pipeline(cond)
        while True:
            inner, term, term_body = self.parse_block(("else", "end"))
            branches.append((pipe, inner))
            if term == "end":
                return _If(branches=branches)
            if term == "":
                raise GoTemplateError("if without end")
            after = term_body[len("else") :].strip()
            if after.startswith("if ") or after == "if":
                pipe = _parse_pipeline(after[2:])
                continue
            else_block, term, _ = self.parse_block(("end",))
            if term != "end":
                raise GoTemplateError("if without end")
            return _If(branches=branches, else_=else_block)


def _parse(source: str) -> _Block:
    parser = _Parser(_tokenize(source))
    block, leftover, _ = parser.parse_block(())
    if leftover:
        raise GoTemplateError(f"stray {{{{ {leftover} }}}}")
    return block


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


_PRIMITIVES = (str, bytes, bytearray, int, float, bool, complex)


def _resolve_field(data: Any, path: tuple[str, ...]) -> Any:
    """Walk a dotted template path over ``data``.

    SEC: this used to fall back to a bare ``getattr`` on any object, and
    the rendered value is stringified straight into the prompt. Templates
    are attacker-controlled, so ``{{ .Prompt.__class__.__mro__ }}`` walked
    out of the data dict and into Python's object graph — verified in audit.

    Two rules close it without changing any legitimate template:

    1. Segments starting with ``_`` never resolve — no dunder traversal.
    2. Primitives have no template fields, so a lookup on one ends the
       walk instead of exposing ``str``/``list`` internals.

    Mapping and attribute lookup on ordinary objects still work.
    """
    current: Any = data
    for segment in path:
        if current is None:
            return None
        if segment.startswith("_"):
            return None
        if isinstance(current, Mapping):
            current = current.get(segment)
            continue
        if isinstance(current, _PRIMITIVES):
            return None
        current = getattr(current, segment, None)
        if callable(current):
            return None  # a method is not a field
    return current


def _truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _eq(a: Any, *others: Any) -> bool:
    if not others:
        raise GoTemplateError("eq needs two arguments")
    return any(a == b for b in others)


def _ordered(op: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    def compare(a: Any, b: Any) -> bool:
        try:
            return bool(op(a, b))
        except TypeError as exc:
            raise GoTemplateError(
                f"cannot compare {type(a).__name__} and {type(b).__name__}"
            ) from exc

    return compare


def _and(*args: Any) -> Any:
    for arg in args:
        if not _truthy(arg):
            return arg
    return args[-1] if args else None


def _or(*args: Any) -> Any:
    for arg in args:
        if _truthy(arg):
            return arg
    return args[-1] if args else None


def _len(value: Any) -> int:
    if isinstance(value, (str, list, tuple, dict)):
        return len(value)
    raise GoTemplateError(f"len of {type(value).__name__}")


def _index(value: Any, *keys: Any) -> Any:
    for key in keys:
        if isinstance(value, (list, tuple, str)) and isinstance(key, int):
            if not -len(value) <= key < len(value):
                raise GoTemplateError(f"index {key} out of range")
            value = value[key]
        elif isinstance(value, Mapping) and isinstance(key, str) and not key.startswith("_"):
            value = value.get(key)
        else:
            raise GoTemplateError("cannot index that")
    return value


def _slice(value: Any, *bounds: Any) -> Any:
    if not isinstance(value, (list, tuple, str)) or len(bounds) > 2:
        raise GoTemplateError("cannot slice that")
    if not all(isinstance(b, int) and 0 <= b <= len(value) for b in bounds):
        raise GoTemplateError("slice bounds out of range")
    start = bounds[0] if bounds else 0
    stop = bounds[1] if len(bounds) > 1 else len(value)
    if start > stop:
        raise GoTemplateError("slice bounds out of range")
    return value[start:stop]


class GoStruct(dict[str, Any]):
    """A value with Go-style fields (``.Function.Name``) whose ``json`` is
    another shape (``{"function": {"name": ...}}``), as Ollama's structs:
    fields by their Go name, JSON by their tags."""

    def __init__(self, fields: Mapping[str, Any], wire: Any) -> None:
        super().__init__(fields)
        self.wire = wire


def _wire(value: Any) -> Any:
    if isinstance(value, GoStruct):
        return value.wire
    if isinstance(value, Mapping):
        return {k: _wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(v) for v in value]
    return value


def _json(value: Any) -> str:
    """Go's ``json.Marshal``: compact, and HTML-safe (``<`` is ``\\u003c``)."""
    text = json.dumps(_wire(value), ensure_ascii=False, separators=(",", ":"), default=str)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _print(*args: Any) -> str:
    return "".join(_text(a) for a in args)


_VERB = re.compile(r"%([%svdq])")


def _printf(fmt: Any, *args: Any) -> str:
    """Go's ``printf`` for the verbs templates use: ``%s %v %d %q %%``."""
    if not isinstance(fmt, str):
        raise GoTemplateError("printf needs a format string")
    queue = list(args)

    def verb(match: re.Match[str]) -> str:
        if match.group(1) == "%":
            return "%"
        if not queue:
            return f"%!{match.group(1)}(MISSING)"
        value = queue.pop(0)
        return (
            json.dumps(_text(value), ensure_ascii=False) if match.group(1) == "q" else _text(value)
        )

    return _VERB.sub(verb, fmt)


_FUNCS: dict[str, Callable[..., Any]] = {
    "eq": _eq,
    "ne": lambda a, b: a != b,
    "lt": _ordered(lambda a, b: a < b),
    "le": _ordered(lambda a, b: a <= b),
    "gt": _ordered(lambda a, b: a > b),
    "ge": _ordered(lambda a, b: a >= b),
    "and": _and,
    "or": _or,
    "not": lambda a: not _truthy(a),
    "len": _len,
    "index": _index,
    "slice": _slice,
    "json": _json,
    "print": _print,
    "printf": _printf,
}


def _text(value: Any) -> str:
    """How an action prints ``value``: nothing for a missing one."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return str(value)


class _Scope:
    def __init__(self, parent: _Scope | None = None) -> None:
        self.vars: dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]
            scope = scope.parent
        raise GoTemplateError(f"undefined variable {name}")

    def set(self, name: str, value: Any) -> None:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.vars:
                scope.vars[name] = value
                return
            scope = scope.parent
        raise GoTemplateError(f"undefined variable {name}")


class _Renderer:
    def __init__(self) -> None:
        self.out: list[str] = []
        self.size = 0
        self.loops = 0  # ranges being rendered: where break/continue may be
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise GoTemplateError("template does too much work")

    def emit(self, text: str) -> None:
        self.size += len(text)
        if self.size > MAX_OUTPUT:
            raise GoTemplateError("output too large")
        self.out.append(text)

    def operand(self, op: _Operand, dot: Any, scope: _Scope) -> Any:
        if op.kind == "field":
            return _resolve_field(dot, op.path)
        if op.kind == "var":
            return _resolve_field(scope.get(op.name), op.path)
        if op.kind == "lit":
            return op.value
        if op.kind == "sub":
            assert op.sub is not None
            return self.pipeline(op.sub, dot, scope)
        raise GoTemplateError(f"{op.name} is a function, not a value")

    def command(self, cmd: list[_Operand], dot: Any, scope: _Scope, piped: tuple) -> Any:
        head = cmd[0]
        if head.kind == "ident":
            args = [self.operand(op, dot, scope) for op in cmd[1:]] + list(piped)
            try:
                return _FUNCS[head.name](*args)
            except TypeError as exc:
                raise GoTemplateError(f"{head.name}: {exc}") from exc
        if len(cmd) > 1 or piped:
            raise GoTemplateError("only a function takes arguments")
        return self.operand(head, dot, scope)

    def pipeline(self, pipe: _Pipeline, dot: Any, scope: _Scope) -> Any:
        value: Any = None
        for i, cmd in enumerate(pipe.cmds):
            value = self.command(cmd, dot, scope, (value,) if i else ())
        return value

    def bind(self, pipe: _Pipeline, value: Any, scope: _Scope) -> None:
        if len(pipe.decl) != 1:
            raise GoTemplateError("too many variables")
        if pipe.assign:
            scope.set(pipe.decl[0], value)
        else:
            scope.vars[pipe.decl[0]] = value

    def render(self, node: _Node, dot: Any, scope: _Scope) -> None:
        self.step()
        if isinstance(node, _Block):
            inner = _Scope(scope)
            for child in node.children:
                self.render(child, dot, inner)
        elif isinstance(node, _Literal):
            self.emit(node.text)
        elif isinstance(node, _Action):
            value = self.pipeline(node.pipe, dot, scope)
            if node.pipe.decl:
                self.bind(node.pipe, value, scope)
            else:
                self.emit(_text(value))
        elif isinstance(node, _If):
            for pipe, block in node.branches:
                if _truthy(self.pipeline(pipe, dot, scope)):
                    self.render(block, dot, scope)
                    return
            if node.else_ is not None:
                self.render(node.else_, dot, scope)
        elif isinstance(node, _With):
            value = self.pipeline(node.pipe, dot, scope)
            if _truthy(value):
                inner = _Scope(scope)
                if node.pipe.decl:
                    inner.vars[node.pipe.decl[0]] = value
                self.render(node.body, value, inner)
            elif node.else_ is not None:
                self.render(node.else_, dot, scope)
        elif isinstance(node, _Range):
            self.range(node, dot, scope)
        elif isinstance(node, _Jump):
            if not self.loops:
                raise GoTemplateError(f"{{{{ {node.kind} }}}} outside range")
            raise _Break() if node.kind == "break" else _Continue()

    def range(self, node: _Range, dot: Any, scope: _Scope) -> None:
        value = self.pipeline(node.pipe, dot, scope)
        items: Iterable[tuple[Any, Any]]
        if isinstance(value, Mapping):
            items = [(k, value[k]) for k in sorted(value)]
        elif isinstance(value, (list, tuple)):
            items = list(enumerate(value))
        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            # Lazily: the count is the template's (an integer literal), not data.
            items = ((i, i) for i in range(value)) if value else []
        elif value is None:
            items = []
        else:
            raise GoTemplateError(f"range over {type(value).__name__}")
        if isinstance(items, list) and not items:
            if node.else_ is not None:
                self.render(node.else_, dot, scope)
            return
        names = node.pipe.decl
        if len(names) > 2:
            raise GoTemplateError("too many variables in range")
        for key, item in items:
            self.step()
            inner = _Scope(scope)
            if len(names) == 1:
                inner.vars[names[0]] = item
            elif len(names) == 2:
                inner.vars[names[0]], inner.vars[names[1]] = key, item
            self.loops += 1
            try:
                self.render(node.body, item, inner)
            except _Continue:
                continue
            except _Break:
                break
            finally:
                self.loops -= 1


# ----------------------------------------------------------------------
# Public entrypoints
# ----------------------------------------------------------------------


def _mentions_response(pipe: _Pipeline) -> bool:
    return any(
        (op.kind == "field" and "Response" in op.path)
        or (op.kind == "sub" and op.sub is not None and _mentions_response(op.sub))
        for cmd in pipe.cmds
        for op in cmd
    )


def _cut_after_response(block: _Block) -> _Block:
    """``block`` up to and including the first action that prints a
    ``.Response`` field, nothing after it — Ollama's ``deleteNode`` for the
    last turn of a template without ``.Messages``: the prompt ends where the
    model's answer would go. Branch conditions are not looked into, as there."""
    cut = False

    def walk(node: _Node) -> _Node | None:
        nonlocal cut
        if cut:
            return None
        if isinstance(node, _Block):
            kept = [n for n in (walk(c) for c in node.children) if n is not None]
            return _Block(kept)
        if isinstance(node, _Action):
            if _mentions_response(node.pipe):
                cut = True
            return node
        if isinstance(node, _If):
            branches = []
            for pipe, body in node.branches:
                walked = walk(body)
                branches.append((pipe, walked if isinstance(walked, _Block) else _Block()))
            other = walk(node.else_) if node.else_ is not None else None
            return _If(branches, other if isinstance(other, _Block) else None)
        if isinstance(node, (_Range, _With)):
            inner = walk(node.body)
            rest = walk(node.else_) if node.else_ is not None else None
            return type(node)(
                node.pipe,
                inner if isinstance(inner, _Block) else _Block(),
                rest if isinstance(rest, _Block) else None,
            )
        return node

    walked = walk(block)
    return walked if isinstance(walked, _Block) else _Block()


def render_strict(source: str, data: Any, *, cut_after_response: bool = False) -> str:
    """Render ``source`` against ``data``; :class:`GoTemplateError` when it
    cannot be parsed or evaluated, too deep or too large.
    ``cut_after_response``: see :func:`_cut_after_response`."""
    try:
        tree = _parse(source)
        if cut_after_response:
            tree = _cut_after_response(tree)
        renderer = _Renderer()
        root = _Scope()
        root.vars["$"] = data
        renderer.render(tree, data, root)
        return "".join(renderer.out)
    except RecursionError as exc:
        raise GoTemplateError("nesting too deep") from exc


def render_go_template(source: str, data: Any) -> str:
    """Render a Go-template string against ``data``.

    ``data`` is typically a dict with string keys matching the Modelfile
    conventions (``Prompt``, ``System``, ``Messages``, ``Response``, …).
    Missing fields render as empty strings; falsey fields inside
    ``{{ if }}`` go down the ``else`` branch.

    On any error the original ``source`` is returned unmodified and a
    WARNING is logged — here the renderer is a convenience layer, never a
    gate (a crafted, deeply nested template must not become a 500).
    """
    try:
        return render_strict(source, data)
    except GoTemplateError as exc:
        logger.warning("Go-template render failed, falling back to literal: %s", exc)
        return source
