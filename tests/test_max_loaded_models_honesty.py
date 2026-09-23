# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`HFL_MAX_LOADED_MODELS` is read, documented, and does nothing.

`docs/env-vars.md` described it as "models kept resident in `ModelPool`
(LRU eviction)". `config` reads it. `ModelPool` implements LRU
multi-residency, with tests. And the server never instantiates
`ModelPool`, so exactly one model is resident whatever the operator sets.

Honouring it is not a wiring job. Multiple resident models touch the
model-lifecycle use-after-free family, the dispatcher's single-slot
assumption and `ServerState`'s single `_engine` — the June architecture
review named that combination "a different product", to start only with
explicit owner direction. So this is left unimplemented on purpose.

What is NOT acceptable is the silence. A knob that accepts a value and
discards it is worse than no knob, because the operator believes they
configured something and plans capacity around it. The server now says
so at startup and the documentation says so in the table.

These tests guard the saying-so, which is the part that was missing.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs" / "env-vars.md"


class TestTheServerAdmitsIt:
    @staticmethod
    def _lifespan_tree() -> ast.Module:
        from hfl.api import server

        return ast.parse(textwrap.dedent(inspect.getsource(server.lifespan)))

    def test_startup_reads_the_setting(self):
        source = inspect.getsource(__import__("hfl.api.server", fromlist=["x"]).lifespan)
        assert "max_loaded_models" in source, (
            "the server does not look at the setting at all, so an operator "
            "who sets it gets no signal of any kind"
        )

    def test_a_value_above_one_warns(self):
        """Asserted over the AST: there must be a branch on the value.

        A comment explaining the situation is not a warning; the operator
        never reads the source.
        """
        tree = self._lifespan_tree()
        compares = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Compare)
            and isinstance(n.ops[0], ast.Gt)
            and any(isinstance(c, ast.Constant) and c.value == 1 for c in n.comparators)
        ]
        assert compares, (
            "nothing branches on max_loaded_models > 1, so the setting is "
            "still accepted and silently dropped"
        )

        warns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "warning"
        ]
        assert warns, "the branch exists but emits no warning"

    def test_the_default_says_nothing(self):
        """One resident model is the normal case; it must stay quiet.

        A warning that fires for everybody is noise, and noise is what
        makes the real warnings unreadable.
        """
        tree = self._lifespan_tree()
        source = ast.unparse(tree)
        assert "> 1" in source, (
            "the warning is not gated on a value above 1, so every default start-up would print it"
        )


class TestTheDocumentationStoppedPromising:
    def test_the_table_no_longer_claims_lru_residency(self):
        row = next(
            line
            for line in DOCS.read_text(encoding="utf-8").splitlines()
            if "HFL_MAX_LOADED_MODELS" in line
        )
        assert "not yet honoured" in row.lower(), (
            "docs/env-vars.md still presents the setting as working, which is "
            "where the false expectation came from in the first place"
        )

    def test_the_row_says_what_actually_happens(self):
        row = next(
            line
            for line in DOCS.read_text(encoding="utf-8").splitlines()
            if "HFL_MAX_LOADED_MODELS" in line
        )
        assert "evict" in row.lower(), (
            "telling an operator the knob does nothing without telling them "
            "what the server does instead leaves them no better off"
        )


class TestTheUnderlyingStateIsUnchanged:
    def test_model_pool_is_still_not_instantiated_by_the_server(self):
        """If somebody wires it, this fails and these tests are obsolete.

        Deliberate: the day multi-residency lands, the warning and the
        documentation row both become lies in the other direction.
        """
        import hfl.api as api_pkg

        pattern = re.compile(r"\bModelPool\s*\(")
        hits = [
            path.name
            for path in Path(api_pkg.__file__).parent.rglob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert not hits, (
            f"{hits} now construct a ModelPool. Multi-residency may be real: "
            "remove the startup warning, update docs/env-vars.md, and delete "
            "this file."
        )
