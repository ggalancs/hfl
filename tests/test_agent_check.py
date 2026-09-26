# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``scripts/agent_check.py`` judges an agent by running its work."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "agent_check", Path(__file__).resolve().parents[1] / "scripts" / "agent_check.py"
)
ac = importlib.util.module_from_spec(_SPEC)
sys.modules["agent_check"] = ac
_SPEC.loader.exec_module(ac)


def test_the_planted_bug_is_not_fixed_until_it_is(tmp_path):
    (tmp_path / "calc.py").write_text(ac.BUGGY)
    assert ac._fixed(tmp_path) is False
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    assert ac._fixed(tmp_path) is True


def test_leaked_markup_is_seen():
    assert ac.MARKUP.search("Done.\n<tool_call>")
    assert ac.MARKUP.search("<|channel|>final")
    assert not ac.MARKUP.search("I fixed calc.py: add now returns a + b.")
