# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The audit harness's own logic (audit/local_audit.py): which checks run, in
what order, and how a verdict is reached. The checks themselves run the real
HFL and are not exercised here."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))

import local_audit as la  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def checks() -> None:
    if not la.REGISTRY:
        la.load_checks()


def test_every_check_has_a_unique_id_in_its_section() -> None:
    ids = [c.cid for c in la.REGISTRY]
    assert [cid for cid, n in Counter(ids).items() if n > 1] == []
    assert {cid[0] for cid in ids} == set("ABCDEF")


def test_every_need_names_a_registered_check() -> None:
    known = {c.cid for c in la.REGISTRY}
    assert [(c.cid, n) for c in la.REGISTRY for n in c.needs if n not in known] == []


def test_a_check_brings_what_it_needs_first() -> None:
    assert [c.cid for c in la.selected("E8")] == ["B32", "B54", "E8"]
    f6 = [c.cid for c in la.selected("F6")]
    assert f6[-1] == "F6" and len(f6) == 2 and f6[0].startswith("C")


def test_a_need_already_chosen_is_not_run_twice() -> None:
    ids = [c.cid for c in la.selected("B32,E8")]
    assert ids.count("B32") == 1


def test_a_section_letter_selects_the_whole_section() -> None:
    ids = [c.cid for c in la.selected("D")]
    assert ids and all(cid.startswith("D") for cid in ids)


def test_parts_lists_every_failure_not_just_the_first() -> None:
    part = la.Parts()
    part("one", lambda: la.expect(False, "first"))
    part("two", lambda: None)
    part("three", lambda: la.expect(False, "second"))
    with pytest.raises(la.Broken) as caught:
        part.verdict()
    assert "first" in str(caught.value) and "second" in str(caught.value)


def test_parts_never_counts_an_unchecked_part_as_fine() -> None:
    def unchecked() -> None:
        raise la.Uncheckable("no tool")

    only_unchecked = la.Parts()
    only_unchecked("a", unchecked)
    with pytest.raises(la.Uncheckable):
        only_unchecked.verdict()

    mixed = la.Parts()
    mixed("a", lambda: None)
    mixed("b", unchecked)
    assert "NOT checked: b (no tool)" in mixed.verdict()


def test_parts_false_is_a_failure() -> None:
    part = la.Parts()
    part("a", lambda: False)
    with pytest.raises(la.Broken):
        part.verdict()


def test_report_puts_what_is_not_ok_first(tmp_path: Path) -> None:
    results = {
        "A1": {"title": "hfl alias", "status": la.OK, "evidence": "fine"},
        "B3": {"title": "bench", "status": la.BROKEN, "evidence": "only done"},
    }
    text = la.report(tmp_path, results).read_text()
    assert text.index("## Not OK") < text.index("## A.")
    assert "| B3 | bench | ROTO | only done |" in text
