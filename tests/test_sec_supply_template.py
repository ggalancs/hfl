# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""A per-request Go template cannot exhaust memory or CPU with ranges that
print nothing (MAX_OUTPUT never trips on them)."""

from __future__ import annotations

import time
import tracemalloc

import pytest

from hfl.converter.go_template import GoTemplateError, render_go_template, render_strict


def _peak_mb(source: str) -> float:
    tracemalloc.start()
    try:
        with pytest.raises(GoTemplateError):
            render_strict(source, {})
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def test_a_range_over_a_big_integer_literal_allocates_nothing_for_it():
    # Before: a 3M-item list (~319 MB) built before the first iteration.
    assert _peak_mb("{{ range 3000000 }}{{ end }}") < 20


def test_a_range_over_a_huge_integer_is_cut_short():
    started = time.monotonic()
    with pytest.raises(GoTemplateError):
        render_strict("{{ range 9999999999 }}{{ end }}", {})
    assert time.monotonic() - started < 10


def test_nested_ranges_that_print_nothing_are_cut_short():
    started = time.monotonic()
    with pytest.raises(GoTemplateError):
        render_strict("{{ range 5000 }}{{ range 5000 }}{{ end }}{{ end }}", {})
    assert time.monotonic() - started < 10
    # /api/generate's renderer falls back to the literal source, not a 500.
    source = "{{ range 9999999999 }}{{ end }}"
    assert render_go_template(source, {}) == source


def test_real_templates_still_render():
    messages = [{"Role": "user", "Content": f"m{i}"} for i in range(2000)]
    out = render_strict(
        "{{ range $i, $m := .Messages }}<{{ $m.Role }}>{{ $m.Content }}{{ end }}",
        {"Messages": messages},
    )
    assert out.count("<user>") == 2000
    assert render_strict("{{ range 3 }}{{ . }}{{ end }}", {}) == "012"
    assert render_strict("{{ range 0 }}x{{ else }}none{{ end }}", {}) == "none"
    assert (
        render_strict("{{ range 5 }}{{ if eq . 2 }}{{ break }}{{ end }}{{ . }}{{ end }}", {})
        == "01"
    )
