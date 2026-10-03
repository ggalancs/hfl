# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Go-template renderer: the error paths, built-in functions and control
flow the main suite does not reach. Every error case is checked through
``render_strict`` (it raises) and, for a few, through the lenient
``render_go_template`` (it falls back to the literal source)."""

from __future__ import annotations

import logging

import pytest

from hfl.converter import go_template as gt
from hfl.converter.go_template import GoStruct, GoTemplateError, render_go_template, render_strict


def r(src, data=None, **kw):
    return render_strict(src, {} if data is None else data, **kw)


class TestParseErrors:
    @pytest.mark.parametrize(
        "src, message",
        [
            ("{{ .A # }}", "cannot read"),
            ("{{ .A ) }}", "unexpected"),
            ("{{ .A | }}", "empty command"),
            ("{{ (.A }}", "unclosed ("),
            ("{{ .A := 1 }}", "unexpected"),
            ("{{ nosuchfn 1 }}", "not defined"),
            ("{{ else }}", "unexpected {{ else }}"),
            ("{{ end }}", "unexpected {{ end }}"),
            ("{{ range .X }}body", "range without end"),
            ("{{ with .X }}body", "with without end"),
            ('{{ define "x" }}{{ end }}', "is not supported"),
            ('{{ template "x" }}', "is not supported"),
            ("{{ if .X }}a", "if without end"),
            ("{{ if .X }}a{{ else }}b", "if without end"),
            ("{{ if .X }}a{{ else if .Y }}b", "if without end"),
        ],
    )
    def test_strict_raises(self, src, message):
        with pytest.raises(GoTemplateError, match=message.replace("(", r"\(")):
            r(src)

    @pytest.mark.parametrize(
        "src", ["{{ with .X }}body{{ else }}fallback", "{{ range .L }}x{{ else }}empty"]
    )
    def test_unclosed_else_of_range_and_with(self, src):
        with pytest.raises(GoTemplateError, match="without end"):
            r(src)

    def test_lenient_returns_source_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hfl.converter.go_template"):
            assert render_go_template("{{ if .X }}never closed", {"X": 1}) == (
                "{{ if .X }}never closed"
            )
        assert "falling back to literal" in caplog.text

    def test_comment_is_skipped(self):
        assert r("a{{/* note */}}b") == "ab"

    def test_break_word_with_arguments_is_an_action(self):
        # "break" followed by something is not the break keyword -> a field/ident
        with pytest.raises(GoTemplateError, match="not defined"):
            r("{{ range .L }}{{ break .X }}{{ end }}", {"L": [1]})


class TestLiteralsAndVariables:
    def test_literal_kinds(self):
        assert r('{{ "a\\tb" }}') == "a\tb"
        assert r("{{ `raw\\n` }}") == "raw\\n"
        assert r("{{ 3 }}|{{ -2.5 }}") == "3|-2.5"
        assert r("{{ true }}|{{ false }}|{{ nil }}") == "true|false|"

    def test_declare_assign_and_root(self):
        src = '{{ $x := "a" }}{{ $x }}{{ if true }}{{ $x = "b" }}{{ end }}{{ $x }}{{ $.Top }}'
        assert r(src, {"Top": "T"}) == "abT"

    def test_assigning_undeclared_variable_fails(self):
        with pytest.raises(GoTemplateError, match="undefined variable \\$y"):
            r('{{ $y = "b" }}')
        with pytest.raises(GoTemplateError, match="undefined variable \\$z"):
            r("{{ $z }}")

    def test_two_variables_outside_range(self):
        with pytest.raises(GoTemplateError, match="too many variables"):
            r("{{ $a, $b := 1 }}")

    def test_function_name_used_as_value(self):
        with pytest.raises(GoTemplateError, match="is a function"):
            r("{{ print len }}")

    def test_value_with_arguments(self):
        with pytest.raises(GoTemplateError, match="only a function takes arguments"):
            r("{{ .A .B }}", {"A": 1})
        with pytest.raises(GoTemplateError, match="only a function takes arguments"):
            r("{{ 1 | .A }}", {"A": 1})

    def test_wrong_arity_is_template_error(self):
        with pytest.raises(GoTemplateError, match="not:"):
            r("{{ not 1 2 }}")


class _Obj:
    def __init__(self):
        self.Name = "obj"
        self.method = lambda: "x"

    def Method(self):
        return "called"


class TestFieldResolution:
    def test_none_midway_and_primitives_and_callables(self):
        assert r("{{ .A.B.C }}", {"A": None}) == ""
        assert r("{{ .S.upper }}", {"S": "str"}) == ""
        assert r("{{ .O.Name }}", {"O": _Obj()}) == "obj"
        assert r("{{ .O.Method }}", {"O": _Obj()}) == ""
        assert r("{{ .O.method }}", {"O": _Obj()}) == ""
        assert r("{{ .O.__class__ }}", {"O": _Obj()}) == ""

    def test_dot_and_text_of_containers(self):
        assert r("{{ . }}", {"a": [1, "<"]}) == '{"a":[1,"\\u003c"]}'
        assert r("{{ .L }}", {"L": (1, 2)}) == "[1,2]"


class TestTruthiness:
    @pytest.mark.parametrize(
        "value, expected",
        [(None, "F"), (0, "F"), (0.0, "F"), ("", "F"), ([], "F"), (set(), "F"), (1.5, "T")],
    )
    def test_values(self, value, expected):
        assert r("{{ if .V }}T{{ else }}F{{ end }}", {"V": value}) == expected

    def test_arbitrary_object_is_truthy(self):
        assert r("{{ if .V }}T{{ end }}", {"V": _Obj()}) == "T"


class TestFunctions:
    def test_eq_ne_and_ordering(self):
        assert r('{{ eq .A "x" "y" }}', {"A": "y"}) == "true"
        assert r("{{ ne 1 2 }}|{{ lt 1 2 }}|{{ le 2 2 }}|{{ gt 1 2 }}|{{ ge 3 2 }}") == (
            "true|true|true|false|true"
        )

    def test_eq_needs_two(self):
        with pytest.raises(GoTemplateError, match="eq needs two"):
            r("{{ eq 1 }}")

    def test_ordering_mixed_types(self):
        with pytest.raises(GoTemplateError, match="cannot compare str and int"):
            r('{{ lt "a" 1 }}')

    def test_and_or_return_operands(self):
        assert r('{{ and 1 "" 3 }}|{{ and 1 2 }}|{{ or 0 "" }}|{{ or 0 "x" }}') == "|2||x"
        assert gt._and() is None and gt._or() is None

    def test_len(self):
        assert r("{{ len .L }}|{{ len .M }}|{{ len .S }}", {"L": [1, 2], "M": {}, "S": "abc"}) == (
            "2|0|3"
        )
        with pytest.raises(GoTemplateError, match="len of int"):
            r("{{ len 5 }}")

    def test_index(self):
        data = {"L": [[1, 2], [3, 4]], "M": {"k": {"j": "v"}}}
        assert r('{{ index .L 1 0 }}|{{ index .L -1 1 }}|{{ index .M "k" "j" }}', data) == ("3|4|v")
        with pytest.raises(GoTemplateError, match="index 5 out of range"):
            r("{{ index .L 5 }}", data)
        with pytest.raises(GoTemplateError, match="cannot index that"):
            r('{{ index .M "_private" }}', data)
        with pytest.raises(GoTemplateError, match="cannot index that"):
            r('{{ index .L "x" }}', data)

    def test_slice(self):
        data = {"L": [1, 2, 3], "S": "abcd"}
        assert r("{{ slice .L 1 }}|{{ slice .S 1 3 }}|{{ slice .L }}", data) == "[2,3]|bc|[1,2,3]"
        for src, msg in [
            ("{{ slice 5 }}", "cannot slice that"),
            ("{{ slice .L 0 1 2 }}", "cannot slice that"),
            ("{{ slice .L 4 }}", "out of range"),
            ("{{ slice .L 2 1 }}", "out of range"),
        ]:
            with pytest.raises(GoTemplateError, match=msg):
                r(src, data)

    def test_json_wire_and_html_safety(self):
        call = GoStruct({"Function": {"Name": "f"}}, {"function": {"name": "f"}})
        assert r("{{ .C.Function.Name }}", {"C": call}) == "f"
        assert r("{{ json .Calls }}", {"Calls": [call, {"x": "a&b>"}]}) == (
            '[{"function":{"name":"f"}},{"x":"a\\u0026b\\u003e"}]'
        )

    def test_print_and_printf(self):
        assert r('{{ print "a" 1 true nil }}') == "a1true"
        assert r('{{ printf "%s=%d %q 100%% %v" "k" 3 "q" }}') == 'k=3 "q" 100% %!v(MISSING)'
        assert r('{{ "x" | printf "<%s>" }}') == "<x>"
        with pytest.raises(GoTemplateError, match="format string"):
            r("{{ printf 3 }}")

    def test_subexpression(self):
        assert r("{{ len (slice .L 1) }}", {"L": [1, 2, 3]}) == "2"


class TestControlFlow:
    def test_else_if_chain(self):
        src = "{{ if eq .N 1 }}one{{ else if eq .N 2 }}two{{ else }}many{{ end }}"
        assert [r(src, {"N": n}) for n in (1, 2, 3)] == ["one", "two", "many"]

    def test_if_without_else_false(self):
        assert r("{{ if .X }}a{{ else if .Y }}b{{ end }}") == ""

    def test_with_binds_dot_and_variable(self):
        src = "{{ with $u := .User }}{{ .Name }}/{{ $u.Name }}{{ else }}nobody{{ end }}"
        assert r(src, {"User": {"Name": "ada"}}) == "ada/ada"
        assert r(src, {}) == "nobody"
        assert r("{{ with .U }}x{{ end }}", {}) == ""

    def test_range_over_map_sorted_with_key_value(self):
        src = "{{ range $k, $v := .M }}{{ $k }}={{ $v }};{{ end }}"
        assert r(src, {"M": {"b": 2, "a": 1}}) == "a=1;b=2;"

    def test_range_over_int_and_none_and_else(self):
        assert r("{{ range 3 }}{{ . }}{{ end }}") == "012"
        assert r("{{ range 0 }}x{{ else }}empty{{ end }}") == "empty"
        assert r("{{ range .Missing }}x{{ else }}none{{ end }}") == "none"
        assert r("{{ range .L }}x{{ end }}", {"L": []}) == ""

    def test_range_over_unsupported_value(self):
        with pytest.raises(GoTemplateError, match="range over str"):
            r("{{ range .S }}{{ end }}", {"S": "abc"})
        with pytest.raises(GoTemplateError, match="range over bool"):
            r("{{ range true }}{{ end }}")

    def test_range_with_three_variables(self):
        with pytest.raises(GoTemplateError, match="too many variables in range"):
            r("{{ range $a, $b, $c := .L }}{{ end }}", {"L": [1]})

    def test_range_with_one_variable_keeps_outer_dot(self):
        src = "{{ range $m := .L }}{{ $m }}{{ $.Sep }}{{ end }}"
        assert r(src, {"L": ["a", "b"], "Sep": ","}) == "a,b,"

    def test_break_and_continue(self):
        src = (
            "{{ range .L }}{{ if eq . 2 }}{{ continue }}{{ end }}"
            "{{ if eq . 4 }}{{ break }}{{ end }}{{ . }}{{ end }}"
        )
        assert r(src, {"L": [1, 2, 3, 4, 5]}) == "13"

    def test_break_outside_range(self):
        with pytest.raises(GoTemplateError, match="outside range"):
            r("{{ break }}")

    def test_step_and_output_caps(self, monkeypatch):
        monkeypatch.setattr(gt, "MAX_STEPS", 50)
        with pytest.raises(GoTemplateError, match="too much work"):
            r("{{ range 1000 }}{{ end }}")
        monkeypatch.setattr(gt, "MAX_OUTPUT", 10)
        with pytest.raises(GoTemplateError, match="output too large"):
            r("{{ range 3 }}abcd{{ end }}")

    def test_deep_nesting_is_an_error(self):
        src = "{{ if true }}" * 2000 + "x" + "{{ end }}" * 2000
        with pytest.raises(GoTemplateError, match="nesting too deep"):
            r(src)


class TestCutAfterResponse:
    def test_cut_inside_branches_and_loops(self):
        src = (
            "{{ range .Msgs }}[{{ . }}]{{ else }}none{{ end }}"
            "{{ with .U }}<{{ . }}>{{ end }}"
            "{{ if .Sys }}S{{ else }}{{ (print .Response) }}after-else{{ end }}"
            "TAIL{{ .Prompt }}"
        )
        data = {"Msgs": ["m"], "U": "u", "Sys": "", "Response": "R", "Prompt": "P"}
        assert r(src, data) == "[m]<u>Rafter-elseTAILP"
        # Everything after the first action that prints .Response is gone,
        # including the rest of the branch and the outer tail.
        assert r(src, data, cut_after_response=True) == "[m]<u>R"

    def test_nodes_after_cut_inside_range_and_with_become_empty(self):
        src = "{{ .Response }}{{ range .L }}x{{ else }}y{{ end }}{{ with .U }}z{{ end }}"
        assert r(src, {"Response": "R", "L": [1], "U": 1}, cut_after_response=True) == "R"

    def test_range_body_with_response_cuts_its_else(self):
        src = "{{ range .L }}{{ .Response }}more{{ else }}empty{{ end }}after"
        data = {"L": [{"Response": "a"}, {"Response": "b"}]}
        assert r(src, data, cut_after_response=True) == "ab"
        assert r(src, {"L": []}, cut_after_response=True) == ""

    def test_without_response_nothing_is_cut(self):
        src = "{{ if .A }}a{{ end }}{{ .B }}"
        assert r(src, {"A": 1, "B": "b"}, cut_after_response=True) == "ab"
