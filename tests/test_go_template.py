# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the minimal Go-template renderer (Phase 11 P1 — V2 row 23)."""

from __future__ import annotations

from hfl.converter.go_template import render_go_template


class TestFieldSubstitution:
    def test_simple_field(self):
        assert render_go_template("Hello {{ .Name }}!", {"Name": "world"}) == "Hello world!"

    def test_nested_field(self):
        assert render_go_template("{{ .User.Name }}", {"User": {"Name": "ada"}}) == "ada"

    def test_missing_field_renders_empty(self):
        assert render_go_template("Hi {{ .Missing }}", {}) == "Hi "

    def test_plain_text_passthrough(self):
        assert render_go_template("no placeholders here", {}) == "no placeholders here"

    def test_multiple_fields(self):
        tmpl = "{{ .A }} and {{ .B }}"
        assert render_go_template(tmpl, {"A": "x", "B": "y"}) == "x and y"


class TestWhitespaceTrim:
    def test_left_trim_removes_preceding_whitespace(self):
        tmpl = "foo\n  {{- .X }}"
        assert render_go_template(tmpl, {"X": "bar"}) == "foobar"

    def test_right_trim_removes_following_whitespace(self):
        tmpl = "{{ .X -}}   bar"
        assert render_go_template(tmpl, {"X": "foo"}) == "foobar"

    def test_both_sides_trim(self):
        tmpl = "  {{- .X -}}  "
        assert render_go_template(tmpl, {"X": "y"}) == "y"


class TestConditionals:
    def test_if_true(self):
        tmpl = "{{ if .Flag }}YES{{ end }}"
        assert render_go_template(tmpl, {"Flag": True}) == "YES"

    def test_if_false_renders_empty(self):
        tmpl = "{{ if .Flag }}YES{{ end }}"
        assert render_go_template(tmpl, {"Flag": False}) == ""

    def test_if_missing_field_is_falsy(self):
        tmpl = "{{ if .Flag }}YES{{ end }}"
        assert render_go_template(tmpl, {}) == ""

    def test_if_else(self):
        tmpl = "{{ if .X }}A{{ else }}B{{ end }}"
        assert render_go_template(tmpl, {"X": ""}) == "B"
        assert render_go_template(tmpl, {"X": "hi"}) == "A"

    def test_if_zero_int_is_falsy(self):
        tmpl = "{{ if .N }}non-zero{{ end }}"
        assert render_go_template(tmpl, {"N": 0}) == ""
        assert render_go_template(tmpl, {"N": 1}) == "non-zero"


class TestRange:
    def test_range_over_list_of_dicts(self):
        tmpl = "{{ range .Messages }}[{{ .Role }}:{{ .Content }}]{{ end }}"
        data = {
            "Messages": [
                {"Role": "user", "Content": "hi"},
                {"Role": "assistant", "Content": "hey"},
            ]
        }
        assert render_go_template(tmpl, data) == "[user:hi][assistant:hey]"

    def test_range_over_empty_list_renders_nothing(self):
        tmpl = "{{ range .Messages }}X{{ end }}"
        assert render_go_template(tmpl, {"Messages": []}) == ""

    def test_range_over_missing_field_renders_nothing(self):
        tmpl = "{{ range .Messages }}X{{ end }}"
        assert render_go_template(tmpl, {}) == ""

    def test_range_inside_text(self):
        tmpl = "Items: {{ range .Xs }}{{ . }} {{ end }}done"
        data = {"Xs": ["a", "b", "c"]}
        assert render_go_template(tmpl, data) == "Items: a b c done"


class TestStringLiteral:
    def test_string_literal_passthrough(self):
        tmpl = '{{ "hello" }} world'
        assert render_go_template(tmpl, {}) == "hello world"


class TestParseFailureFallback:
    def test_parse_error_returns_original(self):
        # Unbalanced action: no closing ``end``.
        tmpl = "{{ if .X }}forever"
        # Renderer does not raise — caller sees the raw template.
        out = render_go_template(tmpl, {"X": True})
        assert "forever" in out


class TestRealisticTemplates:
    def test_llama3_style(self):
        tmpl = (
            "<|begin_of_text|>"
            "{{ range .Messages }}"
            "<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>"
            "{{ end }}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        data = {
            "Messages": [
                {"Role": "system", "Content": "You are helpful."},
                {"Role": "user", "Content": "Hi"},
            ]
        }
        rendered = render_go_template(tmpl, data)
        assert "system" in rendered
        assert "You are helpful." in rendered
        assert "Hi" in rendered
        assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_gemma4_style_with_system(self):
        tmpl = (
            "{{ if .System }}<start_of_turn>system\n{{ .System }}<end_of_turn>\n{{ end }}"
            "<start_of_turn>user\n{{ .Prompt }}<end_of_turn>\n"
            "<start_of_turn>assistant\n"
        )
        rendered = render_go_template(
            tmpl,
            {"System": "Answer tersely.", "Prompt": "2+2?"},
        )
        assert "system" in rendered
        assert "Answer tersely." in rendered
        assert "2+2?" in rendered

    def test_gemma4_style_no_system(self):
        tmpl = "{{ if .System }}SYSTEM:{{ .System }}\n{{ end }}USER:{{ .Prompt }}\nASSISTANT:"
        rendered = render_go_template(tmpl, {"Prompt": "hi"})
        assert "SYSTEM" not in rendered
        assert rendered == "USER:hi\nASSISTANT:"
