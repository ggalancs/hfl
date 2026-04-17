# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for Modelfile v2 directives (Phase 14 — V2 rows 19 / 21 / 22)."""

from __future__ import annotations

import pytest

from hfl.converter.modelfile_parser import ModelfileParseError, parse_modelfile


class TestENV:
    def test_simple(self):
        doc = parse_modelfile("FROM x\nENV HF_HOME=/opt/hf")
        assert doc.env == {"HF_HOME": "/opt/hf"}

    def test_quoted_value(self):
        doc = parse_modelfile('FROM x\nENV KEY="a b c"')
        assert doc.env == {"KEY": "a b c"}

    def test_multiple(self):
        doc = parse_modelfile("FROM x\nENV A=1\nENV B=2")
        assert doc.env == {"A": "1", "B": "2"}

    def test_missing_equals_rejected(self):
        with pytest.raises(ModelfileParseError):
            parse_modelfile("FROM x\nENV NOEQUALS")

    def test_empty_key_rejected(self):
        with pytest.raises(ModelfileParseError):
            parse_modelfile("FROM x\nENV =v")

    def test_propagates_to_manifest_fields(self):
        doc = parse_modelfile("FROM x\nENV HF_HOME=/a")
        fields = doc.to_manifest_fields()
        assert fields["env_vars"] == {"HF_HOME": "/a"}


class TestCAPABILITIES:
    def test_comma_separated(self):
        doc = parse_modelfile("FROM x\nCAPABILITIES completion,tools,vision")
        assert doc.capabilities == ["completion", "tools", "vision"]

    def test_space_separated(self):
        doc = parse_modelfile("FROM x\nCAPABILITIES completion tools vision")
        assert doc.capabilities == ["completion", "tools", "vision"]

    def test_dedupes(self):
        doc = parse_modelfile("FROM x\nCAPABILITIES tools,tools,embedding")
        assert doc.capabilities == ["tools", "embedding"]

    def test_propagates_to_manifest_fields(self):
        doc = parse_modelfile("FROM x\nCAPABILITIES completion,tools")
        fields = doc.to_manifest_fields()
        assert fields["declared_capabilities"] == ["completion", "tools"]

    def test_empty_capabilities_line_rejected(self):
        with pytest.raises(ModelfileParseError):
            parse_modelfile("FROM x\nCAPABILITIES")


class TestINCLUDE:
    def test_inlines_sibling(self, tmp_path):
        helper = tmp_path / "common.modelfile"
        helper.write_text('SYSTEM """shared"""\n')
        main = tmp_path / "main.modelfile"
        main.write_text("FROM x\nINCLUDE ./common.modelfile\n")
        doc = parse_modelfile(main.read_text(), base_path=tmp_path)
        assert doc.system == "shared"

    def test_nested_includes(self, tmp_path):
        (tmp_path / "inner.modelfile").write_text("PARAMETER num_ctx 2048\n")
        (tmp_path / "middle.modelfile").write_text("INCLUDE ./inner.modelfile\n")
        main = tmp_path / "main.modelfile"
        main.write_text("FROM x\nINCLUDE ./middle.modelfile\n")
        doc = parse_modelfile(main.read_text(), base_path=tmp_path)
        assert doc.parameters["num_ctx"] == 2048

    def test_cycle_detected(self, tmp_path):
        a = tmp_path / "a.modelfile"
        b = tmp_path / "b.modelfile"
        a.write_text("INCLUDE ./b.modelfile\n")
        b.write_text("INCLUDE ./a.modelfile\n")
        main = tmp_path / "main.modelfile"
        main.write_text("FROM x\nINCLUDE ./a.modelfile\n")
        with pytest.raises(ModelfileParseError) as exc:
            parse_modelfile(main.read_text(), base_path=tmp_path)
        assert "cycle" in str(exc.value).lower()

    def test_missing_target_rejected(self, tmp_path):
        main = tmp_path / "main.modelfile"
        main.write_text("FROM x\nINCLUDE ./ghost.modelfile\n")
        with pytest.raises(ModelfileParseError) as exc:
            parse_modelfile(main.read_text(), base_path=tmp_path)
        assert "missing" in str(exc.value).lower()

    def test_include_is_unknown_when_no_base_path(self):
        # Without an anchor, INCLUDE is just an unknown instruction.
        with pytest.raises(ModelfileParseError):
            parse_modelfile("FROM x\nINCLUDE something\n")

    def test_include_comment_tail_stripped(self, tmp_path):
        helper = tmp_path / "helper.modelfile"
        helper.write_text("PARAMETER num_ctx 4096\n")
        main = tmp_path / "main.modelfile"
        main.write_text("FROM x\nINCLUDE ./helper.modelfile  # inline doc\n")
        doc = parse_modelfile(main.read_text(), base_path=tmp_path)
        assert doc.parameters["num_ctx"] == 4096
