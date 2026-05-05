# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the V4 F5 ``DRAFT`` Modelfile directive."""

from __future__ import annotations

from hfl.converter.modelfile_parser import parse_modelfile


class TestDraftDirective:
    def test_bare_path(self):
        text = """\
FROM /tmp/llama-70b.gguf
DRAFT /tmp/llama-1b-q4.gguf
"""
        doc = parse_modelfile(text)
        assert doc.draft == "/tmp/llama-1b-q4.gguf"

    def test_quoted_path(self):
        text = """\
FROM /tmp/llama-70b.gguf
DRAFT "/tmp/with spaces/llama-1b-q4.gguf"
"""
        doc = parse_modelfile(text)
        assert doc.draft == "/tmp/with spaces/llama-1b-q4.gguf"

    def test_repo_id_value(self):
        text = """\
FROM Qwen/Qwen2.5-72B-Instruct
DRAFT Qwen/Qwen2.5-1.5B-Instruct
"""
        doc = parse_modelfile(text)
        assert doc.draft == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_last_wins(self):
        text = """\
FROM /tmp/x.gguf
DRAFT /first.gguf
DRAFT /second.gguf
"""
        doc = parse_modelfile(text)
        assert doc.draft == "/second.gguf"

    def test_omitted_keeps_none(self):
        text = "FROM /tmp/x.gguf\n"
        doc = parse_modelfile(text)
        assert doc.draft is None

    def test_to_manifest_fields_includes_draft_model_path(self):
        text = """\
FROM /tmp/x.gguf
DRAFT /tmp/draft.gguf
"""
        doc = parse_modelfile(text)
        fields = doc.to_manifest_fields()
        assert fields["draft_model_path"] == "/tmp/draft.gguf"
