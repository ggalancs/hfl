# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for the Modelfile parser (Phase 6 P2-1 part 1).

The parser is the entry point for ``POST /api/create`` — every failure
mode here eventually renders as a 400 response. These tests pin down:

- the grammar (instructions, quotes, comments, escapes)
- value coercion (int/float/bool PARAMETER types)
- round-trip: parse → render → parse is an identity on documents
"""

from __future__ import annotations

import pytest

from hfl.converter.modelfile_parser import (
    BOOL_PARAMETERS,
    FLOAT_PARAMETERS,
    INT_PARAMETERS,
    ModelfileParseError,
    parse_modelfile,
    render_modelfile_document,
)

# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


class TestBasicGrammar:
    def test_minimal_modelfile_is_just_from(self):
        doc = parse_modelfile("FROM llama3.3")
        assert doc.from_ == "llama3.3"
        assert doc.template is None
        assert doc.system is None
        assert doc.parameters == {}

    def test_from_with_digest(self):
        doc = parse_modelfile("FROM sha256:abc123def456")
        assert doc.from_ == "sha256:abc123def456"

    def test_from_with_local_path(self):
        doc = parse_modelfile("FROM ./models/llama-3.3-q4.gguf")
        assert doc.from_ == "./models/llama-3.3-q4.gguf"

    def test_from_with_absolute_path(self):
        doc = parse_modelfile("FROM /home/user/models/phi-4.gguf")
        assert doc.from_ == "/home/user/models/phi-4.gguf"

    def test_lowercase_instruction(self):
        doc = parse_modelfile("from llama3.3")
        assert doc.from_ == "llama3.3"

    def test_mixed_case_instruction(self):
        doc = parse_modelfile("From llama3.3")
        assert doc.from_ == "llama3.3"

    def test_blank_lines_are_skipped(self):
        doc = parse_modelfile("\n\nFROM llama3.3\n\n\n")
        assert doc.from_ == "llama3.3"

    def test_full_line_comment_is_skipped(self):
        text = "# comment before\nFROM llama3.3\n# comment after"
        doc = parse_modelfile(text)
        assert doc.from_ == "llama3.3"

    def test_indented_instruction_is_allowed(self):
        doc = parse_modelfile("   FROM llama3.3")
        assert doc.from_ == "llama3.3"


# ----------------------------------------------------------------------
# Quoted values
# ----------------------------------------------------------------------


class TestQuoting:
    def test_triple_quoted_template_single_line(self):
        text = 'FROM llama3.3\nTEMPLATE """{{ .Prompt }}"""'
        doc = parse_modelfile(text)
        assert doc.template == "{{ .Prompt }}"

    def test_triple_quoted_template_multi_line(self):
        text = 'FROM llama3.3\nTEMPLATE """<|begin|>\n{{ .Prompt }}\n<|end|>"""'
        doc = parse_modelfile(text)
        assert doc.template == "<|begin|>\n{{ .Prompt }}\n<|end|>"

    def test_triple_quoted_system(self):
        text = 'FROM llama3.3\nSYSTEM """You are a helpful assistant."""'
        doc = parse_modelfile(text)
        assert doc.system == "You are a helpful assistant."

    def test_unterminated_triple_quote_fails(self):
        text = 'FROM llama3.3\nTEMPLATE """never closes'
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile(text)
        assert "unterminated" in str(exc_info.value).lower()
        assert exc_info.value.line == 2

    def test_single_quoted_stop(self):
        text = 'FROM llama3.3\nPARAMETER stop "<|eot_id|>"'
        doc = parse_modelfile(text)
        assert doc.stop_sequences == ["<|eot_id|>"]

    def test_escape_sequences_in_quoted_string(self):
        text = r"FROM llama3.3" + "\n" + r'PARAMETER stop "a\nb\tc\"d\\e"'
        doc = parse_modelfile(text)
        assert doc.stop_sequences == ['a\nb\tc"d\\e']


# ----------------------------------------------------------------------
# PARAMETER block
# ----------------------------------------------------------------------


class TestParameters:
    def test_int_parameter(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER num_ctx 4096")
        assert doc.parameters["num_ctx"] == 4096

    def test_float_parameter(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER temperature 0.7")
        assert doc.parameters["temperature"] == 0.7

    def test_bool_parameter_true(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER use_mmap true")
        assert doc.parameters["use_mmap"] is True

    def test_bool_parameter_false(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER use_mmap false")
        assert doc.parameters["use_mmap"] is False

    def test_multiple_stops(self):
        text = 'FROM llama3.3\nPARAMETER stop "<|eot_id|>"\nPARAMETER stop "<|end_of_text|>"'
        doc = parse_modelfile(text)
        assert doc.stop_sequences == ["<|eot_id|>", "<|end_of_text|>"]

    def test_unknown_parameter_is_preserved_as_string(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER weird_knob weird_value")
        assert doc.parameters["weird_knob"] == "weird_value"

    def test_int_parameter_with_bad_value_fails(self):
        text = "FROM llama3.3\nPARAMETER num_ctx notanumber"
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile(text)
        assert "int" in str(exc_info.value).lower()
        assert exc_info.value.line == 2

    def test_bool_parameter_with_bad_value_fails(self):
        text = "FROM llama3.3\nPARAMETER use_mmap maybe"
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile(text)
        assert "bool" in str(exc_info.value).lower()

    def test_parameter_without_value_fails(self):
        with pytest.raises(ModelfileParseError):
            parse_modelfile("FROM llama3.3\nPARAMETER")

    def test_parameter_key_is_lowercased(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER NUM_CTX 2048")
        assert doc.parameters["num_ctx"] == 2048


# ----------------------------------------------------------------------
# ADAPTER / LICENSE / MESSAGE / REQUIRES
# ----------------------------------------------------------------------


class TestAdapterLicenseMessageRequires:
    def test_adapter(self):
        doc = parse_modelfile("FROM llama3.3\nADAPTER ./lora/my-adapter.gguf")
        assert doc.adapters == ["./lora/my-adapter.gguf"]

    def test_multiple_adapters(self):
        text = "FROM llama3.3\nADAPTER a.gguf\nADAPTER b.gguf"
        doc = parse_modelfile(text)
        assert doc.adapters == ["a.gguf", "b.gguf"]

    def test_license_triple_quoted(self):
        text = 'FROM llama3.3\nLICENSE """Apache 2.0"""'
        doc = parse_modelfile(text)
        assert doc.license == "Apache 2.0"

    def test_message_user(self):
        text = 'FROM llama3.3\nMESSAGE user "Hello"'
        doc = parse_modelfile(text)
        assert len(doc.messages) == 1
        assert doc.messages[0].role == "user"
        assert doc.messages[0].content == "Hello"

    def test_message_invalid_role_fails(self):
        text = 'FROM llama3.3\nMESSAGE bogus "x"'
        with pytest.raises(ModelfileParseError):
            parse_modelfile(text)

    def test_multiple_messages_ordered(self):
        text = 'FROM llama3.3\nMESSAGE user "q1"\nMESSAGE assistant "a1"\nMESSAGE user "q2"'
        doc = parse_modelfile(text)
        assert [m.role for m in doc.messages] == ["user", "assistant", "user"]
        assert [m.content for m in doc.messages] == ["q1", "a1", "q2"]

    def test_requires(self):
        doc = parse_modelfile("FROM llama3.3\nREQUIRES >=0.6.0")
        assert doc.requires == ">=0.6.0"

    def test_requires_quoted_is_unwrapped(self):
        doc = parse_modelfile('FROM llama3.3\nREQUIRES ">=0.6.0"')
        assert doc.requires == ">=0.6.0"


# ----------------------------------------------------------------------
# Error conditions
# ----------------------------------------------------------------------


class TestErrors:
    def test_missing_from_fails(self):
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile("PARAMETER num_ctx 2048")
        assert "FROM" in str(exc_info.value)

    def test_empty_file_fails(self):
        with pytest.raises(ModelfileParseError):
            parse_modelfile("")

    def test_duplicate_from_fails(self):
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile("FROM a\nFROM b")
        assert "duplicate" in str(exc_info.value).lower()

    def test_unknown_instruction_fails(self):
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile("FROM llama3.3\nBOGUS value")
        assert "unknown instruction" in str(exc_info.value).lower()

    def test_parse_error_carries_line_number(self):
        with pytest.raises(ModelfileParseError) as exc_info:
            parse_modelfile("FROM a\nBOGUS x")
        assert exc_info.value.line == 2


# ----------------------------------------------------------------------
# to_manifest_fields
# ----------------------------------------------------------------------


class TestManifestMapping:
    def test_system_maps_to_manifest(self):
        doc = parse_modelfile('FROM llama3.3\nSYSTEM """hi"""')
        fields = doc.to_manifest_fields()
        assert fields["system"] == "hi"

    def test_template_maps_to_chat_template(self):
        doc = parse_modelfile('FROM llama3.3\nTEMPLATE """{{ .Prompt }}"""')
        fields = doc.to_manifest_fields()
        assert fields["chat_template"] == "{{ .Prompt }}"

    def test_num_ctx_lifts_to_context_length(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER num_ctx 8192")
        fields = doc.to_manifest_fields()
        assert fields["context_length"] == 8192
        # num_ctx must NOT double-count under default_parameters.
        assert "default_parameters" not in fields or (
            "num_ctx" not in fields.get("default_parameters", {})
        )

    def test_other_params_go_to_default_parameters(self):
        doc = parse_modelfile("FROM llama3.3\nPARAMETER temperature 0.7\nPARAMETER top_k 40")
        fields = doc.to_manifest_fields()
        defaults = fields["default_parameters"]
        assert defaults["temperature"] == 0.7
        assert defaults["top_k"] == 40

    def test_stop_sequences_go_to_default_parameters(self):
        doc = parse_modelfile(
            'FROM llama3.3\nPARAMETER stop "<|eot_id|>"\nPARAMETER stop "<|end|>"'
        )
        fields = doc.to_manifest_fields()
        assert fields["default_parameters"]["stop"] == ["<|eot_id|>", "<|end|>"]

    def test_adapters_go_to_manifest(self):
        doc = parse_modelfile("FROM llama3.3\nADAPTER ./a.gguf")
        fields = doc.to_manifest_fields()
        assert fields["adapter_paths"] == ["./a.gguf"]

    def test_license_goes_to_license_name(self):
        doc = parse_modelfile('FROM llama3.3\nLICENSE """Apache 2.0"""')
        fields = doc.to_manifest_fields()
        assert fields["license_name"] == "Apache 2.0"


# ----------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_minimal(self):
        doc = parse_modelfile("FROM llama3.3")
        text2 = render_modelfile_document(doc)
        doc2 = parse_modelfile(text2)
        assert doc2 == doc

    def test_round_trip_full_featured(self):
        text = (
            "FROM llama3.3\n"
            'SYSTEM """You are helpful."""\n'
            'TEMPLATE """{{ .Prompt }}"""\n'
            "PARAMETER num_ctx 4096\n"
            "PARAMETER temperature 0.7\n"
            "PARAMETER use_mmap true\n"
            'PARAMETER stop "<|eot|>"\n'
            "ADAPTER ./lora.gguf\n"
            'MESSAGE user "Hello"\n'
            'MESSAGE assistant "Hi!"\n'
            'LICENSE """MIT"""\n'
            "REQUIRES >=0.6.0\n"
        )
        doc = parse_modelfile(text)
        text2 = render_modelfile_document(doc)
        doc2 = parse_modelfile(text2)
        assert doc2 == doc

    def test_round_trip_renders_bool_as_lowercase(self):
        doc = parse_modelfile("FROM x\nPARAMETER use_mmap true")
        rendered = render_modelfile_document(doc)
        assert "PARAMETER use_mmap true" in rendered
        assert "True" not in rendered

    def test_render_is_deterministic(self):
        text = "FROM x\nPARAMETER top_k 40\nPARAMETER temperature 0.7\nPARAMETER num_ctx 2048\n"
        doc = parse_modelfile(text)
        a = render_modelfile_document(doc)
        b = render_modelfile_document(doc)
        assert a == b


# ----------------------------------------------------------------------
# Realistic corpora
# ----------------------------------------------------------------------


class TestRealisticCorpora:
    def test_llama3_style_modelfile(self):
        text = (
            "# Llama 3.3 fine-tune\n"
            "FROM llama3.3:70b-q4\n"
            "\n"
            'TEMPLATE """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n'
            "{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            '"""\n'
            "\n"
            'SYSTEM """You are a coding assistant."""\n'
            "\n"
            "PARAMETER num_ctx 8192\n"
            "PARAMETER temperature 0.3\n"
            'PARAMETER stop "<|eot_id|>"\n'
            'PARAMETER stop "<|end_of_text|>"\n'
        )
        doc = parse_modelfile(text)
        assert doc.from_ == "llama3.3:70b-q4"
        assert doc.system == "You are a coding assistant."
        assert "<|begin_of_text|>" in (doc.template or "")
        assert doc.parameters["num_ctx"] == 8192
        assert doc.parameters["temperature"] == 0.3
        assert doc.stop_sequences == ["<|eot_id|>", "<|end_of_text|>"]

    def test_bare_value_stripping_inline_comment(self):
        text = "FROM llama3.3  # default model"
        doc = parse_modelfile(text)
        assert doc.from_ == "llama3.3"


# ----------------------------------------------------------------------
# Parameter taxonomy
# ----------------------------------------------------------------------


class TestParameterTaxonomy:
    def test_int_float_bool_sets_are_disjoint(self):
        assert INT_PARAMETERS.isdisjoint(FLOAT_PARAMETERS)
        assert INT_PARAMETERS.isdisjoint(BOOL_PARAMETERS)
        assert FLOAT_PARAMETERS.isdisjoint(BOOL_PARAMETERS)

    def test_common_params_are_classified(self):
        assert "num_ctx" in INT_PARAMETERS
        assert "temperature" in FLOAT_PARAMETERS
        assert "use_mmap" in BOOL_PARAMETERS
