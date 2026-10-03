# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Converter edge cases: model-type detection, GGUF header parsing, the
llama.cpp converter fetch/verify/build plumbing (subprocess, git and the
network are all faked), Modelfile parsing errors, REQUIRES checks."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfl.converter import formats
from hfl.converter import gguf_converter as gc
from hfl.converter import gguf_header as gh
from hfl.converter.formats import ModelType, detect_model_type
from hfl.exceptions import ConversionError, ToolNotFoundError

# ----------------------------------------------------------------------
# formats.detect_model_type
# ----------------------------------------------------------------------

_ARCH_SETS = [
    ("LLM_ARCHITECTURES", ModelType.LLM),
    ("TTS_ARCHITECTURES", ModelType.TTS),
    ("STT_ARCHITECTURES", ModelType.STT),
    ("IMAGE_GEN_ARCHITECTURES", ModelType.IMAGE_GEN),
    ("IMAGE_CLASS_ARCHITECTURES", ModelType.IMAGE_CLASS),
    ("OBJECT_DETECT_ARCHITECTURES", ModelType.OBJECT_DETECT),
    ("IMAGE_SEG_ARCHITECTURES", ModelType.IMAGE_SEG),
    ("EMBEDDING_ARCHITECTURES", ModelType.EMBEDDING),
    ("FILL_MASK_ARCHITECTURES", ModelType.FILL_MASK),
    ("TOKEN_CLASS_ARCHITECTURES", ModelType.TOKEN_CLASS),
    ("QA_ARCHITECTURES", ModelType.QA),
    ("SEQ2SEQ_ARCHITECTURES", ModelType.SUMMARIZATION),
    ("VISUAL_QA_ARCHITECTURES", ModelType.VISUAL_QA),
    ("IMAGE_TEXT_ARCHITECTURES", ModelType.IMAGE_TEXT),
    ("DEPTH_ARCHITECTURES", ModelType.DEPTH),
    ("AUDIO_CLASS_ARCHITECTURES", ModelType.AUDIO_CLASS),
    ("VIDEO_ARCHITECTURES", ModelType.VIDEO),
]


def _config(folder: Path, data: dict, weights: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(data))
    if weights:
        (folder / weights).write_bytes(b"\0")
    return folder


class TestArchitectureMatch:
    @pytest.mark.parametrize("set_name, expected", _ARCH_SETS)
    def test_each_family_maps_to_its_type(self, set_name, expected):
        earlier = set()
        for name, _ in _ARCH_SETS:
            if name == set_name:
                break
            earlier |= set(getattr(formats, name))
        arch = sorted(set(getattr(formats, set_name)) - earlier)[0]
        assert formats._check_architecture_match(["Unknown", arch]) == expected

    def test_no_match(self):
        assert formats._check_architecture_match(["NothingLikeThis"]) is None
        assert formats._check_architecture_match([]) is None


class TestDetectModelType:
    @pytest.mark.parametrize(
        "model_type, expected",
        [
            ("acme_tts", ModelType.TTS),
            ("acmetts", ModelType.TTS),
            ("acme-speechgen", ModelType.TTS),
            ("acme_asr", ModelType.STT),
            ("acme_stt", ModelType.STT),
            ("acme_speech_to_text_v2", ModelType.STT),
            ("acmeasrnet", ModelType.STT),
        ],
    )
    def test_model_type_patterns(self, tmp_path, model_type, expected):
        assert model_type.replace("-", "_") not in formats.MODEL_TYPE_FIELD_TO_TYPE
        _config(tmp_path, {"model_type": model_type})
        assert detect_model_type(tmp_path) == expected

    def test_pipeline_tag(self, tmp_path):
        tag, kind = next(iter(formats.PIPELINE_TAG_TO_TYPE.items()))
        _config(tmp_path, {"model_type": "zzz", "pipeline_tag": tag})
        assert detect_model_type(tmp_path) == kind

    def test_diffusers_scheduler_or_pipeline_class(self, tmp_path):
        a = _config(tmp_path / "a", {})
        (a / "scheduler").mkdir()
        assert detect_model_type(a) == ModelType.IMAGE_GEN
        b = _config(tmp_path / "b", {"_class_name": "FluxPipeline"})
        assert detect_model_type(b) == ModelType.IMAGE_GEN

    def test_sentence_transformers(self, tmp_path):
        _config(tmp_path, {})
        (tmp_path / "sentence_bert_config.json").write_text("{}")
        assert detect_model_type(tmp_path) == ModelType.EMBEDDING

    def test_weights_default_llm_or_encoder_embedding(self, tmp_path):
        llm = _config(tmp_path / "llm", {"architectures": ["AcmeForCausalLM"]}, "m.safetensors")
        assert detect_model_type(llm) == ModelType.LLM
        enc = _config(
            tmp_path / "enc", {"is_encoder_decoder": False, "architectures": ["AcmeModel"]}, "m.bin"
        )
        assert detect_model_type(enc) == ModelType.EMBEDDING
        gen = _config(
            tmp_path / "gen",
            {"is_encoder_decoder": False, "architectures": ["AcmeForConditionalGeneration"]},
            "m.bin",
        )
        assert detect_model_type(gen) == ModelType.LLM

    def test_no_weights_unknown_and_bad_json(self, tmp_path):
        assert detect_model_type(_config(tmp_path / "x", {})) == ModelType.UNKNOWN
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{nope")
        assert detect_model_type(bad) == ModelType.UNKNOWN

    def test_file_inside_folder_uses_folder_config(self, tmp_path):
        _config(tmp_path, {"model_type": "acme_tts"}, "m.safetensors")
        assert detect_model_type(tmp_path / "m.safetensors") == ModelType.TTS


# ----------------------------------------------------------------------
# gguf_header
# ----------------------------------------------------------------------


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path: Path, entries: list[bytes], version: int = 3) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", version, 0, len(entries)) + b"".join(entries))
    return path


def _kv_u32(key, value):
    return _s(key) + struct.pack("<II", 4, value)


def _kv_str(key, value):
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_u32_array(key, values):
    return (
        _s(key)
        + struct.pack("<IIQ", 9, 4, len(values))
        + b"".join(struct.pack("<I", v) for v in values)
    )


def _kv_str_array(key, values):
    return _s(key) + struct.pack("<IIQ", 9, 8, len(values)) + b"".join(_s(v) for v in values)


class TestGgufHeader:
    def test_skipped_and_kept_arrays(self, tmp_path):
        f = _gguf(
            tmp_path / "m.gguf",
            [
                _kv_u32_array("tokenizer.ggml.token_type", [1, 2, 3]),
                _kv_str_array("tokenizer.ggml.tokens", ["a", "b"]),
                _kv_u32_array("wanted.ids", [7, 8]),
                _kv_str("general.architecture", "llama"),
            ],
        )
        assert gh.read_fields(f, {"wanted.ids", "general.architecture"}) == {
            "wanted.ids": [7, 8],
            "general.architecture": "llama",
        }
        everything = gh.read_fields(f)
        assert everything["tokenizer.ggml.tokens"] == ["a", "b"]
        assert everything["tokenizer.ggml.token_type"] == [1, 2, 3]

    def test_unknown_value_type_and_old_version(self, tmp_path):
        bad = _gguf(tmp_path / "bad.gguf", [_s("k") + struct.pack("<I", 99)])
        with pytest.raises(ValueError, match="unknown GGUF value type 99"):
            gh.read_fields(bad)
        old = _gguf(tmp_path / "old.gguf", [], version=1)
        with pytest.raises(ValueError, match="version 1 is not supported"):
            gh.read_fields(old)

    def test_header_without_architecture_is_not_embedding(self, tmp_path):
        f = _gguf(tmp_path / "noarch.gguf", [_kv_u32("general.file_type", 15)])
        assert gh.is_embedding_gguf(f) is False

    def test_encoder_architecture_is_embedding(self, tmp_path):
        f = _gguf(tmp_path / "bert.gguf", [_kv_str("general.architecture", "bert")])
        assert gh.is_embedding_gguf(f) is True
        chat = _gguf(
            tmp_path / "chat.gguf",
            [_kv_str("general.architecture", "llama"), _kv_u32("llama.pooling_type", 0)],
        )
        assert gh.is_embedding_gguf(chat) is False
        pooled = _gguf(
            tmp_path / "qe.gguf",
            [_kv_str("general.architecture", "qwen3"), _kv_u32("qwen3.pooling_type", 3)],
        )
        assert gh.is_embedding_gguf(pooled) is True


# ----------------------------------------------------------------------
# gguf_converter helpers
# ----------------------------------------------------------------------


class TestConvertibility:
    def test_architecture_keyword(self, tmp_path):
        _config(tmp_path, {"model_type": "llama", "architectures": ["LlamaForVoiceGeneration"]})
        ok, reason = gc.check_model_convertibility(tmp_path)
        assert ok is False
        assert "appears to be VOICE" in reason and "LlamaForVoiceGeneration" in reason

    def test_plain_llm_is_convertible(self, tmp_path):
        _config(tmp_path, {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        assert gc.check_model_convertibility(tmp_path) == (True, "")

    def test_audio_config_field(self, tmp_path):
        _config(tmp_path, {"model_type": "llama", "num_mel_bins": 80})
        ok, reason = gc.check_model_convertibility(tmp_path)
        assert ok is False and "'num_mel_bins'" in reason


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


class TestVersionAndPin:
    def test_version_from_marker_git_or_unknown(self, tmp_path, monkeypatch):
        (tmp_path / gc._PIN_MARKER).write_text("b123\n")
        assert gc._get_llama_cpp_version(tmp_path) == "b123"
        (tmp_path / gc._PIN_MARKER).write_text("  ")
        assert gc._get_llama_cpp_version(tmp_path) == "unknown"

    def test_unreadable_marker_falls_back_to_git(self, tmp_path, monkeypatch):
        (tmp_path / gc._PIN_MARKER).write_text("b1")
        real_read = Path.read_text

        def read_text(self, *a, **kw):
            if self.name == gc._PIN_MARKER:
                raise OSError("EACCES")
            return real_read(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", read_text)
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return _completed(0, "abc1234\n")

        monkeypatch.setattr(gc.subprocess, "run", run)
        assert gc._get_llama_cpp_version(tmp_path) == "abc1234"
        assert calls == [["git", "rev-parse", "--short", "HEAD"]]
        # The same unreadable marker means "fetched before the pin".
        assert gc._fetched_before_pin(tmp_path) is True

    def test_verify_git_clone_without_remote(self, tmp_path, monkeypatch):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return _completed(1, "", "no such remote")

        monkeypatch.setattr(gc.subprocess, "run", run)
        assert gc._verify_git_clone(tmp_path, gc.LLAMA_CPP_REPO) is False
        assert len(calls) == 1  # HEAD is not even asked

    def test_verify_git_clone_rejects_other_remote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            gc.subprocess, "run", lambda cmd, **kw: _completed(0, "https://evil.example/x.git\n")
        )
        assert gc._verify_git_clone(tmp_path, gc.LLAMA_CPP_REPO) is False

    def test_verify_git_clone_without_git(self, tmp_path, monkeypatch):
        def run(cmd, **kw):
            raise FileNotFoundError("git")

        monkeypatch.setattr(gc.subprocess, "run", run)
        assert gc._verify_git_clone(tmp_path, gc.LLAMA_CPP_REPO) is False

    def test_clone_at_pinned_commit_is_not_before_pin(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()

        def run(cmd, **kw):
            if cmd[:2] == ["git", "config"]:
                return _completed(0, gc.LLAMA_CPP_REPO.removesuffix(".git") + "\n")
            return _completed(0, gc.LLAMA_CPP_COMMIT + "\n")

        monkeypatch.setattr(gc.subprocess, "run", run)
        assert gc._fetched_before_pin(tmp_path) is False


class TestRemoteCode:
    def test_auto_map_inside_a_list(self):
        assert gc._declares_remote_code({"a": [{"b": 1}, {"auto_map": {}}]}) is True
        assert gc._declares_remote_code([1, "x", {"c": [2]}]) is False

    def test_hub_cache_and_unparseable_json_are_ignored(self, tmp_path):
        cache = tmp_path / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "hook.py").write_text("print('x')")
        (tmp_path / "broken.json").write_text("{auto_map")
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        assert gc.remote_code_in(tmp_path) is None
        (tmp_path / "tokenizer_config.json").write_text('{"auto_map": {"AutoTokenizer": "x"}}')
        assert gc.remote_code_in(tmp_path) == "tokenizer_config.json"
        assert gc.remote_code_in(tmp_path / "config.json") is None


class _Stream:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield self.payload[:10]
        yield self.payload[10:]


def _tarball(members: list[tuple[str, bytes | None, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            info.type = kind or tarfile.REGTYPE
            if info.type == tarfile.REGTYPE:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                if info.type == tarfile.SYMTYPE:
                    info.linkname = "/etc/passwd"
                tar.addfile(info)
    return buf.getvalue()


class TestDownloadConverter:
    def test_unpacks_only_safe_regular_members(self, tmp_path, monkeypatch):
        import httpx

        payload = _tarball(
            [
                ("README", None, b"top-level, no inner path"),
                ("llama.cpp-b1/", tarfile.DIRTYPE, b""),
                ("llama.cpp-b1/gguf-py/", tarfile.DIRTYPE, b""),
                ("llama.cpp-b1/convert_hf_to_gguf.py", None, b"# converter"),
                ("llama.cpp-b1/gguf-py/gguf.py", None, b"# gguf"),
                # unpacked is <staging>/src: this would land beside the staging dir
                ("llama.cpp-b1/../../escape.txt", None, b"nope"),
                ("llama.cpp-b1/link", tarfile.SYMTYPE, b""),
                ("llama.cpp-b1/nofile.txt", None, b"extractfile gives None"),
            ]
        )
        monkeypatch.setattr(gc, "LLAMA_CPP_SHA256", hashlib.sha256(payload).hexdigest())
        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _Stream(payload))
        real_extract = tarfile.TarFile.extractfile

        def extractfile(self, member):
            if member.name.endswith("nofile.txt"):
                return None
            return real_extract(self, member)

        monkeypatch.setattr(tarfile.TarFile, "extractfile", extractfile)
        target = tmp_path / "llama.cpp"
        gc._download_converter(target)
        files = sorted(str(p.relative_to(target)) for p in target.rglob("*"))
        assert files == ["convert_hf_to_gguf.py", "gguf-py", "gguf-py/gguf.py"]
        assert (target / "convert_hf_to_gguf.py").read_bytes() == b"# converter"
        assert not (tmp_path / "escape.txt").exists()
        # The staging directory is gone.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["llama.cpp"]


# ----------------------------------------------------------------------
# GGUFConverter methods
# ----------------------------------------------------------------------


@pytest.fixture
def conv(temp_config):
    return gc.GGUFConverter()


class TestVerifyOutput:
    def test_missing_and_empty_output(self, conv, tmp_path):
        with pytest.raises(RuntimeError, match="not created"):
            conv._verify_output(tmp_path / "none.gguf", tmp_path)
        empty = tmp_path / "e.gguf"
        empty.write_bytes(b"")
        with pytest.raises(RuntimeError, match="empty file"):
            conv._verify_output(empty, tmp_path)
        assert not empty.exists()

    def test_output_within_input_size_or_no_safetensors(self, conv, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(gc.console, "print", lambda msg, *a, **kw: printed.append(msg))
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "o.gguf"
        out.write_bytes(b"y" * 10)
        conv._verify_output(out, src)  # no safetensors: nothing to compare
        (src / "m.safetensors").write_bytes(b"x" * 100)
        conv._verify_output(out, src)  # smaller than input: fine
        assert not any("larger than input" in m for m in printed)
        assert sum("Output verified" in m for m in printed) == 2

    def test_output_larger_than_input_warns(self, conv, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(gc.console, "print", lambda msg, *a, **kw: printed.append(msg))
        src = tmp_path / "src"
        src.mkdir()
        (src / "m.safetensors").write_bytes(b"x" * 10)
        out = tmp_path / "o.gguf"
        out.write_bytes(b"y" * 100)
        conv._verify_output(out, src)
        assert any("unusually small" in m for m in printed)
        assert any("Output larger than input" in m for m in printed)
        assert "Output verified" in printed[-1]

    def test_unreadable_source_skips_size_comparison(self, conv, tmp_path, monkeypatch):
        printed = []
        monkeypatch.setattr(gc.console, "print", lambda msg, *a, **kw: printed.append(msg))
        out = tmp_path / "o.gguf"
        out.write_bytes(b"y")

        class _Src:
            def glob(self, pattern):
                raise OSError("gone")

        conv._verify_output(out, _Src())
        assert not any("larger than input" in m for m in printed)
        assert "Output verified" in printed[-1]


class TestConversionEnvironment:
    def test_probe_failure_is_actionable(self, conv, monkeypatch):
        monkeypatch.setattr(
            gc.subprocess,
            "run",
            lambda cmd, **kw: _completed(2, "", "ImportError: cannot import name 'x'"),
        )
        with pytest.raises(ConversionError) as exc:
            conv._check_conversion_environment()
        assert "cannot import name 'x'" in str(exc.value)
        assert "hfl[transformers]" in str(exc.value)

    def test_probe_failure_without_stderr(self, conv, monkeypatch):
        monkeypatch.setattr(gc.subprocess, "run", lambda cmd, **kw: _completed(1, "", ""))
        with pytest.raises(ConversionError, match="<no stderr>"):
            conv._check_conversion_environment()

    def test_probe_success(self, conv, monkeypatch):
        monkeypatch.setattr(gc.subprocess, "run", lambda cmd, **kw: _completed(0))
        conv._check_conversion_environment()  # no raise


class TestEnsureTools:
    def test_clone_that_fails_verification_is_removed(self, conv, monkeypatch):
        monkeypatch.setattr(gc.console, "print", lambda *a, **kw: None)
        monkeypatch.setattr(gc.shutil, "which", lambda name: "/usr/bin/git")

        def run(cmd, **kw):
            if cmd[:2] == ["git", "clone"]:
                Path(cmd[-1]).mkdir(parents=True)
            return _completed(0)

        monkeypatch.setattr(gc.subprocess, "run", run)
        monkeypatch.setattr(gc, "_verify_git_clone", lambda d, repo: False)
        with pytest.raises(RuntimeError, match="integrity verification failed"):
            conv.ensure_tools()
        assert not conv.llama_cpp_dir.exists()

    def test_fetched_tree_without_converter(self, conv, monkeypatch):
        monkeypatch.setattr(gc.console, "print", lambda *a, **kw: None)
        monkeypatch.setattr(gc.shutil, "which", lambda name: None)
        monkeypatch.setattr(gc, "_download_converter", lambda target: target.mkdir())
        with pytest.raises(ToolNotFoundError) as exc:
            conv.ensure_tools()
        assert "convert_hf_to_gguf.py" in str(exc.value)
        assert not (conv.llama_cpp_dir / gc._PIN_MARKER).exists()


class TestQuantizer:
    def test_last_resort_builds_llama_quantize(self, conv, monkeypatch):
        import hfl.engine.llama_server_dist as dist

        monkeypatch.setattr(gc.shutil, "which", lambda name: None)
        monkeypatch.setattr(dist, "bundled_binary", lambda name: None)
        monkeypatch.setattr(dist, "managed_binary", lambda name: None)
        monkeypatch.setattr(gc.subprocess, "run", lambda cmd, **kw: _completed(1))
        built = []
        monkeypatch.setattr(conv, "_build_quantizer", lambda: built.append(True))
        assert conv._quantizer() == [str(conv.quantize_bin)]
        assert built == [True]

    def test_build_needs_git_and_cmake(self, conv, monkeypatch):
        monkeypatch.setattr(gc.shutil, "which", lambda name: None if name == "cmake" else "/x")
        with pytest.raises(ToolNotFoundError) as exc:
            conv._build_quantizer()
        assert "cmake" in str(exc.value)

    def test_build_needs_cmakelists(self, conv, monkeypatch):
        monkeypatch.setattr(gc.shutil, "which", lambda name: "/usr/bin/" + name)
        conv.llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ToolNotFoundError, match="build files"):
            conv._build_quantizer()

    @pytest.mark.parametrize("cuda", [True, False])
    def test_build_runs_cmake_with_cuda_when_nvcc(self, conv, monkeypatch, cuda):
        monkeypatch.setattr(gc.console, "print", lambda *a, **kw: None)
        monkeypatch.setattr(
            gc.shutil,
            "which",
            lambda name: None if (name == "nvcc" and not cuda) else "/usr/bin/" + name,
        )
        conv.llama_cpp_dir.mkdir(parents=True, exist_ok=True)
        (conv.llama_cpp_dir / "CMakeLists.txt").write_text("")
        calls = []
        monkeypatch.setattr(
            gc.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or _completed(0)
        )
        conv._build_quantizer()
        build_dir = conv.llama_cpp_dir / "build"
        assert build_dir.is_dir()
        configure = ["cmake", "..", "-DGGML_CUDA=ON"] if cuda else ["cmake", ".."]
        assert calls[0] == (configure, {"cwd": build_dir, "check": True})
        assert calls[1][0][:3] == ["cmake", "--build", "."]
        assert "llama-quantize" in calls[1][0]


# ----------------------------------------------------------------------
# Modelfile parsing / rendering, REQUIRES
# ----------------------------------------------------------------------


class TestModelfileParserEdges:
    def _parse(self, text, base=None):
        from hfl.converter.modelfile_parser import parse_modelfile

        return parse_modelfile(text, base)

    def test_escapes_and_multiline_opener_on_own_line(self):
        doc = self._parse('FROM m\nSYSTEM "a\\rb\\\\c\\x"\nTEMPLATE """\nline1\nline2"""\n')
        assert doc.system == "a\rb\\c\\x"
        assert doc.template == "line1\nline2"

    @pytest.mark.parametrize(
        "text, message",
        [
            ('FROM m\nSYSTEM "a" trailing', "unexpected tokens after quoted value"),
            ('FROM m\nSYSTEM "never closed', "unterminated double-quoted string"),
            ("FROM m\nPARAMETER temperature\n", "requires a key and a value"),
            ("FROM m\nPARAMETER temperature hot\n", "expects float"),
            ("FROM m\nMESSAGE user\n", "requires a role and content"),
            ("FROM\n", "FROM requires a value"),
            ("FROM a\nFROM b\n", "duplicate FROM"),
        ],
    )
    def test_errors(self, text, message):
        from hfl.converter.modelfile_parser import ModelfileParseError

        with pytest.raises(ModelfileParseError, match=message):
            self._parse(text)

    def test_quoted_value_with_trailing_comment_and_bare_comment(self):
        doc = self._parse(
            'FROM m  # base\nSYSTEM "hi" # note\nPARAMETER mirostat_mode sha:#1 # c\n'
        )
        assert doc.from_ == "m"
        assert doc.system == "hi"
        assert doc.parameters["mirostat_mode"] == "sha:#1"

    @pytest.mark.parametrize(
        "rel, ok",
        [
            ("common.modelfile", True),
            ("./sub/common.modelfile", True),
            ("./", False),
            ("..", False),
            ("/abs", False),
            ("a//b", False),
            ("a/./b", False),
            ("a/../b", False),
            ("bad name", False),
            ("x" * 513, False),
            ("a\0b", False),
        ],
    )
    def test_include_whitelist(self, rel, ok):
        from hfl.converter.modelfile_parser import _is_safe_include_rel

        assert _is_safe_include_rel(rel) is ok

    @pytest.mark.parametrize(
        "line, expected",
        [
            ("INCLUDE common.modelfile", "common.modelfile"),
            ('  include "q.modelfile" # why', "q.modelfile"),
            ("INCLUDE", None),
            ("INCLUDEX common", None),
            ("FROM somethinglong", None),
            ("INCLUDE  # only a comment", None),
            ("INCLUDE " + "x" * 2000, None),
        ],
    )
    def test_include_line_matcher(self, line, expected):
        from hfl.converter.modelfile_parser import _match_include_line

        assert _match_include_line(line) == expected

    def test_include_errors(self, tmp_path, monkeypatch):
        from hfl.converter import modelfile_parser as mp

        (tmp_path / "self.modelfile").write_text("INCLUDE self.modelfile\n")
        with pytest.raises(mp.ModelfileParseError, match="cycle"):
            self._parse("FROM m\nINCLUDE self.modelfile\n", tmp_path)
        with pytest.raises(mp.ModelfileParseError, match="whitelist"):
            self._parse("FROM m\nINCLUDE ../etc/passwd\n", tmp_path)
        with pytest.raises(mp.ModelfileParseError, match="depth 16"):
            mp._expand_includes("INCLUDE x", tmp_path, depth=17)
        (tmp_path / "ok.modelfile").write_text('SYSTEM "s"\n')
        real_open = open

        def failing_open(path, *a, **kw):
            if str(path).endswith("ok.modelfile"):
                raise OSError("EIO")
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", failing_open)
        with pytest.raises(mp.ModelfileParseError, match="unreadable"):
            self._parse("FROM m\nINCLUDE ok.modelfile\n", tmp_path)

    def test_triple_quotes_round_trip(self):
        from hfl.converter.modelfile_parser import (
            ModelfileDocument,
            render_modelfile_document,
        )

        doc = ModelfileDocument()
        doc.from_ = "m"
        doc.system = 'say """hi""" \\n'
        text = render_modelfile_document(doc)
        assert '\\"\\"\\"' in text
        assert self._parse(text).system == doc.system


class TestRenderModelfileEdges:
    def _m(self, **kw):
        from hfl.models.manifest import ModelManifest

        return ModelManifest(name="n", repo_id="o/n", local_path="/srv/m.gguf", format="gguf", **kw)

    def test_hash_with_algorithm_prefix_and_system(self):
        from hfl.converter.modelfile import render_modelfile

        m = self._m(file_hash="sha512:abcd")
        m.system = "be brief"
        text = render_modelfile(m)
        assert text.startswith("FROM sha512:abcd\n")
        assert 'SYSTEM """be brief"""' in text

    def test_template_with_triple_quotes_is_escaped(self):
        from hfl.converter.modelfile import render_modelfile

        text = render_modelfile(self._m(chat_template='a """ b \\n'))
        assert 'TEMPLATE """a \\"\\"\\" b \\\\n"""' in text

    def test_bare_hash_gets_sha256_prefix(self):
        from hfl.converter.modelfile import render_modelfile

        assert render_modelfile(self._m(file_hash="abcd")).startswith("FROM sha256:abcd\n")

    def test_hidden_paths_use_the_name(self):
        from hfl.converter.modelfile import render_modelfile

        assert render_modelfile(self._m(), reveal_paths=False).startswith("FROM n\n")


class TestRequiresEdges:
    def test_bad_bare_version_and_bad_current(self):
        from hfl.converter.requires_check import (
            InvalidRequiresError,
            check_requires,
            parse_requires,
        )

        with pytest.raises(InvalidRequiresError, match="invalid version"):
            parse_requires("1.0.0.dev-what")
        with pytest.raises(InvalidRequiresError, match="invalid current version"):
            check_requires(">=0.1", current="not-a-version")
        assert SimpleNamespace(ok=check_requires(">=0.1", current="1.0")).ok is None
