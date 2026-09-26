# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""GGUF embedding models, usable end to end.

Found with nomic-embed-text-v1.5 (2026-09-26), one link at a time: the
Hub's ``sentence-similarity`` tag made ``hfl pull`` refuse it; every GGUF
was taken for a chat model, so ``/api/embed`` refused it; llama-cpp-python
returned vectors of norm 5+ (Ollama's are unit length) and refused inputs
over 512 tokens; and an HFL without llama-cpp-python (Homebrew's) could not
embed at all. Checked for real: both engines give the same vectors (cosine
0.999), unit length.
"""

from __future__ import annotations

import json
import struct

import httpx
import pytest

from hfl.converter.formats import PIPELINE_TAG_TO_TYPE, ModelType, detect_model_type
from hfl.converter.gguf_header import is_embedding_gguf, read_fields
from hfl.engine.embedding_engine import LlamaServerEmbeddingEngine


def _str(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path, fields: list[tuple[str, int, object]]):
    """A GGUF v3 header with ``fields`` as (key, type, value); type 8 is a
    string, 4 a u32, 9 an array of strings."""
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for key, kind, value in fields:
        body += _str(key) + struct.pack("<I", kind)
        if kind == 8:
            body += _str(value)
        elif kind == 4:
            body += struct.pack("<I", value)
        elif kind == 9:
            body += struct.pack("<IQ", 8, len(value)) + b"".join(_str(v) for v in value)
    path.write_bytes(body)
    return path


VOCAB = ("tokenizer.ggml.tokens", 9, ["<s>", "hello", "world"] * 50)


class TestHeader:
    def test_reads_past_a_vocabulary(self, tmp_path):
        model = _gguf(tmp_path / "m.gguf", [("general.architecture", 8, "qwen3"), VOCAB,
                                            ("qwen3.pooling_type", 4, 3)])  # fmt: skip
        assert read_fields(model, {"qwen3.pooling_type"}) == {"qwen3.pooling_type": 3}

    @pytest.mark.parametrize(
        ("fields", "embeds"),
        [
            ([("general.architecture", 8, "nomic-bert"), VOCAB], True),  # an encoder
            ([("general.architecture", 8, "qwen3"), VOCAB, ("qwen3.pooling_type", 4, 3)], True),
            ([("general.architecture", 8, "qwen3"), VOCAB], False),  # a chat model
            ([("general.architecture", 8, "qwen3"), ("qwen3.pooling_type", 4, 0)], False),
        ],
    )
    def test_what_is_an_embedding_model(self, tmp_path, fields, embeds):
        assert is_embedding_gguf(_gguf(tmp_path / "m.gguf", fields)) is embeds

    def test_unreadable_is_not_an_error(self, tmp_path):
        (tmp_path / "cut.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 5) + b"\x09")
        (tmp_path / "other.gguf").write_bytes(b"NOPE" * 10)
        assert not is_embedding_gguf(tmp_path / "cut.gguf")
        assert not is_embedding_gguf(tmp_path / "other.gguf")
        assert not is_embedding_gguf(tmp_path / "missing.gguf")


class TestModelType:
    def test_a_gguf_says_what_it_is(self, tmp_path):
        (tmp_path / "e").mkdir()
        (tmp_path / "c").mkdir()
        embed = _gguf(tmp_path / "e" / "nomic.gguf", [("general.architecture", 8, "nomic-bert")])
        chat = _gguf(tmp_path / "c" / "qwen.gguf", [("general.architecture", 8, "qwen3"), VOCAB])
        _gguf(tmp_path / "c" / "mmproj-qwen.gguf", [("general.architecture", 8, "clip")])
        assert detect_model_type(embed) == ModelType.EMBEDDING
        assert detect_model_type(tmp_path / "e") == ModelType.EMBEDDING
        assert detect_model_type(chat) == ModelType.LLM
        assert detect_model_type(tmp_path / "c") == ModelType.LLM  # not its projector

    def test_the_hubs_sentence_similarity_tag_is_embeddings(self):
        assert PIPELINE_TAG_TO_TYPE["sentence-similarity"] == ModelType.EMBEDDING


def _engine(handler, n_ctx=16, n_embd=4) -> LlamaServerEmbeddingEngine:
    engine = LlamaServerEmbeddingEngine()
    engine._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ls")
    engine._loaded, engine._n_ctx, engine._n_embd = True, n_ctx, n_embd
    return engine


def _server(calls: list):
    """A fake llama-server: a token per word; embeddings echo the length."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(len(body["content"].split())))})
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": " ".join(["w"] * len(body["tokens"]))})
        rows = [
            {"index": i, "embedding": [float(len(t.split())), 0.0, 3.0, 4.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(
            200,
            json={"data": rows[::-1], "usage": {"prompt_tokens": 7}},  # out of order
        )

    return handler


class TestLlamaServerEngine:
    def test_vectors_in_order_and_unit_length(self):
        result = _engine(_server([])).embed(["a", "a b c"])
        assert [round(sum(x * x for x in v), 6) for v in result.embeddings] == [1.0, 1.0]
        assert result.embeddings[0][0] < result.embeddings[1][0]  # "a" first, "a b c" second
        assert result.total_tokens == 7

    def test_dimensions_cut_and_unit_again(self):
        vec = _engine(_server([])).embed(["a b c"], dimensions=2).embeddings[0]
        assert len(vec) == 2 and round(sum(x * x for x in vec), 6) == 1.0

    def test_a_long_input_is_cut_to_the_context_or_refused(self):
        calls: list = []
        engine = _engine(_server(calls), n_ctx=6)
        long = " ".join(["word"] * 20)
        engine.embed([long])
        sent = next(body for path, body in calls if path == "/v1/embeddings")["input"][0]
        assert len(sent.split()) == 4  # the context less room for CLS/SEP
        with pytest.raises(ValueError, match="exceeds the model's context"):
            engine.embed([long], truncate=False)

    def test_what_it_refuses(self):
        engine = _engine(_server([]))
        with pytest.raises(ValueError, match="pooling"):
            engine.embed(["a"], pooling="cls")
        with pytest.raises(ValueError, match="native size"):
            engine.embed(["a"], dimensions=5)
        failing = _engine(lambda request: httpx.Response(500, text="boom"))
        failing._fit = lambda text, truncate: text  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="HTTP 500"):
            failing.embed(["a"])


def test_without_llama_cpp_python_llama_server_embeds(tmp_path, monkeypatch):
    import importlib.util

    from hfl.api import routes_embed
    from hfl.engine import llama_server
    from hfl.engine.embedding_engine import LlamaCppEmbeddingEngine

    model = _gguf(tmp_path / "nomic.gguf", [("general.architecture", 8, "nomic-bert")])
    real_find = importlib.util.find_spec
    monkeypatch.setattr(llama_server, "binary", lambda: "/usr/bin/llama-server")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "llama_cpp" else real_find(name, *a),
    )
    assert isinstance(routes_embed._select_embedding_backend(model), LlamaServerEmbeddingEngine)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    assert isinstance(routes_embed._select_embedding_backend(model), LlamaCppEmbeddingEngine)
