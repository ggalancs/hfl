# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Embedding engines without their optional backends.

- :class:`LlamaServerEmbeddingEngine` against a stand-in llama-server
  client (its process start and stop replaced): the command line, the
  context fitting, ``truncate``, ``dimensions`` and the reply ordering.
- :class:`TransformersEmbeddingEngine.embed` against a small numpy-backed
  stand-in for the torch tensor operations it uses, so the pooling and
  Matryoshka normalisation are checked in the CI venv (no torch there).
- :class:`LlamaCppEmbeddingEngine` with a llama-cpp-python that returns
  one row per input as a list of lists.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

from hfl.engine import embedding_engine as ee

# ------------------------------------------------------------ llama-server


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


class _Client:
    """The endpoints LlamaServerEmbeddingEngine calls: one token per word."""

    def __init__(self):
        self.posts: list[tuple[str, dict]] = []
        self.closed = False
        self.status = 200
        self.rows = [
            {"index": 1, "embedding": [0.0, 3.0, 4.0]},
            {"index": 0, "embedding": [3.0, 4.0, 0.0]},
        ]

    def post(self, path, json):
        self.posts.append((path, json))
        if path == "/tokenize":
            return _Resp({"tokens": list(range(len(json["content"].split())))})
        if path == "/detokenize":
            return _Resp({"content": " ".join(f"w{t}" for t in json["tokens"])})
        if path == "/v1/embeddings":
            return _Resp({"data": self.rows, "usage": {"prompt_tokens": 11}}, self.status)
        raise AssertionError(path)

    def close(self):
        self.closed = True


@pytest.fixture
def server(monkeypatch, temp_config):
    import hfl.converter.gguf_header as gh
    import hfl.engine.llama_server as ls

    state: dict = {"fields": {"general.architecture": "bert", "bert.context_length": 512,
                              "bert.embedding_length": 3}, "stopped": []}  # fmt: skip
    client = _Client()

    def read_fields(path, keys):
        if state["fields"] is None:
            raise OSError("unreadable")
        return {k: v for k, v in state["fields"].items() if k in keys}

    def start_server(argv, model_path, log_path, timeout):
        state["argv"], state["log_path"], state["timeout"] = argv, log_path, timeout
        return "proc", client

    monkeypatch.setattr(gh, "read_fields", read_fields)
    monkeypatch.setattr(ls, "binary", lambda: "/bin/llama-server")
    monkeypatch.setattr(ls, "start_server", start_server)
    monkeypatch.setattr(ls, "stop_server", lambda proc: state["stopped"].append(proc))
    state["client"] = client
    return state


def _loaded(path="/m/bge.gguf", **kwargs) -> ee.LlamaServerEmbeddingEngine:
    engine = ee.LlamaServerEmbeddingEngine()
    engine.load(path, **kwargs)
    return engine


class TestLlamaServerEmbedding:
    def test_a_missing_binary_is_a_clear_error(self, monkeypatch):
        import hfl.engine.llama_server as ls

        monkeypatch.setattr(ls, "binary", lambda: None)
        with pytest.raises(RuntimeError, match="hfl install llama-server"):
            ee.LlamaServerEmbeddingEngine().load("/m/x.gguf")

    def test_command_line_from_the_header(self, server, temp_config):
        engine = _loaded()
        argv = server["argv"]
        assert argv[:3] == ["/bin/llama-server", "-m", "/m/bge.gguf"]
        assert "--embeddings" in argv
        # The trained context, also the batch: an encoder takes one batch.
        for flag in ("-c", "-b", "-ub"):
            assert argv[argv.index(flag) + 1] == "512"
        assert server["log_path"] == temp_config.home_dir / "logs" / "llama-server-embed-bge.log"
        assert engine.is_loaded and engine.model_name == "/m/bge.gguf"
        assert engine._n_embd == 3

    def test_an_explicit_context_and_an_unknown_training_length(self, server):
        server["fields"] = {"general.architecture": "bert"}
        engine = _loaded(n_ctx=64)
        assert server["argv"][server["argv"].index("-c") + 1] == "64"
        assert engine._n_embd is None
        _loaded()
        # Unknown training length: 8192 at most, as the in-process engine.
        assert server["argv"][server["argv"].index("-c") + 1] == "8192"

    def test_an_unreadable_header_falls_back_to_8192(self, server):
        server["fields"] = None  # read_fields raises, as on a non-GGUF file
        engine = _loaded()
        assert server["argv"][server["argv"].index("-c") + 1] == "8192"
        assert engine._n_embd is None

    def test_embed_orders_rows_truncates_and_normalises(self, server):
        engine = _loaded()
        result = engine.embed(["first text", "second"], dimensions=2)
        # Row with index 0 first; [3, 4] and [0, 3] made unit length.
        assert result.embeddings == [[0.6, 0.8], [0.0, 1.0]]
        assert result.total_tokens == 11 and result.model == "/m/bge.gguf"
        assert server["client"].posts[-1] == ("/v1/embeddings", {"input": ["first text", "second"]})

    def test_long_inputs_are_cut_to_the_context(self, server):
        engine = _loaded(n_ctx=5)
        long = "a b c d e f g"
        engine.embed([long])
        sent = server["client"].posts[-1][1]["input"]
        # Room for CLS/SEP: 3 tokens kept, as llama-server detokenized them.
        assert sent == ["w0 w1 w2"]
        with pytest.raises(ValueError, match="exceeds the model's context"):
            engine.embed([long], truncate=False)

    def test_an_http_error(self, server):
        engine = _loaded()
        server["client"].status = 500
        with pytest.raises(RuntimeError, match="HTTP 500"):
            engine.embed(["x"])

    def test_requests_it_refuses(self, server):
        engine = ee.LlamaServerEmbeddingEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.embed(["x"])
        engine = _loaded()
        with pytest.raises(ValueError, match="pooling='cls'"):
            engine.embed(["x"], pooling="cls")
        with pytest.raises(ValueError, match="non-empty"):
            engine.embed([])
        with pytest.raises(ValueError, match="positive"):
            engine.embed(["x"], dimensions=0)
        with pytest.raises(ValueError, match="exceeds model's native size"):
            engine.embed(["x"], dimensions=4)

    def test_unload_closes_and_stops(self, server):
        engine = _loaded()
        engine.unload()
        assert server["client"].closed and server["stopped"] == ["proc"]
        assert not engine.is_loaded and engine.model_name == ""
        # A second unload has no client left to close; still stops nothing.
        engine.unload()
        assert server["stopped"] == ["proc", None]


# ------------------------------------------------------------ transformers


class _T:
    """The torch.Tensor operations TransformersEmbeddingEngine.embed uses."""

    def __init__(self, data, dtype=None):
        self.a = np.asarray(data, dtype=dtype or float)
        self.dtype = self.a.dtype
        self.device = "cpu"

    def __iter__(self):
        return (_T(row) for row in self.a)

    def __getitem__(self, key):
        return _T(self.a[key])

    def __mul__(self, other):
        return _T(self.a * other.a)

    def __truediv__(self, other):
        return _T(self.a / other.a)

    def sum(self, dim=None):
        return _T(self.a.sum(axis=dim))

    def item(self):
        return self.a.item()

    def unsqueeze(self, dim):
        return _T(np.expand_dims(self.a, dim))

    def float(self):
        return self

    def clamp(self, min):
        return _T(np.maximum(self.a, min))

    def norm(self, p, dim, keepdim):
        return _T(np.linalg.norm(self.a, ord=p, axis=dim, keepdims=keepdim))

    def cpu(self):
        return self

    def tolist(self):
        return self.a.tolist()


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.fixture
def torch_stub(monkeypatch):
    torch = types.ModuleType("torch")
    torch.no_grad = _NoGrad
    torch.tensor = lambda rows, dtype=None, device=None: _T(rows, dtype=dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# Two inputs, three positions, two hidden units; the second input is padded.
HIDDEN = [
    [[1.0, 0.0], [3.0, 0.0], [5.0, 5.0]],
    [[0.0, 2.0], [0.0, 4.0], [9.0, 9.0]],
]
MASK = [[1, 1, 1], [1, 1, 0]]


class _Encoded(dict):
    def to(self, device):
        self.device = device
        return self


def _transformers_engine(dims=2):
    engine = ee.TransformersEmbeddingEngine()
    calls: dict = {}

    def tokenizer(inputs, padding, truncation, return_tensors):
        calls["tokenizer"] = {"inputs": inputs, "truncation": truncation}
        return _Encoded(attention_mask=_T(MASK), input_ids=_T([[1, 2, 3], [1, 2, 0]]))

    def model(**encoded):
        calls["model"] = sorted(encoded)
        return types.SimpleNamespace(last_hidden_state=_T(HIDDEN))

    engine._tokenizer, engine._model = tokenizer, model
    engine._n_embd, engine._loaded, engine._model_path = dims, True, "org/encoder"
    return engine, calls


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestTransformersEmbed:
    def test_mean_pooling_over_the_mask(self, torch_stub):
        engine, calls = _transformers_engine()
        result = engine.embed(["a", "b"], truncate=False)
        # Row 0: mean of all three; row 1: of the first two (third is padding).
        assert result.embeddings[0] == pytest.approx(_unit([3.0, 5.0 / 3.0]))
        assert result.embeddings[1] == pytest.approx(_unit([0.0, 3.0]))
        assert result.total_tokens == 5 and result.model == "org/encoder"
        assert calls["tokenizer"] == {"inputs": ["a", "b"], "truncation": False}
        assert calls["model"] == ["attention_mask", "input_ids"]

    def test_cls_and_last_token_pooling(self, torch_stub):
        engine, _ = _transformers_engine()
        cls = engine.embed(["a", "b"], pooling="cls").embeddings
        assert cls[0] == pytest.approx(_unit([1.0, 0.0]))
        assert cls[1] == pytest.approx(_unit([0.0, 2.0]))
        last = engine.embed(["a", "b"], pooling="last").embeddings
        # The last token the mask keeps, not the padding after it.
        assert last[0] == pytest.approx(_unit([5.0, 5.0]))
        assert last[1] == pytest.approx(_unit([0.0, 4.0]))

    def test_matryoshka_truncates_before_normalising(self, torch_stub):
        engine, _ = _transformers_engine()
        out = engine.embed(["a", "b"], dimensions=1).embeddings
        assert out == [pytest.approx([1.0]), pytest.approx([0.0])]

    def test_an_unknown_pooling_is_refused(self, torch_stub):
        engine, _ = _transformers_engine()
        with pytest.raises(ValueError, match="unknown pooling 'max'"):
            engine.embed(["a"], pooling="max")


# --------------------------------------------------------------- llama.cpp


def test_llama_cpp_rows_given_as_a_list_of_lists(monkeypatch):
    class Llama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def n_embd(self):
            return 3

        def embed(self, text, truncate, normalize):
            return [[0.0, 0.6, 0.8]]

        def tokenize(self, data):
            return [1, 2]

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama))
    engine = ee.LlamaCppEmbeddingEngine()
    engine.load("/m/e.gguf")
    result = engine.embed(["x"], dimensions=2)
    # The first row, cut to two numbers and made unit length again.
    assert result.embeddings == [[0.0, 1.0]]
    assert result.total_tokens == 2
