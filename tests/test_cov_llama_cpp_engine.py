# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl.engine.llama_cpp`` against a stand-in ``llama_cpp`` package.

The CI venv has no llama-cpp-python, so most of the engine's tests skip
there. Here a small fake package — ``Llama`` and its vocabulary, the
grammar classes, the chat formatter, the low-level ``llama_cpp.llama_cpp``
calls (perf counters, LoRA adapters, the log callback, EOG) — sits in
``sys.modules`` so load, generate, chat, streams, logprobs, grammars, LoRA
and the template plumbing run everywhere, with their results checked.
"""

from __future__ import annotations

import ctypes
import io
import logging
import sys
import types

import numpy as np
import pytest

import hfl.engine.llama_cpp as lc
from hfl.engine.base import ChatMessage, GenerationConfig

EOS = 2

# --------------------------------------------------------------- the fake


class _Vocab:
    def __init__(self, vocab=None, bos=1, add_bos=True):
        self.vocab = vocab
        self._bos = bos
        self._add_bos = add_bos

    def token_bos(self):
        return self._bos

    def token_eos(self):
        return EOS

    def token_get_text(self, token):
        return {1: "<s>", EOS: "</s>"}.get(token, "?")

    def add_bos_token(self):
        return self._add_bos


class _Formatter:
    """llama-cpp-python's ``Jinja2ChatFormatter``, reduced: renders the
    messages as ``role:content`` lines and records what it was called with."""

    def __init__(self, template, eos_token, bos_token, stop_token_ids):
        self.template, self.eos, self.bos, self.stop_ids = (
            template,
            eos_token,
            bos_token,
            stop_token_ids,
        )
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        lines = [f"{m['role']}:{m.get('content')}" for m in kwargs["messages"]]
        if kwargs.get("tools"):
            lines.append(f"tools:{len(kwargs['tools'])}")
        return types.SimpleNamespace(
            prompt="\n".join(lines), added_special=self.template.startswith("{{ bos_token }}")
        )

    def to_chat_handler(self):
        return ("handler", self)


class _Handler:
    def __init__(self, clip_model_path, verbose=False):
        self.clip_model_path, self.verbose = clip_model_path, verbose


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``."""

    metadata: dict = {}
    emit_log = b""
    fail_with: Exception | None = None
    instances: list[FakeLlama] = []

    def __init__(self, **kwargs):
        if FakeLlama.fail_with is not None:
            raise FakeLlama.fail_with
        if FakeLlama.emit_log and FAKE.lcpp.log_cb is not None:
            FAKE.lcpp.log_cb(2, FakeLlama.emit_log, None)
        self.kwargs = kwargs
        self.metadata = dict(FakeLlama.metadata)
        self._chat_handlers: dict = {}
        self._model = _Vocab()
        self._ctx = types.SimpleNamespace(ctx="ctx-ptr")
        self.model, self.ctx = "model-ptr", "ctx-ptr"
        self.chat_format = kwargs.get("chat_format")
        self.n_tokens = 0
        self.calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.reject_kwargs: set[str] = set()
        self.reply: dict | list = {
            "choices": [{"text": "out", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
        self.chat_reply: dict | list = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        self.script: list[int] = []
        self.resets = 0
        FakeLlama.instances.append(self)

    def n_ctx(self):
        return self.kwargs.get("n_ctx") or 4096

    def n_vocab(self):
        return 5

    def token_eos(self):
        return EOS

    def tokenize(self, data, add_bos=True, special=False):
        return ([1] if add_bos else []) + [7] * len(data.split())

    def detokenize(self, tokens, prev_tokens=None, special=False):
        words = {0: b"z", 1: b"a", 3: b"Hel", 4: b"lo"}
        return b"".join(words.get(t, b"?") for t in tokens) + (b"!" if special else b"")

    def set_seed(self, seed):
        self.seed = seed

    def reset(self):
        self.resets += 1

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if kwargs.get("stream"):
            return iter(self.reply)
        return self.reply

    def create_chat_completion(self, **kwargs):
        self.chat_calls.append(dict(kwargs))
        bad = self.reject_kwargs & set(kwargs)
        if bad:
            raise TypeError(f"unexpected keyword {sorted(bad)[0]}")
        if kwargs.get("stream"):
            return iter(self.chat_reply)
        return self.chat_reply

    def generate(self, tokens, top_k, top_p, temp, repeat_penalty, logits_processor):
        self.gen_args = {"tokens": list(tokens), "temp": temp, "repeat_penalty": repeat_penalty}
        for token in self.script:
            row = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
            for proc in logits_processor:
                row = proc(None, row)
            yield token


class _Lcpp(types.ModuleType):
    """``llama_cpp.llama_cpp``: the C API calls the engine makes."""

    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 = 0, 1, 2, 8
    llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self):
        super().__init__("llama_cpp.llama_cpp")
        self.log_cb = None
        self.perf = types.SimpleNamespace(t_p_eval_ms=3.0, n_p_eval=4, t_eval_ms=5.0, n_eval=2)
        self.resets: list = []
        self.gpu = True
        self.lora_sets: list = []
        self.lora_freed: list = []
        self.set_result = 0

    def llama_log_set(self, cb, user_data):
        self.log_cb = cb

    def llama_perf_context_reset(self, ctx):
        self.resets.append(ctx)

    def llama_perf_context(self, ctx):
        return self.perf

    def llama_supports_gpu_offload(self):
        return self.gpu

    def llama_adapter_lora_init(self, model, path):
        return 0 if b"bad" in path else {b"/a.gguf": 0x1000, b"/b.gguf": 0x2000}.get(path, 0x3000)

    def llama_set_adapters_lora(self, ctx, handles, count, scales):
        self.lora_sets.append([(handles[i], round(scales[i], 3)) for i in range(count)])
        return self.set_result

    def llama_adapter_lora_free(self, handle):
        self.lora_freed.append(handle.value if hasattr(handle, "value") else handle)

    def llama_vocab_is_eog(self, vocab, token):
        return token == EOS


class _Grammar:
    @staticmethod
    def from_string(text, verbose=True):
        return ("gbnf", text)

    @staticmethod
    def from_json_schema(text, verbose=True):
        return ("schema", text)


class _Fake:
    pass


FAKE = _Fake()


@pytest.fixture
def fake(monkeypatch):
    """The stand-in ``llama_cpp`` package, installed for one test."""
    pkg = types.ModuleType("llama_cpp")
    lcpp = _Lcpp()
    grammar = types.ModuleType("llama_cpp.llama_grammar")
    grammar.JSON_GBNF = "json-gbnf"
    grammar.LlamaGrammar = _Grammar

    def json_schema_to_gbnf(text):
        if "unbuildable" in text:
            raise ValueError("no")
        return "gbnf-of:" + text

    grammar.json_schema_to_gbnf = json_schema_to_gbnf
    chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    chat_format.Jinja2ChatFormatter = _Formatter
    for name in (
        "Llava15ChatHandler",
        "Llava16ChatHandler",
        "MoondreamChatHandler",
        "Qwen25VLChatHandler",
    ):
        setattr(chat_format, name, type(name, (_Handler,), {}))
    speculative = types.ModuleType("llama_cpp.llama_speculative")
    speculative.LlamaPromptLookupDecoding = lambda num_pred_tokens, max_ngram_size: (
        "lookup",
        num_pred_tokens,
        max_ngram_size,
    )
    pkg.llama_cpp = lcpp
    pkg.llama_grammar = grammar
    pkg.llama_chat_format = chat_format
    pkg.Llama = FakeLlama
    pkg.LlamaGrammar = _Grammar
    pkg.StoppingCriteriaList = type("StoppingCriteriaList", (list,), {})
    pkg.LogitsProcessorList = type("LogitsProcessorList", (list,), {})
    pkg.llama_log_callback = lambda fn: fn
    pkg._logger = types.SimpleNamespace(llama_log_callback="library-callback")
    pkg.samplers = []
    pkg.freed = []

    def init_grammar(vocab, source, root):
        pkg.samplers.append(source)
        return 0 if b"UNDEFINED" in source else 99

    pkg.llama_sampler_init_grammar = init_grammar
    pkg.llama_sampler_free = lambda sampler: pkg.freed.append(sampler)
    for name, mod in {
        "llama_cpp": pkg,
        "llama_cpp.llama_cpp": lcpp,
        "llama_cpp.llama_grammar": grammar,
        "llama_cpp.llama_chat_format": chat_format,
        "llama_cpp.llama_speculative": speculative,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(lc, "Llama", FakeLlama)
    monkeypatch.setattr(FakeLlama, "metadata", {})
    monkeypatch.setattr(FakeLlama, "emit_log", b"")
    monkeypatch.setattr(FakeLlama, "fail_with", None)
    FakeLlama.instances.clear()
    FAKE.lcpp = lcpp
    FAKE.pkg = pkg
    return FAKE


@pytest.fixture
def no_jinja(monkeypatch):
    """Templates cannot be rendered here (as in the CI venv), in both venvs."""
    import hfl.models.chat_template as ct

    monkeypatch.setattr(ct, "template_env", lambda: None)
    lc._probe_template.cache_clear()
    yield
    lc._probe_template.cache_clear()


@pytest.fixture
def gguf(tmp_path, monkeypatch):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"GGUF")
    monkeypatch.setattr(lc, "_read_gguf_model_info", lambda _p: None)
    return path


def _engine(fake, **attrs) -> lc.LlamaCppEngine:
    """An engine holding a FakeLlama, as after a load."""
    engine = lc.LlamaCppEngine()
    engine._model = FakeLlama(n_ctx=2048)
    engine._model_path = "/models/tiny.gguf"
    for key, value in attrs.items():
        setattr(engine, key, value)
    return engine


# ---------------------------------------------------------- stderr and logs


class _FdStream(io.StringIO):
    def __init__(self, fd):
        super().__init__()
        self._fd = fd

    def fileno(self):
        return self._fd


class _NoFdStream(io.StringIO):
    def fileno(self):
        raise io.UnsupportedOperation("no fd")


class _FixedStreamHandler(logging.StreamHandler):
    """A handler whose stream cannot be set (as logging's _StderrHandler)."""

    @property
    def stream(self):
        return sys.__stderr__

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ assigns it once
        pass

    def setStream(self, stream):
        raise AttributeError("read-only stream")


class TestStderr:
    def test_handlers_on_fd_2_are_found(self):
        log = logging.getLogger("hfl.test_cov_llama.fd")
        on_two = logging.StreamHandler(_FdStream(2))
        elsewhere = logging.StreamHandler(_FdStream(9))
        no_fd = logging.StreamHandler(_NoFdStream())
        for handler in (on_two, elsewhere, no_fd):
            log.addHandler(handler)
        try:
            found = lc._stream_handlers_on(2)
        finally:
            for handler in (on_two, elsewhere, no_fd):
                log.removeHandler(handler)
        assert on_two in found
        assert elsewhere not in found and no_fd not in found

    def test_logging_keeps_writing_while_fd_2_is_silenced(self, monkeypatch):
        original = io.StringIO()
        movable = logging.StreamHandler(original)
        fixed = _FixedStreamHandler()
        monkeypatch.setattr(lc, "_stream_handlers_on", lambda fd: [fixed, movable])
        with lc._suppress_stderr():
            # The movable handler writes to a copy of the real stderr now.
            assert movable.stream is not original
            assert movable.stream.fileno() not in (2,)
        assert movable.stream is original


class TestLogCapture:
    def test_the_loader_log_lands_in_the_sink_and_the_library_callback_returns(self, fake):
        sink: list[str] = []
        with lc._capture_llama_log(sink):
            fake.lcpp.log_cb(2, b"offloaded 3/3 layers to GPU\n", None)
            assert lc._ACTIVE_LOG_CALLBACKS
        assert sink == ["offloaded 3/3 layers to GPU\n"]
        assert fake.lcpp.log_cb == "library-callback"
        assert lc._ACTIVE_LOG_CALLBACKS == []

    def test_a_failure_after_installing_drops_the_callback(self, fake, monkeypatch):
        def boom(cb, user_data):
            raise OSError("no symbol")

        monkeypatch.setattr(fake.lcpp, "llama_log_set", boom)
        sink: list[str] = []
        with lc._capture_llama_log(sink):
            pass
        assert sink == [] and lc._ACTIVE_LOG_CALLBACKS == []


# ------------------------------------------------------------------ helpers


class TestValidatedPath:
    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            lc._validated_gguf_path(str(tmp_path / "none.gguf"))

    def test_a_directory(self, tmp_path):
        folder = tmp_path / "dir.gguf"
        folder.mkdir()
        with pytest.raises(ValueError, match="not a file"):
            lc._validated_gguf_path(str(folder))

    def test_not_a_gguf(self, tmp_path):
        other = tmp_path / "model.bin"
        other.write_bytes(b"x")
        with pytest.raises(ValueError, match=r"must be a \.gguf"):
            lc._validated_gguf_path(str(other))


class TestKvCache:
    def test_quantised_types_map_to_ggml_ids(self, fake):
        assert lc._kv_cache_kwargs("q8_0") == {"type_k": 8, "type_v": 8}
        assert lc._kv_cache_kwargs("Q4_0") == {"type_k": 2, "type_v": 2}
        assert lc._kv_cache_kwargs("f16") == {}

    def test_without_the_library_it_falls_back_to_f16(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            assert lc._kv_cache_kwargs("q8_0") == {}
        assert "falling back to f16" in caplog.text


class TestSpeculativeDraft:
    def test_prompt_lookup(self, fake):
        draft, callable_ = lc._speculative_draft("prompt-lookup", 512, -1, False)
        assert draft is None and callable_ == ("lookup", 10, 2)

    def test_prompt_lookup_unavailable(self, fake, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_speculative", None)
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            assert lc._speculative_draft("prompt-lookup", 512, -1, False) == (None, None)
        assert "prompt-lookup decoding unavailable" in caplog.text

    def test_a_draft_model_behind_the_adapter(self, fake):
        draft, callable_ = lc._speculative_draft("/m/draft.gguf", 512, 0, False)
        assert isinstance(draft, FakeLlama)
        assert draft.kwargs == {
            "model_path": "/m/draft.gguf",
            "n_ctx": 512,
            "n_gpu_layers": 0,
            "verbose": False,
        }
        assert isinstance(callable_, lc._LlamaModelDraftAdapter)

    def test_a_draft_that_fails_to_load_is_dropped(self, fake, monkeypatch, caplog):
        monkeypatch.setattr(FakeLlama, "fail_with", RuntimeError("bad draft"))
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            assert lc._speculative_draft("/m/draft.gguf", 512, 0, False) == (None, None)
        assert "draft model load failed" in caplog.text

    def test_no_draft(self, fake):
        assert lc._speculative_draft(None, 512, 0, False) == (None, None)


class TestDraftAdapter:
    def test_without_sample_it_proposes_nothing_and_keeps_the_cache(self):
        class Draft:
            def __init__(self):
                self.evals: list[list[int]] = []

            def eval(self, ids):
                self.evals.append(list(ids))

            def reset(self):
                raise AssertionError("no divergence here")

        draft = Draft()
        adapter = lc._LlamaModelDraftAdapter(draft)
        assert adapter(np.array([1, 2, 3])).size == 0
        # The same ids again: nothing new to evaluate.
        assert adapter(np.array([1, 2, 3])).size == 0
        assert draft.evals == [[1, 2, 3]]
        assert adapter(None).size == 0


class TestSmallHelpers:
    def test_normalised_tool_calls(self):
        assert lc._normalised_tool_calls(None) is None
        assert lc._normalised_tool_calls([]) is None
        calls = [
            {"function": {"name": "a", "arguments": '{"x": 1}'}},
            {"function": {"name": "b", "arguments": "not json"}},
            {"function": {"name": "c", "arguments": {"y": 2}}},
            {},
        ]
        assert lc._normalised_tool_calls(calls) == [
            {"function": {"name": "a", "arguments": {"x": 1}}},
            {"function": {"name": "b", "arguments": {}}},
            {"function": {"name": "c", "arguments": {"y": 2}}},
            {"function": {"name": "", "arguments": {}}},
        ]

    def test_physical_cores_without_psutil(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setattr(lc.os, "cpu_count", lambda: 6)
        assert lc._physical_cores() == 6

    def test_offload_check(self, fake):
        assert lc._offloads_to_gpu(0) is False
        assert lc._offloads_to_gpu(-1) is True
        fake.lcpp.gpu = False
        assert lc._offloads_to_gpu(-1) is False

    def test_battery_warning_once(self, monkeypatch, caplog):
        import hfl.engine.power as power

        monkeypatch.setattr(lc, "_BATTERY_WARNED", False)
        monkeypatch.setattr(power, "on_battery", lambda: True)
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            lc._warn_if_on_battery()
            lc._warn_if_on_battery()
        assert caplog.text.count("Running on battery power") == 1

    def test_image_mime_types(self):
        assert lc._image_data_uri(b"RIFF\0\0\0\0WEBPxx").startswith("data:image/webp;base64,")
        assert lc._image_data_uri(b"GIF89a...").startswith("data:image/gif;")
        # Unknown bytes: PNG as a best effort.
        assert lc._image_data_uri(b"\0\1\2").startswith("data:image/png;base64,AAEC")

    def test_fold_system(self):
        assert lc._fold_system([{"role": "user", "content": "q"}]) == [
            {"role": "user", "content": "q"}
        ]
        folded = lc._fold_system(
            [
                {"role": "system", "content": "S1"},
                {"role": "system", "content": "S2"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q"},
            ]
        )
        assert folded == [
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "S1\n\nS2\n\nq"},
        ]
        # No user turn with text to fold into: the system text becomes one.
        parts = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        assert lc._fold_system([{"role": "system", "content": "S"}, *parts]) == [
            {"role": "user", "content": "S"},
            *parts,
        ]

    def test_glm_history_keeps_the_text_beside_a_call(self):
        template = "{{ metadata }} observation"
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"function": {"name": "w", "arguments": '{"c": 1}'}}],
            },
            {"role": "tool", "content": None},
        ]
        assert lc._history_for_template(msgs, template) == [
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "metadata": "w", "content": '{"c": 1}'},
            {"role": "observation", "content": ""},
        ]

    def test_tools_as_text_keeps_unparseable_arguments_as_text(self):
        msgs = [{"role": "assistant", "tool_calls": [{"name": "w", "arguments": "{oops"}]}]
        out = lc._tools_as_text(msgs, None)
        assert out == [
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "w", "arguments": "{oops"}\n</tool_call>',
            }
        ]

    def test_harmony_reply_is_stripped_for_gpt_oss(self):
        text = (
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>42<|return|>"
        )
        assert lc._strip_channel_markers(text, "gpt-oss") == "42"
        assert lc._strip_channel_markers("<|channel>final\nok<channel|>", "gemma4") == "ok"


class TestStreamFilter:
    def test_empty_chunks_and_bare_angle_brackets(self):
        filt = lc._Gemma4StreamFilter()
        assert filt.feed("") == ""
        # "<x" cannot grow into any marker: the "<" is plain text.
        assert filt.feed("a<x") == "a<x"

    def test_angle_brackets_inside_a_thought_are_dropped(self):
        filt = lc._Gemma4StreamFilter()
        out = filt.feed("<|think>a<b and c") + filt.feed("<think|>done")
        assert out == "done"

    def test_an_unclosed_thought_is_dropped_at_the_end(self):
        filt = lc._Gemma4StreamFilter()
        assert filt.feed("ok <|channel>thought\nsecret") == "ok "
        assert filt.flush() == ""

    def test_the_held_tail_is_emitted_at_the_end(self):
        chunks = list(lc._filter_gemma4_stream(iter(["Hi <|chan"])))
        assert chunks == ["Hi ", "<|chan"]


class TestProbeTemplate:
    """``_probe_template`` with a stand-in Jinja environment, in both venvs."""

    @pytest.fixture
    def env(self, monkeypatch):
        import hfl.models.chat_template as ct

        class Template:
            def __init__(self, source):
                self.source = source

            def render(self, messages, tools, **kw):
                if self.source == "broken":
                    raise ValueError("cannot render")
                if self.source == "no-system" and messages[0]["role"] == "system":
                    raise ValueError("Only user and assistant roles")
                out = " ".join(str(m["content"]) for m in messages)
                if "tools" in self.source:
                    out += " " + tools[0]["function"]["name"]
                return out

        env = types.SimpleNamespace(from_string=Template)
        monkeypatch.setattr(ct, "template_env", lambda: env)
        lc._probe_template.cache_clear()
        yield
        lc._probe_template.cache_clear()

    def test_tools_and_system_rendered(self, env):
        assert lc._probe_template("tools") == (True, True)
        assert lc._template_renders_tools("tools", None) is True
        assert lc._template_takes_system("tools") is True

    def test_a_template_refusing_system(self, env):
        assert lc._probe_template("no-system") == (False, False)
        assert lc._template_takes_system("no-system") is False

    def test_a_template_that_cannot_render(self, env):
        assert lc._probe_template("broken") == (False, True)

    def test_without_jinja(self, no_jinja):
        assert lc._probe_template("anything") == (False, True)

    def test_a_static_function_calling_format(self):
        assert lc._template_renders_tools("", "chatml-function-calling") is True
        assert lc._template_renders_tools("", None) is False


class TestGrammars:
    def test_chat_format_kwargs(self, fake):
        assert lc._chat_format_kwargs(None) == {}
        assert lc._chat_format_kwargs("json") == {"response_format": {"type": "json_object"}}
        schema = {"type": "object"}
        assert lc._chat_format_kwargs(schema) == {
            "response_format": {"type": "json_object", "schema": schema}
        }
        assert lc._chat_format_kwargs('GBNF:root ::= "a"') == {"grammar": ("gbnf", 'root ::= "a"')}
        assert lc._chat_format_kwargs("yaml") == {}

    def test_completion_grammar(self, fake):
        assert lc._completion_grammar(None) is None
        assert lc._completion_grammar("json") == ("gbnf", "json-gbnf")
        assert lc._completion_grammar({"type": "integer"}) == ("schema", '{"type": "integer"}')
        assert lc._completion_grammar("GBNF:root ::= x") == ("gbnf", "root ::= x")
        assert lc._completion_grammar("yaml") is None

    def test_grammar_source(self, fake):
        assert lc._grammar_source("GBNF:root ::= x") == "root ::= x"
        assert lc._grammar_source("json") is None
        assert lc._grammar_source({"type": "integer"}) == 'gbnf-of:{"type": "integer"}'
        assert lc._grammar_source({"unbuildable": True}) is None

    def test_an_unbuildable_grammar_is_refused_before_sampling(self, fake):
        from hfl.exceptions import ValidationError

        model = types.SimpleNamespace(_model=_Vocab(vocab=1234))
        with pytest.raises(ValidationError, match="cannot build a grammar"):
            lc._refuse_unbuildable_grammar(model, "GBNF:root ::= UNDEFINED")
        assert fake.pkg.freed == []
        lc._refuse_unbuildable_grammar(model, "GBNF:root ::= x")
        assert fake.pkg.freed == [99]
        # Plain JSON mode and a stand-in model are not checked.
        lc._refuse_unbuildable_grammar(model, "json")
        lc._refuse_unbuildable_grammar(types.SimpleNamespace(), "GBNF:root ::= UNDEFINED")
        assert fake.pkg.samplers == [b"root ::= UNDEFINED", b"root ::= x"]

    def test_special_tokens_rendered_only_while_asked(self, fake):
        model = FakeLlama()
        lc._render_special_tokens(model, True)
        assert model.detokenize([1]) == b"a!"
        lc._render_special_tokens(model, False)
        assert model.detokenize([1]) == b"a"
        assert "detokenize" not in model.__dict__


class TestTemplateFormatters:
    def test_every_template_goes_through_hfl_s_formatter(self, fake):
        model = FakeLlama()
        model.metadata = {
            "tokenizer.chat_template": "{% for m in messages %}{{ m }}{% endfor %}",
            "tokenizer.chat_template.tool_use": "{{ bos_token }}{{ tools }}",
            "tokenizer.chat_template.odd": 7,
            "general.name": "x",
        }
        formatters, added_bos = lc._install_template_formatters(model)
        assert added_bos is True
        assert [f.hfl_name for f in formatters] == [
            "chat_template.default",
            "chat_template.tool_use",
        ]
        assert formatters[0].template.startswith("{{ bos_token }}{% for")
        assert formatters[1].template == "{{ bos_token }}{{ tools }}"
        assert set(model._chat_handlers) == {"chat_template.default", "chat_template.tool_use"}
        assert (formatters[0].eos, formatters[0].bos, formatters[0].stop_ids) == (
            "</s>",
            "<s>",
            [EOS],
        )
        # The per-request variables reach the template.
        formatters[0].template_vars = {"enable_thinking": False}
        formatters[0](messages=[{"role": "user", "content": "q"}])
        assert formatters[0].calls[-1]["enable_thinking"] is False

    def test_no_bos_wanted(self, fake):
        model = FakeLlama()
        model._model = _Vocab(bos=-1)
        model.metadata = {"tokenizer.chat_template": "plain"}
        formatters, added_bos = lc._install_template_formatters(model)
        assert added_bos is False and formatters[0].template == "plain"

    def test_a_stand_in_without_a_vocabulary(self, fake):
        assert lc._install_template_formatters(types.SimpleNamespace()) == ([], False)


class TestVisionHandler:
    def test_handlers_by_architecture(self, fake):
        cases = {
            "qwen2vl": "Qwen25VLChatHandler",
            "moondream": "MoondreamChatHandler",
            "llava16": "Llava16ChatHandler",
            "llava": "Llava15ChatHandler",
            "mystery": "Llava15ChatHandler",
        }
        for arch, name in cases.items():
            handler = lc._build_vision_chat_handler(architecture=arch, clip_model_path="/p/mm.gguf")
            assert type(handler).__name__ == name
            assert handler.clip_model_path == "/p/mm.gguf"

    def test_gemma_without_its_handler_is_text_only(self, fake, caplog):
        with caplog.at_level(logging.WARNING, logger="hfl.engine.llama_cpp"):
            assert (
                lc._build_vision_chat_handler(architecture="gemma3", clip_model_path="/p") is None
            )
        assert "no Gemma vision handler" in caplog.text

    def test_gemma4_handler_preferred(self, fake, monkeypatch):
        monkeypatch.setattr(
            fake.pkg.llama_chat_format,
            "Gemma4ChatHandler",
            type("G4", (_Handler,), {}),
            raising=False,
        )
        handler = lc._build_vision_chat_handler(architecture="gemma4", clip_model_path="/p")
        assert type(handler).__name__ == "G4"

    def test_an_old_library_without_handlers(self, fake, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", None)
        assert lc._build_vision_chat_handler(architecture="llava", clip_model_path="/p") is None


class TestContextSizing:
    def test_small_contexts_are_left_alone(self, tmp_path):
        info = {"block_count": 2, "embedding_length": 8}
        assert lc._fit_ctx_to_memory(str(tmp_path / "x.gguf"), info, 2048) == 2048

    def test_an_unreadable_file_keeps_the_context(self, tmp_path, monkeypatch):
        import hfl.engine.memory as memory

        monkeypatch.setattr(memory, "HAS_PSUTIL", True)
        info = {"block_count": 2, "embedding_length": 8}
        assert lc._fit_ctx_to_memory(str(tmp_path / "missing.gguf"), info, 65536) == 65536

    def test_vram_probe_failure_leaves_auto(self, tmp_path, monkeypatch):
        import hfl.engine.vram as vram

        def boom():
            raise RuntimeError("no probe")

        monkeypatch.setattr(vram, "pick_ctx_size", boom)
        assert lc.resolve_n_ctx(str(tmp_path / "m.gguf"), None, 0, False) == 0

    def test_arch_cap_only_lowers(self, tmp_path):
        info = {"architecture": "gemma3", "max_context": 131072}
        # 4096 is under the 8192 cap: kept (and under the advertised max).
        assert lc.resolve_n_ctx(str(tmp_path / "m.gguf"), info, 4096, False) == 4096

    def test_missing_file_estimates_kv_only(self, tmp_path):
        info = {"block_count": 1, "embedding_length": 512}
        gb = lc._estimate_memory_required_gb(str(tmp_path / "none.gguf"), info, 1024)
        assert gb == pytest.approx(2 * 1 * 512 * 2 * 1024 / 1024**3)

    def test_preflight_with_nothing_to_measure(self, tmp_path, monkeypatch):
        import hfl.engine.memory as memory

        monkeypatch.delenv("HFL_DISABLE_MEMORY_PREFLIGHT", raising=False)
        monkeypatch.setattr(memory, "HAS_PSUTIL", True)
        monkeypatch.setattr(memory, "get_memory_snapshot", lambda: pytest.fail("must not measure"))
        lc._preflight_memory_check(str(tmp_path / "none.gguf"), None, 0, None)


# ------------------------------------------------------------------- load


class TestLoad:
    def test_without_llama_cpp_python(self, gguf, monkeypatch):
        monkeypatch.setattr(lc, "Llama", None)
        with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
            lc.LlamaCppEngine().load(str(gguf))

    def test_a_failed_load_is_logged_and_raised(self, fake, gguf, caplog):
        FakeLlama.fail_with = RuntimeError("bad tensor")
        with caplog.at_level(logging.ERROR, logger="hfl.engine.llama_cpp"):
            with pytest.raises(RuntimeError, match="bad tensor"):
                lc.LlamaCppEngine().load(str(gguf), n_ctx=512)
        assert "Failed to load model tiny.gguf" in caplog.text

    def test_the_gguf_template_is_served_with_bos_and_the_offload_reported(
        self, fake, gguf, caplog, no_jinja
    ):
        FakeLlama.metadata = {"tokenizer.chat_template": "{{ messages }}"}
        FakeLlama.emit_log = (
            b"ggml_metal_device_init: GPU name:   MTL0 (Apple M9)\n"
            b"load_tensors: offloaded 3/3 layers to GPU\n"
        )
        engine = lc.LlamaCppEngine()
        with caplog.at_level(logging.INFO, logger="hfl.engine.llama_cpp"):
            engine.load(str(gguf), n_ctx=512, n_gpu_layers=-1, kv_cache_type="q8_0")
        model = FakeLlama.instances[-1]
        assert model.kwargs["type_k"] == 8 and model.kwargs["n_ctx"] == 512
        assert model.chat_format == "chat_template.default"
        assert "chat_template.default" in model._chat_handlers
        assert engine._formatters[0].template == "{{ bos_token }}{{ messages }}"
        assert "HFL adds it" in caplog.text
        assert engine.acceleration == "MTL0 (Apple M9) · 3/3 layers on GPU"
        assert "Acceleration: MTL0 (Apple M9)" in caplog.text
        assert engine.context_size == 512 and engine.is_loaded
        assert engine.generates_on_all_cpu_cores is False
        assert engine.independent_instances is True
        assert engine.supports_structured_output is True
        assert engine.model_name == "tiny.gguf"
        # The template neither lists tools (no Jinja to probe it) nor is static.
        assert engine._chat_template == "{{ messages }}"
        assert engine._template_knows_tools is False

    def test_prompt_lookup_reaches_llama(self, fake, gguf):
        engine = lc.LlamaCppEngine()
        engine.load(str(gguf), n_ctx=512, draft_model_path="prompt-lookup")
        assert FakeLlama.instances[-1].kwargs["draft_model"] == ("lookup", 10, 2)
        assert engine._draft_model is None

    def test_unload_frees_the_draft_model_too(self, fake, gguf):
        engine = lc.LlamaCppEngine()
        engine.load(str(gguf), n_ctx=512, draft_model_path=str(gguf))
        assert isinstance(engine._draft_model, FakeLlama)
        engine.unload()
        assert engine._draft_model is None and not engine.is_loaded
        assert engine.context_size == 0 and engine.model_name == "tiny.gguf"


# ------------------------------------------------------------------- LoRA


class TestLora:
    def test_applied_in_order_and_removed(self, fake):
        engine = _engine(fake)
        engine.apply_lora("/a.gguf", 0.5, adapter_id="a")
        engine.apply_lora("/b.gguf", 1.0)
        sets = [[(h, s) for h, s in each] for each in fake.lcpp.lora_sets]
        assert sets == [[(0x1000, 0.5)], [(0x1000, 0.5), (0x2000, 1.0)]]
        assert engine._model.resets == 2
        engine.remove_lora("/b.gguf")
        assert fake.lcpp.lora_sets[-1] == [(0x1000, 0.5)]
        assert fake.lcpp.lora_freed == [0x2000]
        with pytest.raises(RuntimeError, match="not applied"):
            engine.remove_lora("zzz")

    def test_an_adapter_llama_cpp_cannot_open(self, fake):
        engine = _engine(fake)
        with pytest.raises(ValueError, match="bad.gguf is not a LoRA adapter"):
            engine.apply_lora("/x/bad.gguf", 1.0)
        assert engine._loras == []

    def test_a_refused_set_is_rolled_back(self, fake):
        engine = _engine(fake)
        fake.lcpp.set_result = -1
        with pytest.raises(RuntimeError, match="refused the LoRA adapter set"):
            engine.apply_lora("/a.gguf", 1.0)
        assert engine._loras == [] and fake.lcpp.lora_freed == [0x1000]

    def test_no_model(self, fake):
        with pytest.raises(RuntimeError, match="no model loaded"):
            lc.LlamaCppEngine().apply_lora("/a.gguf", 1.0)


# ------------------------------------------------------------- generation


class TestGenerate:
    def test_measured_timings_and_the_grammar(self, fake):
        engine = _engine(fake)
        result = engine.generate("hi", GenerationConfig(response_format="json", seed=3))
        call = engine._model.calls[-1]
        assert call["grammar"] == ("gbnf", "json-gbnf") and call["seed"] == 3
        assert isinstance(call["stopping_criteria"], fake.pkg.StoppingCriteriaList)
        assert result.text == "out" and result.tokens_prompt == 4
        # llama.cpp's own counters, not a proportional split.
        assert result.prompt_eval_duration == 3_000_000
        assert result.eval_duration == 5_000_000
        assert fake.lcpp.resets == ["ctx-ptr"]

    def test_zeroed_counters_fall_back_to_the_split(self, fake):
        fake.lcpp.perf = types.SimpleNamespace(t_p_eval_ms=0, n_p_eval=0, t_eval_ms=0, n_eval=0)
        engine = _engine(fake)
        result = engine.generate("hi")
        assert result.prompt_eval_duration + result.eval_duration == result.total_duration

    def test_the_stopping_criteria_follows_cancel(self, fake):
        engine = _engine(fake)
        stopping = engine._stopping()
        assert stopping[0](None, None) is False
        engine.cancel()
        assert stopping[0](None, None) is True

    def test_stream_counts_from_the_context(self, fake):
        engine = _engine(fake)
        model = engine._model
        chunks = [
            {"choices": [{"text": "a", "finish_reason": None}]},
            {"choices": [{"text": "", "finish_reason": None}]},
            {"choices": [{"text": "b", "finish_reason": "length"}]},
        ]
        model.reply = chunks
        model.n_tokens = 6
        stream = engine.generate_stream("p", GenerationConfig(response_format={"type": "integer"}))
        assert list(stream) == ["a", "b"]
        assert model.calls[-1]["grammar"] == ("schema", '{"type": "integer"}')
        # n_tokens stayed 6: prompt 6, completion 0 + the unevaluated last one.
        assert (stream.prompt_tokens, stream.completion_tokens) == (6, 1)


class TestLogprobs:
    def test_generate_with_logprobs_and_alternatives(self, fake):
        engine = _engine(fake)
        engine._model.script = [3, 4, EOS]
        result = engine.generate("hello world", GenerationConfig(logprobs=2, seed=5))
        assert result.text == "Hello" and result.stop_reason == "stop"
        assert result.tokens_generated == 2 and result.tokens_prompt == 3
        first = result.logprobs[0]
        assert first["token"] == "Hel" and first["bytes"] == list(b"Hel")
        expected = 3.0 - np.log(np.exp(np.arange(5.0)).sum())
        assert first["logprob"] == pytest.approx(expected)
        # Best two alternatives, most likely first: tokens 4 and 3.
        assert [alt["token"] for alt in first["top_logprobs"]] == ["lo", "Hel"]
        assert engine._model.seed == 5

    def test_a_stop_sequence_cuts_the_text(self, fake):
        engine = _engine(fake)
        engine._model.script = [3, 4, 1, 1]
        result = engine.generate("p", GenerationConfig(logprobs=0, stop=["lo"]))
        assert result.text == "Hel" and result.stop_reason == "stop"
        assert result.logprobs[0]["top_logprobs"] == []

    def test_max_tokens(self, fake):
        engine = _engine(fake)
        engine._model.script = [1, 1, 1, 1]
        result = engine.generate("p", GenerationConfig(logprobs=1, max_tokens=2))
        assert result.text == "aa" and result.stop_reason == "length"

    def test_cancel_stops_at_the_next_token(self, fake):
        engine = _engine(fake)
        model = engine._model
        model.script = [1, 1, 1]
        original = model.generate

        def generate(*args, **kwargs):
            for i, token in enumerate(original(*args, **kwargs)):
                if i == 1:
                    engine.cancel()
                yield token

        model.generate = generate
        result = engine.generate("p", GenerationConfig(logprobs=0))
        assert result.text == "a" and result.stop_reason == "stop"

    def test_an_unexpected_logits_layout_is_refused(self, fake):
        engine = _engine(fake)
        engine._model.script = [1]
        engine._model.n_vocab = lambda: 9
        with pytest.raises(NotImplementedError, match="unexpected logits layout"):
            engine.generate("p", GenerationConfig(logprobs=0))

    def test_not_with_a_response_format(self, fake):
        engine = _engine(fake)
        with pytest.raises(NotImplementedError, match="response format"):
            engine.generate("p", GenerationConfig(logprobs=1, response_format="json"))
        with pytest.raises(NotImplementedError, match="response format"):
            engine.chat(
                [ChatMessage("user", "q")], GenerationConfig(logprobs=1, response_format="json")
            )

    def test_chat_logprobs_through_the_template(self, fake):
        engine = _engine(fake)
        default = _Formatter("{{ bos_token }}t", "</s>", "<s>", [EOS])
        default.hfl_name = "chat_template.default"
        default.template_vars = {}
        engine._formatters = [default]
        engine._model.script = [3, EOS]
        result = engine.chat(
            [ChatMessage("user", "q")], GenerationConfig(logprobs=0, reasoning="off")
        )
        assert result.text == "Hel" and result.tokens_generated == 1
        # The template wrote BOS, so tokenize did not add another.
        assert result.tokens_prompt == 1
        assert default.template_vars == {
            "enable_thinking": False,
            "thinking": False,
            "reasoning_effort": "low",
        }


class TestCountPromptTokens:
    def test_counted_as_chat_renders_it(self, fake):
        engine = _engine(fake)
        default = _Formatter("t", "</s>", "<s>", [EOS])
        default.hfl_name = "chat_template.default"
        engine._formatters = [default]
        tools = [{"type": "function", "function": {"name": "w"}}]
        n = engine.count_prompt_tokens([ChatMessage("user", "q")], tools=tools)
        # "user:q\ntools:1" is two words, plus BOS (the template wrote none).
        assert n == 3
        assert default.calls[-1]["tools"] == tools

    def test_without_a_gguf_template(self, fake):
        engine = _engine(fake)
        with pytest.raises(NotImplementedError, match="no chat template"):
            engine.count_prompt_tokens([ChatMessage("user", "q")])

    def test_without_a_model(self):
        with pytest.raises(RuntimeError, match="no model loaded"):
            lc.LlamaCppEngine().count_prompt_tokens([ChatMessage("user", "q")])


class TestChat:
    def test_tools_formatters_and_stop_strings(self, fake):
        engine = _engine(fake, _architecture="gemma4")
        formatter = types.SimpleNamespace(template_vars={"stale": True})
        engine._formatters = [formatter]
        tools = [{"type": "function", "function": {"name": "w"}}]
        msgs = [
            ChatMessage("system", "be brief"),
            ChatMessage("tool", "18C", name="w", tool_call_id="c1"),
        ]
        engine.chat(msgs, GenerationConfig(stop=["END"], reasoning="off"), tools=tools)
        call = engine._model.chat_calls[-1]
        assert call["tools"] == tools
        assert call["stop"] == ["END", "<tool_call|>", "<|observation|>"]
        assert call["messages"][1] == {
            "role": "tool",
            "content": "18C",
            "name": "w",
            "tool_call_id": "c1",
        }
        assert "stale" not in formatter.template_vars
        assert formatter.template_vars.get("enable_thinking") is False

    def test_system_folded_for_a_template_without_one(self, fake):
        engine = _engine(fake, _template_takes_system=False)
        engine.chat([ChatMessage("system", "S"), ChatMessage("user", "q")])
        assert engine._model.chat_calls[-1]["messages"] == [{"role": "user", "content": "S\n\nq"}]

    def test_an_old_library_gets_the_call_without_tools(self, fake):
        engine = _engine(fake)
        engine._model.reject_kwargs = {"tools"}
        tools = [{"type": "function", "function": {"name": "w"}}]
        result = engine.chat([ChatMessage("user", "q")], GenerationConfig(), tools=tools)
        assert result.text == "hi"
        first, second = engine._model.chat_calls
        assert "tools" in first and "tools" not in second
        assert "detokenize" not in engine._model.__dict__

    def test_cancel_forces_the_end_of_the_reply(self, fake):
        engine = _engine(fake)
        engine.chat([ChatMessage("user", "q")])
        (processor,) = engine._model.chat_calls[-1]["logits_processor"]
        scores = np.zeros(5)
        assert processor(None, scores) is scores
        engine.cancel()
        forced = processor(None, scores)
        assert forced[EOS] == 0.0 and np.isneginf(np.delete(forced, EOS)).all()

    def test_no_cancel_processor_under_a_grammar(self, fake):
        engine = _engine(fake)
        engine.chat([ChatMessage("user", "q")], GenerationConfig(response_format="json"))
        assert engine._model.chat_calls[-1]["logits_processor"] is None


class TestChatStream:
    def _chunks(self, *texts, finish="stop"):
        out = [{"choices": [{"delta": {"content": t}, "finish_reason": None}]} for t in texts]
        out[-1]["choices"][0]["finish_reason"] = finish
        return out

    def test_tools_and_formatters_and_counts(self, fake):
        engine = _engine(fake)
        formatter = types.SimpleNamespace(template_vars={})
        engine._formatters = [formatter]
        engine._model.chat_reply = self._chunks("He", "", "llo")
        engine._model.n_tokens = 9
        tools = [{"type": "function", "function": {"name": "w"}}]
        stream = engine.chat_stream(
            [ChatMessage("user", "q")], GenerationConfig(reasoning="on"), tools=tools
        )
        assert list(stream) == ["He", "llo"]
        assert engine._model.chat_calls[-1]["tools"] == tools
        assert formatter.template_vars.get("enable_thinking") is True
        assert (stream.prompt_tokens, stream.completion_tokens) == (9, 0)

    def test_an_old_library_streams_without_tools(self, fake):
        engine = _engine(fake)
        engine._model.reject_kwargs = {"tools"}
        engine._model.chat_reply = self._chunks("ok")
        tools = [{"type": "function", "function": {"name": "w"}}]
        assert list(engine.chat_stream([ChatMessage("user", "q")], tools=tools)) == ["ok"]
        assert "tools" not in engine._model.chat_calls[-1]

    def test_gpt_oss_streams_without_its_reasoning(self, fake):
        engine = _engine(fake, _architecture="gpt-oss")
        engine._model.chat_reply = self._chunks(
            "<|channel|>analysis<|message|>hmm", "<|end|>", "<|start|>assistant",
            "<|channel|>final<|message|>42", "<|return|>",
        )  # fmt: skip
        assert "".join(engine.chat_stream([ChatMessage("user", "q")])) == "42"


class TestMessages:
    def test_images_become_content_parts(self):
        out = lc.LlamaCppEngine._messages_to_llama_cpp(
            [ChatMessage("user", "", images=[b"\x89PNGxx"])]
        )
        assert out[0]["content"][0]["type"] == "image_url"
        assert out[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
