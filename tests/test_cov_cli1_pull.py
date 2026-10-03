# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``hfl pull`` / ``hfl run`` helpers in ``hfl.cli.main``: what each failure
says, and which path a pull or a chat takes. The Hub, the downloader and the
engines are faked; nothing is downloaded or loaded."""

from __future__ import annotations

import errno
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from hfl.cli import main
from hfl.cli.commands._utils import console
from hfl.hub.resolver import ResolvedModel
from hfl.models.manifest import ModelManifest

runner = CliRunner()


@pytest.fixture
def out(capsys):
    """What the shared Rich console printed (it writes to sys.stdout)."""

    def read() -> str:
        return capsys.readouterr().out

    return read


def _manifest(name="m1", model_type="llm", **kw) -> ModelManifest:
    return ModelManifest(
        name=name,
        repo_id=kw.pop("repo_id", "org/m1"),
        local_path=kw.pop("local_path", "/nonexistent/m1.gguf"),
        format=kw.pop("format", "gguf"),
        model_type=model_type,
        size_bytes=kw.pop("size_bytes", 1024**3),
        **kw,
    )


# --- _resolve_or_exit / _show_resolved_or_exit -------------------------------


class TestResolve:
    def test_offline_says_offline_not_the_socket_error(self, monkeypatch, out):
        def offline(*a, **k):
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided")

        monkeypatch.setattr("hfl.hub.resolver.resolve", offline)
        with pytest.raises(typer.Exit) as exc:
            main._resolve_or_exit("org/model", "Q4_K_M", None)
        assert exc.value.exit_code == 1
        text = out()
        assert "you appear to be offline" in text
        assert "nodename" not in text

    def test_any_other_error_is_named(self, monkeypatch, out):
        def boom(*a, **k):
            raise RuntimeError("weird failure")

        monkeypatch.setattr("hfl.hub.resolver.resolve", boom)
        with pytest.raises(typer.Exit) as exc:
            main._resolve_or_exit("org/model", "Q4_K_M", None)
        assert exc.value.exit_code == 1
        text = out()
        assert "Error resolving model" in text and "weird failure" in text

    def test_pinned_revision_and_commit_are_shown(self, out):
        resolved = ResolvedModel(
            repo_id="org/model",
            format="gguf",
            filename="m.gguf",
            revision="v1.2",
            commit_sha="0123456789abcdef0123",
            pipeline_tag="text-generation",
        )
        main._show_resolved_or_exit(resolved)
        text = out()
        assert "Revision: v1.2" in text
        assert "Commit: 0123456789ab" in text
        assert "0123456789abc" not in text  # only the first 12 characters

    def test_main_revision_is_not_shown(self, out):
        main._show_resolved_or_exit(ResolvedModel(repo_id="org/model", format="gguf"))
        assert "Revision" not in out()

    def test_an_unsupported_type_is_refused_before_downloading(self, out):
        resolved = ResolvedModel(
            repo_id="org/sd", format="safetensors", pipeline_tag="text-to-image"
        )
        with pytest.raises(typer.Exit) as exc:
            main._show_resolved_or_exit(resolved)
        assert exc.value.exit_code == 1
        assert "Unsupported model type" in out()


# --- _license_or_exit ---------------------------------------------------------


class TestLicense:
    def test_accepted_returns_the_license_and_a_timestamp(self, monkeypatch):
        info = object()
        monkeypatch.setattr(
            "hfl.hub.license_checker.check_model_license", lambda repo, revision=None: info
        )
        monkeypatch.setattr("hfl.hub.license_checker.require_user_acceptance", lambda i, repo: True)
        got, accepted_at = main._license_or_exit("org/m", skip_license=False, revision="abc")
        assert got is info
        assert accepted_at and "T" in accepted_at  # an ISO timestamp

    def test_skip_license_checks_nothing(self, monkeypatch):
        def never(*a, **k):
            raise AssertionError("checked a license with --skip-license")

        monkeypatch.setattr("hfl.hub.license_checker.check_model_license", never)
        assert main._license_or_exit("org/m", skip_license=True) == (None, None)

    def test_unverifiable_license_declined_exits_0(self, monkeypatch, out):
        def fails(*a, **k):
            raise RuntimeError("license endpoint down")

        monkeypatch.setattr("hfl.hub.license_checker.check_model_license", fails)
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
        with pytest.raises(typer.Exit) as exc:
            main._license_or_exit("org/m", skip_license=False)
        assert exc.value.exit_code == 0
        assert "Could not verify license" in out()

    def test_unverifiable_license_accepted_goes_on_without_one(self, monkeypatch):
        def fails(*a, **k):
            raise RuntimeError("license endpoint down")

        monkeypatch.setattr("hfl.hub.license_checker.check_model_license", fails)
        monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
        assert main._license_or_exit("org/m", skip_license=False) == (None, None)

    def test_declining_the_license_cancels_without_a_second_question(self, monkeypatch, out):
        monkeypatch.setattr("hfl.hub.license_checker.check_model_license", lambda *a, **k: object())
        monkeypatch.setattr(
            "hfl.hub.license_checker.require_user_acceptance", lambda i, repo: False
        )
        asked: list[str] = []

        def confirm(text, **k):
            asked.append(text)
            return True  # "yes, continue without a license"

        monkeypatch.setattr(typer, "confirm", confirm)
        with pytest.raises(typer.Exit) as exc:
            main._license_or_exit("org/m", skip_license=False)
        assert exc.value.exit_code == 0
        assert asked == []
        assert "Could not verify license" not in out()


# --- _download_or_exit --------------------------------------------------------


def _download_raising(monkeypatch, exc: BaseException):
    def pull_model(resolved):
        raise exc

    monkeypatch.setattr("hfl.hub.downloader.pull_model", pull_model)


class TestDownload:
    resolved = ResolvedModel(repo_id="org/gated-model", format="gguf")

    def test_success_prints_the_path(self, monkeypatch, out, tmp_path):
        monkeypatch.setattr("hfl.hub.downloader.pull_model", lambda r: tmp_path)
        assert main._download_or_exit(self.resolved) == tmp_path
        assert str(tmp_path) in out().replace("\n", "")

    def test_gated_repo_says_how_to_get_access(self, monkeypatch, out):
        from huggingface_hub.utils import GatedRepoError

        _download_raising(monkeypatch, GatedRepoError("403", response=MagicMock()))
        with pytest.raises(typer.Exit) as exc:
            main._download_or_exit(self.resolved)
        assert exc.value.exit_code == 1
        text = out()
        assert "This model is gated" in text
        assert "org/gated-model" in text

    def test_retries_exhausted_name_the_real_cause(self, monkeypatch, out):
        from hfl.utils.retry import RetryExhausted

        cause = ConnectionResetError("peer reset the connection")
        _download_raising(monkeypatch, RetryExhausted("gave up", last_exception=cause))
        with pytest.raises(typer.Exit) as exc:
            main._download_or_exit(self.resolved)
        assert exc.value.exit_code == 1
        text = out()
        assert "Download failed" in text and "peer reset the connection" in text
        assert "gave up" not in text

    def test_retries_exhausted_without_a_cause_names_the_wrapper(self, monkeypatch, out):
        from hfl.utils.retry import RetryExhausted

        _download_raising(monkeypatch, RetryExhausted("gave up after 3"))
        with pytest.raises(typer.Exit):
            main._download_or_exit(self.resolved)
        assert "gave up after 3" in out()

    def test_integrity_failure_says_which_file(self, monkeypatch, out):
        from hfl.exceptions import DownloadIntegrityError

        _download_raising(
            monkeypatch, DownloadIntegrityError("org/m", "model.gguf", "a" * 64, "b" * 64)
        )
        with pytest.raises(typer.Exit) as exc:
            main._download_or_exit(self.resolved)
        assert exc.value.exit_code == 1
        text = out()
        assert "Download failed" in text and "model.gguf" in text

    def test_a_full_disk_is_said_plainly(self, monkeypatch, out, temp_config):
        _download_raising(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
        with pytest.raises(typer.Exit) as exc:
            main._download_or_exit(self.resolved)
        assert exc.value.exit_code == 1
        assert "The disk filled up" in out()

    def test_an_unknown_failure_propagates(self, monkeypatch):
        _download_raising(monkeypatch, ValueError("not ours"))
        with pytest.raises(ValueError, match="not ours"):
            main._download_or_exit(self.resolved)


def test_disk_full_and_hub_lost_follow_the_cause_chain():
    try:
        try:
            raise RuntimeError("No space left on device (os error 28)")
        except RuntimeError as inner:
            raise ValueError("wrapper") from inner
    except ValueError as outer:
        assert main._disk_full(outer)
        assert not main._hub_lost(outer)
    try:
        try:
            raise httpx.ReadTimeout("slow")
        except httpx.ReadTimeout:
            raise ValueError("wrapper")  # noqa: B904 - implicit context on purpose
    except ValueError as outer:
        assert main._hub_lost(outer)
        assert not main._disk_full(outer)


# --- what a finished pull prints ------------------------------------------------


class TestPullMessages:
    def _ready(self, detected):
        from hfl.converter.formats import ModelType

        manifest = SimpleNamespace(alias="emb", name="org-emb", display_size="1 GB")
        main._print_ready(manifest, getattr(ModelType, detected))

    def test_an_embedding_model_points_at_the_api(self, out):
        self._ready("EMBEDDING")
        assert "/api/embed" in out()

    def test_a_tts_model_points_at_hfl_tts(self, out):
        self._ready("TTS")
        assert 'hfl tts emb "..."' in out()

    def test_a_model_without_alias_is_named_only(self, out):
        from hfl.converter.formats import ModelType

        manifest = SimpleNamespace(alias=None, name="org-x", display_size="2 GB")
        main._print_ready(manifest, ModelType.LLM)
        text = out()
        assert "org-x (2 GB)" in text and "hfl run" not in text

    def test_step_errors(self, out):
        from hfl.hub.pull_service import PullStepError

        main._print_pull_step_error(
            PullStepError("errors.unsupported_model_type", type="Video"), "org/v"
        )
        text = out()
        assert "Unsupported model type" in text and "Video" in text

        main._print_pull_step_error(
            PullStepError("errors.cannot_convert_gguf", reason="no tokenizer"), "org/model-x"
        )
        text = out()
        assert "Cannot convert to GGUF" in text and "no tokenizer" in text
        assert "hfl search model-x --gguf" in text

        main._print_pull_step_error(
            PullStepError("errors.conversion_failed", reason="cmake died"), "org/m"
        )
        text = out()
        assert "GGUF conversion failed" in text and "cmake died" in text

    def _kept(self, kept, model_type="llm", notes=()):
        from hfl.converter.formats import ModelType
        from hfl.hub.pull_service import Kept

        finished = SimpleNamespace(
            kept=getattr(Kept, kept), model_type=ModelType(model_type), notes=list(notes)
        )
        main._print_kept(finished, "Q4_K_M")

    def test_kept_as_is_for_a_non_llm(self, out):
        self._kept("AS_IS", model_type="tts")
        assert "No GGUF conversion needed" in out()

    def test_kept_as_is_for_an_llm_says_nothing(self, out):
        self._kept("AS_IS")
        assert out() == ""

    def test_mlx_repo_without_mlx(self, out):
        self._kept("MLX_ELSEWHERE")
        assert "not convertible to GGUF" in out().replace("\n", " ")

    def test_template_notes(self, out):
        self._kept("CONVERTED", notes=["template_recovered", "template_missing"])
        text = out()
        assert "Recovered missing chat_template" in text
        assert "has no chat_template" in text


# --- pull through the CLI -------------------------------------------------------


def _fake_pull_chain(monkeypatch, temp_config, resolved, seen):
    from hfl.converter.formats import ModelType
    from hfl.hub.pull_service import Finished, Kept

    def resolve(model, quantization=None, revision=None):
        seen["resolve"] = (model, quantization, revision)
        return resolved

    monkeypatch.setattr("hfl.hub.resolver.resolve", resolve)
    monkeypatch.setattr("hfl.hub.downloader.pull_model", lambda r: temp_config.models_dir)
    monkeypatch.setattr("hfl.hub.pull_service.disk_space", lambda r, c: None)
    monkeypatch.setattr(
        "hfl.hub.pull_service.finish_download",
        lambda *a, **k: Finished(path=temp_config.models_dir, model_type=ModelType.LLM,
                                 kept=Kept.AS_IS),
    )  # fmt: skip

    def register(resolved, finished, **kw):
        seen["register"] = kw
        return SimpleNamespace(alias=kw["alias"], name="org-model", display_size="1 GB")

    monkeypatch.setattr("hfl.hub.pull_service.register_pulled", register)


class TestPullShortName:
    def test_a_short_name_pulls_the_choice_under_its_alias(self, monkeypatch, temp_config):
        seen: dict = {}
        resolved = ResolvedModel(repo_id="org/qwen3-8b-gguf", format="gguf")
        _fake_pull_chain(monkeypatch, temp_config, resolved, seen)
        choice = SimpleNamespace(reference="org/qwen3-8b-gguf:Q5_K_M", quantization="Q5_K_M")
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: choice)
        result = runner.invoke(main.app, ["pull", "qwen3:8b", "--skip-license", "-y"])
        assert result.exit_code == 0, result.stdout
        assert seen["resolve"] == ("org/qwen3-8b-gguf:Q5_K_M", "Q5_K_M", None)
        from hfl.hub.shortname import alias_for

        assert seen["register"]["alias"] == alias_for("qwen3:8b")
        assert seen["register"]["quantize"] == "Q5_K_M"

    def test_an_explicit_alias_wins_over_the_short_name(self, monkeypatch, temp_config):
        seen: dict = {}
        resolved = ResolvedModel(repo_id="org/qwen3-8b-gguf", format="gguf")
        _fake_pull_chain(monkeypatch, temp_config, resolved, seen)
        choice = SimpleNamespace(reference="org/qwen3-8b-gguf:Q4_K_M", quantization="Q4_K_M")
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: choice)
        result = runner.invoke(
            main.app, ["pull", "qwen3:8b", "--skip-license", "-y", "--alias", "mine"]
        )
        assert result.exit_code == 0, result.stdout
        assert seen["register"]["alias"] == "mine"

    def test_a_short_name_with_no_match_exits_1(self, monkeypatch, temp_config):
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: None)
        result = runner.invoke(main.app, ["pull", "nosuchmodel", "--skip-license"])
        assert result.exit_code == 1
        assert "nosuchmodel" in result.stdout


# --- _choose_short_name / _from_short_name -------------------------------------


class TestShortNames:
    def test_offline_lookup_says_offline(self, monkeypatch, out):
        def offline(name):
            raise httpx.ConnectError("no route")

        monkeypatch.setattr("hfl.hub.shortname.find_options", offline)
        with pytest.raises(typer.Exit) as exc:
            main._choose_short_name("qwen3", assume_yes=True)
        assert exc.value.exit_code == 1
        assert "you appear to be offline" in out()

    def test_another_lookup_failure_propagates(self, monkeypatch):
        def broken(name):
            raise KeyError("bug")

        monkeypatch.setattr("hfl.hub.shortname.find_options", broken)
        with pytest.raises(KeyError):
            main._choose_short_name("qwen3", assume_yes=True)

    def test_a_local_copy_without_alias_gets_the_short_names_alias(self, monkeypatch):
        from hfl.hub.shortname import alias_for

        local = SimpleNamespace(name="org-qwen3-q4", alias=None)
        registry = MagicMock()
        registry.get.return_value = None
        registry.find_pulled.return_value = local
        choice = SimpleNamespace(repo_id="org/qwen3", quantization="Q4_K_M", reference="x")
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: choice)
        got = main._from_short_name("qwen3", lambda: registry, assume_yes=True)
        assert got is local
        registry.set_alias.assert_called_once_with("org-qwen3-q4", alias_for("qwen3"))

    def test_a_local_copy_with_an_alias_keeps_it(self, monkeypatch):
        local = SimpleNamespace(name="org-qwen3-q4", alias="already")
        registry = MagicMock()
        registry.get.return_value = None
        registry.find_pulled.return_value = local
        choice = SimpleNamespace(repo_id="org/qwen3", quantization="Q4_K_M", reference="x")
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: choice)
        assert main._from_short_name("qwen3", lambda: registry, assume_yes=True) is local
        registry.set_alias.assert_not_called()

    def test_nothing_chosen_is_none(self, monkeypatch):
        registry = MagicMock()
        registry.get.return_value = None
        monkeypatch.setattr(main, "_choose_short_name", lambda name, assume_yes: None)
        assert main._from_short_name("qwen3", lambda: registry, assume_yes=True) is None


# --- _conversion_level ----------------------------------------------------------


def _levels(monkeypatch, *, params=None, choice=None, terminal=False, answers=()):
    monkeypatch.setattr("hfl.engine.selector._mlx_preferred", lambda: False)
    monkeypatch.setattr("hfl.hub.hw_profile.get_hw_profile", lambda: object())
    monkeypatch.setattr(main, "stdin_is_terminal", lambda: terminal)

    def estimate(repo_id, api=None):
        if isinstance(params, Exception):
            raise params
        return params

    monkeypatch.setattr("hfl.hub.params.estimate_params", estimate)
    if choice is not None:
        monkeypatch.setattr("hfl.hub.quant_choice.choose", lambda *a, **k: choice)
    replies = iter(answers)
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(replies))


def _choice(*, split=False, recommended="Q4_K_M"):
    from hfl.hub.quant_choice import LADDER, Choice, Level

    levels = [Level(name=n, size_gb=1.0, fits=n != "F16", fits_split=True) for n in LADDER]
    return Choice(
        memory="cuda",
        fast_gb=8.0,
        total_gb=24.0,
        levels=levels,
        recommended=recommended,
        split=split,
    )


SAFETENSORS_LLM = ResolvedModel(
    repo_id="org/llm", format="safetensors", pipeline_tag="text-generation"
)


class TestConversionLevel:
    def test_size_lookup_failing_falls_back_to_q4(self, monkeypatch, out):
        _levels(monkeypatch, params=RuntimeError("offline"))
        assert main._conversion_level(SAFETENSORS_LLM, "auto", assume_yes=True) == (
            "Q4_K_M",
            "auto",
        )
        assert "Could not tell org/llm's size" in out()

    def test_a_split_recommendation_is_explained(self, monkeypatch, out):
        params = SimpleNamespace(total_b=30.0, active_b=None)
        _levels(monkeypatch, params=params, choice=_choice(split=True))
        level, fmt = main._conversion_level(SAFETENSORS_LLM, "auto", assume_yes=True)
        assert (level, fmt) == ("Q4_K_M", "auto")
        text = out()
        assert "does not fit the GPU alone" in text
        assert "(+ RAM: ~24 GB)" in text

    def test_a_bad_typed_level_is_asked_again(self, monkeypatch, out):
        params = SimpleNamespace(total_b=7.0, active_b=None)
        _levels(
            monkeypatch,
            params=params,
            choice=_choice(),
            terminal=True,
            answers=["q9_x", " q6_k "],
        )
        level, _ = main._conversion_level(SAFETENSORS_LLM, "auto", assume_yes=False)
        assert level == "Q6_K"
        assert "Choose one of: F16, Q8_0" in out()


# --- run: type checks, sessions, the chat loop ----------------------------------


class TestCheckChatModel:
    def test_an_unsupported_type_is_refused_with_the_hint(self, out):
        with pytest.raises(typer.Exit) as exc:
            main._check_chat_model(_manifest(model_type="stt"), "whisper")
        assert exc.value.exit_code == 1
        text = out()
        assert "Wrong model type" in text and "STT (Speech-to-Text)" in text
        assert "Only LLM and TTS models can be run" in text

    def test_a_tts_model_points_at_hfl_tts(self, out):
        with pytest.raises(typer.Exit):
            main._check_chat_model(_manifest(model_type="tts"), "bark")
        assert 'hfl tts <model> "text"' in out()

    def test_an_embedding_model_is_refused_without_the_tts_hint(self, out):
        with pytest.raises(typer.Exit) as exc:
            main._check_chat_model(_manifest(model_type="embedding"), "emb")
        assert exc.value.exit_code == 1
        text = out()
        assert "Embeddings" in text and "hfl tts" not in text

    def test_an_llm_passes(self):
        main._check_chat_model(_manifest(model_type="llm"), "m1")


class FakeEngine:
    def __init__(self, replies, interrupt=False):
        self.replies = list(replies)
        self.interrupt = interrupt
        self.seen: list[list] = []
        self.unloaded = False

    def chat_stream(self, messages, config):
        self.seen.append([(m.role, m.content) for m in messages])
        yield from self.replies.pop(0)
        if self.interrupt:
            raise KeyboardInterrupt

    def unload(self):
        self.unloaded = True


def _typed(monkeypatch, lines):
    """Feed the chat prompt; EOF after the last line (or the exception given)."""
    items = iter(lines)

    def fake_input(prompt=""):
        try:
            item = next(items)
        except StopIteration:
            raise EOFError from None
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(console, "input", fake_input)


class TestRunSessions:
    def _run(self, monkeypatch, engine, args, lines):
        manifest = _manifest()
        monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: manifest)
        monkeypatch.setattr(main, "_memory_check_or_exit", lambda m, c: None)
        monkeypatch.setattr(main, "_load_for_chat", lambda *a: engine)
        _typed(monkeypatch, lines)
        return runner.invoke(main.app, ["run", "m1", *args])

    def test_a_session_is_saved_after_each_exchange_and_resumed(self, monkeypatch, temp_config):
        from hfl.core.sessions import load_session

        engine = FakeEngine([["Hel", "lo"]])
        result = self._run(monkeypatch, engine, ["--session", "s1"], ["hi", "/exit"])
        assert result.exit_code == 0, result.stdout
        assert "Recording a new session as 's1'" in result.stdout
        assert "Session saved to" in result.stdout
        saved = load_session("s1")
        assert saved.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert engine.unloaded

        engine2 = FakeEngine([["again"]])
        result = self._run(
            monkeypatch, engine2, ["--session", "s1", "--system", "be brief"], ["more"]
        )
        assert result.exit_code == 0, result.stdout
        assert "Resumed session 's1' with 2 messages" in result.stdout
        # The resumed conversation reached the model, with the new system prompt.
        sent = engine2.seen[0]
        assert sent[:2] == [("user", "hi"), ("assistant", "Hello")]
        assert ("system", "be brief") in sent and sent[-1] == ("user", "more")
        assert [m["content"] for m in load_session("s1").messages][-2:] == ["more", "again"]

    def test_a_resumed_session_keeps_its_own_system_prompt(self, monkeypatch, temp_config):
        from hfl.core.sessions import ChatSession, load_session, save_session

        save_session(
            ChatSession(
                name="s2",
                model="m1",
                messages=[{"role": "system", "content": "old system"}],
            )
        )
        engine = FakeEngine([["ok"]])
        result = self._run(
            monkeypatch, engine, ["--session", "s2", "--system", "new system"], ["q"]
        )
        assert result.exit_code == 0, result.stdout
        roles = [r for r, _ in engine.seen[0]]
        assert roles.count("system") == 1
        assert engine.seen[0][0] == ("system", "old system")
        assert len(load_session("s2").messages) == 3

    def test_ctrl_c_stops_a_reply_and_at_the_prompt_ends_the_chat(self, monkeypatch, temp_config):
        engine = FakeEngine([["partial"]], interrupt=True)
        result = self._run(monkeypatch, engine, [], ["", "hi", KeyboardInterrupt()])
        assert result.exit_code == 0, result.stdout
        assert "partial" in result.stdout
        assert len(engine.seen) == 1  # the empty line was skipped
        assert engine.unloaded


# --- tts / audio ---------------------------------------------------------------


def test_tts_refuses_a_model_that_is_neither_tts_nor_llm(monkeypatch, out):
    monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: _manifest(model_type="stt"))
    with pytest.raises(typer.Exit) as exc:
        main._load_tts_or_exit("whisper")
    assert exc.value.exit_code == 1
    text = out()
    assert "Wrong model type" in text and "TTS (Text-to-Speech)" in text
    assert "hfl run" not in text


def test_play_audio_uses_sounddevice_when_installed(monkeypatch):
    played: list = []
    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda buf: (["samples"], 0)  # no rate in the file
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.play = lambda data, rate: played.append((data, rate))
    sounddevice.wait = lambda: played.append("waited")
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)

    def no_player(*a, **k):
        raise AssertionError("fell through to the system player")

    monkeypatch.setattr("subprocess.run", no_player)
    main._play_audio(b"RIFF....", 16000)
    assert played == [(["samples"], 16000), "waited"]


def test_declared_extras_without_package_metadata(monkeypatch):
    import importlib.metadata as metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "metadata", missing)
    order, packages = main._declared_extras()
    assert packages == {}
    assert order[-1] == "all"
    assert order[:-1] == [e for e in main._EXTRA_ORDER if e != "all"]


def test_tts_without_its_backend_exits_1(monkeypatch, out):
    from hfl.engine.selector import MissingDependencyError

    def select(path):
        raise MissingDependencyError("the TTS backend is not installed")

    monkeypatch.setattr(main, "_local_or_pulled", lambda *a, **k: _manifest(model_type="tts"))
    monkeypatch.setattr(main, "_memory_check_or_exit", lambda m, c: None)
    monkeypatch.setattr("hfl.engine.selector.select_tts_engine", select)
    with pytest.raises(typer.Exit) as exc:
        main._load_tts_or_exit("bark")
    assert exc.value.exit_code == 1
    text = out()
    assert "Missing dependency" in text and "TTS backend is not installed" in text
