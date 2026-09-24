# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""`hfl tts` and `hfl speak`, restored.

Both were hidden and short-circuited ("TTS functionality temporarily
disabled") and then deleted in a CLI refactor, while the README kept
advertising them. The engines work (measured 2026-09-24: Bark wrote 4.63 s
of speech, RMS 1636/32767), so the commands are back on the current engine
API, with the same resolution and memory check as `hfl run`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hfl.converter.formats import ModelType
from hfl.engine.base import AudioResult

AUDIO = b"RIFF" + b"\0" * 64


class FakeTTS:
    def __init__(self):
        self.loaded_from = None
        self.config = None
        self.unloaded = False

    def load(self, path, **kw):
        self.loaded_from = path

    def synthesize(self, text, config):
        self.config = config
        return AudioResult(audio=AUDIO, sample_rate=24000, duration=1.5, format=config.format)

    def unload(self):
        self.unloaded = True


@pytest.fixture
def cli(monkeypatch, temp_config, tmp_path):
    from hfl.cli import main

    # `hfl tts` writes output.wav in the working directory by default; a
    # regression that lets a refused command through must not litter the repo.
    monkeypatch.chdir(tmp_path)

    engine = FakeTTS()
    state = SimpleNamespace(engine=engine, created=0, model_type=ModelType.TTS, manifest=None)
    state.manifest = SimpleNamespace(name="bark", local_path="/x/bark")

    def select(path, **kw):
        state.created += 1
        return engine

    monkeypatch.setattr(main, "_local_or_pulled", lambda model, cls: state.manifest)
    monkeypatch.setattr(main, "get_model_type", lambda m: state.model_type)
    monkeypatch.setattr(main, "_memory_check_or_exit", lambda m, ctx: None)
    monkeypatch.setattr("hfl.engine.selector.select_tts_engine", select)
    state.run = lambda *args: CliRunner().invoke(main.app, list(args))
    state.main = main
    return state


class TestTts:
    def test_writes_the_audio_with_the_requested_settings(self, cli, tmp_path):
        out = tmp_path / "hola.wav"
        result = cli.run(
            "tts",
            "bark",
            "Hola mundo",
            "-o",
            str(out),
            "--lang",
            "es",
            "--speed",
            "0.9",
            "--rate",
            "16000",
            "--format",
            "wav",
        )
        assert result.exit_code == 0, result.stdout
        assert out.read_bytes() == AUDIO
        cfg = cli.engine.config
        assert (cfg.language, cfg.speed, cfg.sample_rate, cfg.format) == ("es", 0.9, 16000, "wav")
        assert cli.engine.loaded_from == "/x/bark" and cli.engine.unloaded
        assert "1.50s" in result.stdout and "24000" in result.stdout

    def test_an_unknown_model_is_reported(self, cli):
        cli.manifest = None
        result = cli.run("tts", "nope", "hi")
        assert result.exit_code == 1 and cli.created == 0

    def test_a_chat_model_is_pointed_at_hfl_run(self, cli):
        cli.model_type = ModelType.LLM
        result = cli.run("tts", "llama", "hi")
        assert result.exit_code == 1
        assert "hfl run" in result.stdout
        assert cli.created == 0, "an engine was built for a model of the wrong type"

    @pytest.mark.parametrize("args", [("--speed", "5"), ("--speed", "0.1"), ("--format", "flac")])
    def test_bad_options_are_refused_before_loading(self, cli, args):
        result = cli.run("tts", "bark", "hi", *args)
        assert result.exit_code == 2 and cli.created == 0


class TestSpeak:
    def test_plays_what_it_synthesized(self, cli, monkeypatch):
        played = []
        monkeypatch.setattr(
            cli.main, "_play_audio", lambda audio, rate: played.append((audio, rate))
        )
        result = cli.run("speak", "bark", "hi", "--lang", "es")
        assert result.exit_code == 0, result.stdout
        assert played == [(AUDIO, 24000)]
        assert cli.engine.config.format == "wav" and cli.engine.config.language == "es"

    def test_when_playback_fails_the_audio_is_kept(self, cli, monkeypatch):
        def fail(audio, rate):
            raise RuntimeError("no device")

        monkeypatch.setattr(cli.main, "_play_audio", fail)
        result = cli.run("speak", "bark", "hi")
        assert result.exit_code == 0
        saved = [w for w in result.stdout.split() if w.endswith(".wav")]
        assert saved, result.stdout
        from pathlib import Path

        assert Path(saved[-1]).read_bytes() == AUDIO
        Path(saved[-1]).unlink()


class TestPlayer:
    def test_without_sounddevice_the_system_player_is_used(self, monkeypatch):
        import builtins
        import subprocess

        from hfl.cli import main

        real_import = builtins.__import__

        def no_sounddevice(name, *a, **k):
            if name in ("sounddevice", "soundfile"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        calls = []
        monkeypatch.setattr(builtins, "__import__", no_sounddevice)
        monkeypatch.setattr("shutil.which", lambda p: "/usr/bin/afplay" if p == "afplay" else None)
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd))
        main._play_audio(AUDIO, 22050)
        assert calls and calls[0][0] == "afplay" and calls[0][1].endswith(".wav")

    def test_no_player_at_all_raises_so_the_caller_saves_the_file(self, monkeypatch):
        import builtins

        from hfl.cli import main

        real_import = builtins.__import__

        def no_sounddevice(name, *a, **k):
            if name in ("sounddevice", "soundfile"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_sounddevice)
        monkeypatch.setattr("shutil.which", lambda p: None)
        with pytest.raises(RuntimeError, match="hfl\\[audio\\]"):
            main._play_audio(AUDIO, 22050)
