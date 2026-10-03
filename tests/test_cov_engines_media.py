# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The media engines — Bark and Coqui TTS, Whisper ASR, Diffusers images —
against stand-in backends seated in ``sys.modules``: loading, device and
dtype choice, synthesis with resampling / speed / every output format, the
streaming fallbacks, the openai-whisper path and image generation.

None of torch, transformers, TTS, soundfile, pydub, whisper or diffusers
is installed in CI; each test seats only the fakes it needs (``None`` in
``sys.modules`` makes an import fail, which is how "not installed" is
simulated even where the real package exists).
"""

from __future__ import annotations

import base64
import contextlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hfl.engine.base import TTSConfig

# ----------------------------------------------------------------------
# Shared fakes
# ----------------------------------------------------------------------


class _Tensor:
    """Enough of a torch tensor for the resample paths."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def unsqueeze(self, dim):
        return _Tensor(self.a[None])

    def float(self):
        return self

    def squeeze(self):
        return _Tensor(self.a.squeeze())

    def numpy(self):
        return self.a


def _torch(cuda=False, mps=False):
    calls: list[str] = []
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"  # type: ignore[attr-defined]
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.from_numpy = lambda a: _Tensor(a)  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda, empty_cache=lambda: calls.append("cuda.empty_cache")
    )
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    torch.calls = calls  # type: ignore[attr-defined]
    return torch


def _torchaudio(monkeypatch):
    audio = types.ModuleType("torchaudio")

    class Resample:
        def __init__(self, orig, target):
            self.ratio = target / orig

        def __call__(self, t):
            n = int(t.a.shape[-1] * self.ratio)
            return _Tensor(np.zeros((1, n), dtype=np.float32))

    audio.transforms = types.SimpleNamespace(Resample=Resample)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchaudio", audio)


def _soundfile(monkeypatch):
    written: list[tuple] = []
    sf = types.ModuleType("soundfile")

    def write(buffer, audio, rate, format, subtype):  # noqa: A002
        written.append((len(audio), rate, format, subtype))
        buffer.write(f"{format}:{rate}".encode())

    sf.write = write  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    return written


def _pydub(monkeypatch):
    pydub = types.ModuleType("pydub")

    class AudioSegment:
        def __init__(self, data, frame_rate, sample_width, channels):
            self.data, self.rate = data, frame_rate
            assert sample_width == 2 and channels == 1

        def export(self, buffer, format):  # noqa: A002
            buffer.write(f"{format}:{self.rate}:{len(self.data)}".encode())

    pydub.AudioSegment = AudioSegment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydub", pydub)


def _no(monkeypatch, *names):
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


# ======================================================================
# Bark
# ======================================================================


def _bark_transformers(monkeypatch, *, rate=24000, fail=False):
    seen: dict = {}
    tf = types.ModuleType("transformers")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path):
            seen["processor"] = path
            return "processor"

    class _Model:
        def __init__(self):
            self.generation_config = types.SimpleNamespace(sample_rate=rate)

        def to(self, device):
            seen["device"] = device
            return self

    class BarkModel:
        @staticmethod
        def from_pretrained(path, dtype):
            if fail:
                raise OSError("no weights")
            seen["dtype"] = dtype
            return _Model()

    tf.AutoProcessor, tf.BarkModel = AutoProcessor, BarkModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tf)
    return seen


class TestBark:
    def test_load_picks_cuda_and_half_precision(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        seen = _bark_transformers(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _torch(cuda=True))
        engine = BarkEngine()
        engine.load("suno/bark-small")
        assert seen == {"processor": "suno/bark-small", "dtype": "float16", "device": "cuda"}
        assert engine.is_loaded and engine.model_name == "suno/bark-small"
        assert engine._sample_rate == 24000

    def test_load_on_cpu_in_full_precision_without_a_rate(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        seen = _bark_transformers(monkeypatch, rate=None)
        monkeypatch.setitem(sys.modules, "torch", _torch())
        engine = BarkEngine()
        engine._sample_rate = 1
        engine.load("suno/bark", device="cpu")
        assert seen["dtype"] == "float32" and seen["device"] == "cpu"
        assert engine._sample_rate == 24000  # Bark's native rate

    def test_an_explicit_dtype_needs_no_torch(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        seen = _bark_transformers(monkeypatch, rate=16000)
        _no(monkeypatch, "torch")
        engine = BarkEngine()
        engine.load("suno/bark", device="cpu", dtype="bf16")
        assert seen["dtype"] == "bf16" and engine._sample_rate == 16000

    def test_a_failed_load_is_logged_and_raised(self, monkeypatch, caplog):
        from hfl.engine.bark_engine import BarkEngine

        _bark_transformers(monkeypatch, fail=True)
        engine = BarkEngine()
        with caplog.at_level("ERROR"), pytest.raises(OSError, match="no weights"):
            engine.load("suno/bark", device="cpu", dtype="f32")
        assert "Failed to load Bark model" in caplog.text and not engine.is_loaded

    def test_without_transformers_the_error_says_what_to_install(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "transformers")
        with pytest.raises(ImportError, match=r"hfl\[tts\]"):
            BarkEngine().load("suno/bark")

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_detect_device(self, monkeypatch, cuda, mps, expected):
        from hfl.engine.bark_engine import BarkEngine

        monkeypatch.setitem(sys.modules, "torch", _torch(cuda=cuda, mps=mps))
        assert BarkEngine()._detect_device() == expected

    def test_detect_device_without_torch_is_cpu(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "torch")
        assert BarkEngine()._detect_device() == "cpu"

    def test_unload_empties_the_cuda_cache(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        torch = _torch(cuda=True)
        monkeypatch.setitem(sys.modules, "torch", torch)
        engine = BarkEngine()
        engine._model = object()
        engine.unload()
        assert torch.calls == ["cuda.empty_cache"] and not engine.is_loaded
        engine.unload()  # nothing loaded: nothing to do
        assert torch.calls == ["cuda.empty_cache"]

    def test_unload_without_torch_still_unloads(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "torch")
        engine = BarkEngine()
        engine._model = object()
        engine.unload()
        assert not engine.is_loaded

    @staticmethod
    def _loaded(monkeypatch, audio):
        from hfl.engine.bark_engine import BarkEngine

        monkeypatch.setitem(sys.modules, "torch", _torch())

        class _Inputs(dict):
            def to(self, device):
                return self

        class _Out:
            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return audio

        engine = BarkEngine()
        engine._processor = lambda text, voice_preset: _Inputs(input_ids=text)
        engine._model = types.SimpleNamespace(generate=lambda **kw: _Out())
        engine._model_name = "suno/bark"
        return engine

    def test_a_batched_multichannel_output_takes_the_first_channel(self, monkeypatch):
        written = _soundfile(monkeypatch)
        audio = np.zeros((2, 1, 480), dtype=np.float32)  # squeezes to (2, 480)
        engine = self._loaded(monkeypatch, audio)
        result = engine.synthesize("hi", TTSConfig(sample_rate=24000, format="wav"))
        assert written == [(480, 24000, "WAV", "PCM_16")]
        assert result.audio == b"WAV:24000" and result.duration == pytest.approx(0.02)
        assert result.metadata == {"model": "suno/bark"}

    def test_resample_with_torchaudio_and_speed_up(self, monkeypatch):
        _torchaudio(monkeypatch)
        _soundfile(monkeypatch)
        engine = self._loaded(monkeypatch, np.zeros((1, 2400), dtype=np.float32))
        result = engine.synthesize("hi", TTSConfig(sample_rate=12000, speed=2.0))
        # 2400 samples at 24 kHz -> 1200 at 12 kHz -> 600 at double speed.
        assert result.sample_rate == 12000 and result.duration == pytest.approx(0.05)

    def test_resample_falls_back_to_interpolation_without_torchaudio(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "torchaudio")
        monkeypatch.setitem(sys.modules, "torch", _torch())
        out = BarkEngine()._resample(np.arange(100, dtype=np.float32), 100, 50)
        assert len(out) == 50 and out[0] == 0 and out[-1] == 99

    def test_mp3_ogg_and_an_unknown_format(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _pydub(monkeypatch)
        written = _soundfile(monkeypatch)
        engine = BarkEngine()
        audio = np.array([0.0, 2.0, -2.0], dtype=np.float32)  # clipped to [-1, 1]
        assert engine._encode_audio(audio, 8000, "mp3") == b"mp3:8000:6"
        assert engine._encode_audio(audio, 8000, "ogg") == b"OGG:8000"
        assert engine._encode_audio(audio, 8000, "flac") == b"WAV:8000"  # default
        assert [w[2:] for w in written] == [("OGG", "VORBIS"), ("WAV", "PCM_16")]

    def test_wav_without_soundfile_is_encoded_by_hand(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "soundfile")
        wav = BarkEngine()._encode_wav(np.array([1.0, -1.0], dtype=np.float32), 8000)
        assert wav[:4] == b"RIFF" and len(wav) == 44 + 4

    def test_mp3_and_ogg_without_their_libraries_say_what_to_install(self, monkeypatch):
        from hfl.engine.bark_engine import BarkEngine

        _no(monkeypatch, "pydub", "soundfile")
        audio = np.zeros(4, dtype=np.float32)
        with pytest.raises(ImportError, match="pydub"):
            BarkEngine()._encode_mp3(audio, 8000)
        with pytest.raises(ImportError, match="soundfile"):
            BarkEngine()._encode_ogg(audio, 8000)


# ======================================================================
# Coqui
# ======================================================================


class _FakeTTS:
    """``TTS.api.TTS``: records its construction and synthesis calls."""

    made: list[dict] = []

    def __init__(self, model_path, progress_bar, gpu):
        _FakeTTS.made.append({"model": model_path, "progress_bar": progress_bar, "gpu": gpu})
        self.synthesizer = None
        self.calls: list[dict] = []
        self.wav: object = [0.0] * 220

    def tts(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return self.wav


def _tts_module(monkeypatch, synthesizer=None):
    _FakeTTS.made = []
    pkg = types.ModuleType("TTS")
    api = types.ModuleType("TTS.api")

    class TTS(_FakeTTS):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.synthesizer = synthesizer

    api.TTS = TTS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "TTS", pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", api)
    # The compat shim imports torch + transformers; keep it a no-op here
    # (a test that needs torch seats its own fake afterwards).
    _no(monkeypatch, "torch", "transformers.pytorch_utils")


class TestCoqui:
    def test_load_detects_the_gpu_and_reads_the_output_rate(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _tts_module(monkeypatch, types.SimpleNamespace(output_sample_rate=24000))
        monkeypatch.setitem(sys.modules, "torch", _torch(cuda=True))
        engine = CoquiEngine()
        engine.load("tts_models/en/ljspeech/vits")
        assert _FakeTTS.made == [
            {"model": "tts_models/en/ljspeech/vits", "progress_bar": True, "gpu": True}
        ]
        assert engine._sample_rate == 24000 and engine.model_name.endswith("vits")

    def test_load_reads_the_rate_from_the_tts_config(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        config = types.SimpleNamespace(audio={"sample_rate": 16000})
        _tts_module(monkeypatch, types.SimpleNamespace(tts_config=config))
        engine = CoquiEngine()
        engine.load("m", gpu=False, progress_bar=False)
        assert _FakeTTS.made[0]["gpu"] is False and _FakeTTS.made[0]["progress_bar"] is False
        assert engine._sample_rate == 16000

    def test_a_synthesizer_without_rate_information_keeps_the_default(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _tts_module(monkeypatch, types.SimpleNamespace())
        engine = CoquiEngine()
        engine.load("m", gpu=False)
        assert engine._sample_rate == 22050

    def test_without_coqui_the_error_says_what_to_install(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _no(monkeypatch, "torch", "TTS", "TTS.api", "transformers.pytorch_utils")
        with pytest.raises(ImportError, match=r"hfl\[coqui\]"):
            CoquiEngine().load("m")

    def test_has_cuda_without_torch_is_false(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _no(monkeypatch, "torch")
        assert CoquiEngine()._has_cuda() is False

    def test_compat_shim_is_a_no_op_without_transformers(self, monkeypatch):
        from hfl.engine.coqui_engine import _transformers5_compat

        _no(monkeypatch, "torch")
        assert _transformers5_compat() is None

    def test_unload_empties_the_cuda_cache_and_tolerates_no_torch(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        torch = _torch(cuda=True)
        monkeypatch.setitem(sys.modules, "torch", torch)
        engine = CoquiEngine()
        engine._tts, engine._model_name = object(), "m"
        engine.unload()
        assert torch.calls == ["cuda.empty_cache"] and engine.model_name == ""
        _no(monkeypatch, "torch")
        engine._tts = object()
        engine.unload()
        assert not engine.is_loaded
        engine.unload()  # nothing loaded

    @staticmethod
    def _loaded(**attrs):
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()
        tts = _FakeTTS("m", True, False)
        for k, v in attrs.items():
            setattr(tts, k, v)
        engine._tts, engine._model_name, engine._sample_rate = tts, "m", 22050
        return engine, tts

    def test_language_speaker_speed_and_resampling_reach_the_model(self, monkeypatch):
        _torchaudio(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _torch())
        written = _soundfile(monkeypatch)
        engine, tts = self._loaded(is_multi_lingual=True, speakers=["Ana", "Bo"])
        cfg = TTSConfig(voice="Ana", language="es", speed=1.5, sample_rate=11025)
        result = engine.synthesize("hola", cfg)
        assert tts.calls == [{"text": "hola", "language": "es", "speaker": "Ana", "speed": 1.5}]
        assert result.sample_rate == 11025 and written[0][1] == 11025
        assert result.duration == pytest.approx(110 / 11025)
        assert result.metadata == {"model": "m", "language": "es", "voice": "Ana"}

    def test_a_single_speaker_model_gets_no_speaker(self, monkeypatch):
        _soundfile(monkeypatch)
        engine, tts = self._loaded(speakers=None, is_multi_lingual=False)
        engine.synthesize("hi", TTSConfig(voice="Ana", sample_rate=22050))
        assert tts.calls == [{"text": "hi"}]

    def test_resample_falls_back_to_interpolation(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _no(monkeypatch, "torch")
        out = CoquiEngine()._resample(np.arange(10, dtype=np.float32), 10, 20)
        assert len(out) == 20 and out[-1] == 9

    def test_voices_and_languages_of_a_model_without_lists(self):
        engine, _ = self._loaded(speakers=[], languages=[])
        assert engine.supported_voices == ["default"]
        assert engine.supported_languages == ["en"]
        from hfl.engine.coqui_engine import CoquiEngine

        idle = CoquiEngine()
        assert idle._is_multilingual() is False and idle._supports_speakers() is False

    def test_stream_chunks_a_non_streaming_model(self, monkeypatch):
        _soundfile(monkeypatch)
        engine, tts = self._loaded()
        engine._encode_wav = lambda audio, rate: b"x" * 2500  # type: ignore[method-assign]
        chunks = list(engine.synthesize_stream("hi", TTSConfig(sample_rate=22050)))
        assert [len(c) for c in chunks] == [1024, 1024, 452]

    def test_xtts_never_writes_a_file(self):
        """tts_to_file writes output.wav and returns its path; the stream used
        it as audio. It is not called at all now."""
        engine, tts = self._loaded()
        engine._model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        def _writes(**kw):
            raise AssertionError("tts_to_file must not be used to stream")

        tts.tts_to_file = _writes
        engine._encode_wav = lambda audio, rate: b"z" * 1100  # type: ignore[method-assign]
        chunks = list(engine.synthesize_stream("hi", TTSConfig(sample_rate=22050)))
        assert [len(c) for c in chunks] == [1024, 76]

    def test_xtts_falls_back_to_chunks_when_streaming_fails(self, monkeypatch):
        engine, tts = self._loaded()
        engine._model_name = "xtts_v2"

        def _fail(**kw):
            raise RuntimeError("no streaming here")

        tts.tts_to_file = _fail
        engine._encode_wav = lambda audio, rate: b"y" * 1500  # type: ignore[method-assign]
        chunks = list(engine.synthesize_stream("hi", TTSConfig(voice="Ana", sample_rate=22050)))
        assert [len(c) for c in chunks] == [1024, 476]

    def test_xtts_stream_yields_audio_bytes(self):
        engine, tts = self._loaded()
        engine._model_name = "xtts_v2"
        # What Coqui's TTS.tts_to_file does: write ``file_path`` and return it.
        tts.tts_to_file = lambda text, file_path="output.wav", **kw: file_path
        chunks = list(engine.synthesize_stream("hi"))
        assert chunks and all(isinstance(c, bytes) for c in chunks)

    def test_encoders(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _pydub(monkeypatch)
        written = _soundfile(monkeypatch)
        engine = CoquiEngine()
        audio = np.array([0.5, 3.0], dtype=np.float64)
        assert engine._encode_audio(audio, 8000, "mp3") == b"mp3:8000:4"
        assert engine._encode_audio(audio, 8000, "ogg") == b"OGG:8000"
        assert engine._encode_audio(audio, 8000, "aiff") == b"WAV:8000"
        assert engine._encode_audio(audio, 8000, "wav") == b"WAV:8000"
        assert [w[2] for w in written] == ["OGG", "WAV", "WAV"]

    def test_encoders_without_their_libraries(self, monkeypatch):
        from hfl.engine.coqui_engine import CoquiEngine

        _no(monkeypatch, "pydub", "soundfile")
        engine = CoquiEngine()
        audio = np.zeros(3, dtype=np.float32)
        wav = engine._encode_wav(audio, 8000)
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and len(wav) == 44 + 6
        with pytest.raises(ImportError, match="pydub"):
            engine._encode_mp3(audio, 8000)
        with pytest.raises(ImportError, match="soundfile"):
            engine._encode_ogg(audio, 8000)

    def test_list_coqui_models(self, monkeypatch):
        from hfl.engine.coqui_engine import list_coqui_models

        manage = types.ModuleType("TTS.utils.manage")

        class ModelManager:
            def list_tts_models(self):
                return ["tts_models/en/ljspeech/vits"]

            def list_vocoder_models(self):
                return ["vocoder_models/en/ljspeech/hifigan_v2"]

        manage.ModelManager = ModelManager  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))
        monkeypatch.setitem(sys.modules, "TTS.utils", types.ModuleType("TTS.utils"))
        monkeypatch.setitem(sys.modules, "TTS.utils.manage", manage)
        assert list_coqui_models() == {
            "tts_models": ["tts_models/en/ljspeech/vits"],
            "vocoder_models": ["vocoder_models/en/ljspeech/hifigan_v2"],
        }
        _no(monkeypatch, "TTS.utils.manage")
        assert list_coqui_models() == {"error": "coqui-tts not installed"}


# ======================================================================
# Whisper (openai-whisper backend)
# ======================================================================


@pytest.fixture
def openai_whisper(monkeypatch, tmp_path):
    """Only openai-whisper installed; its cache under ``tmp_path``."""
    seen: dict = {"loaded": [], "transcribed": []}
    whisper = types.ModuleType("whisper")

    class _Model:
        result: object = None

        def transcribe(self, path, language=None):
            seen["transcribed"].append((Path(path).read_bytes(), language, path))
            return _Model.result

    def load_model(name):
        seen["loaded"].append(name)
        return _Model()

    whisper.load_model = load_model  # type: ignore[attr-defined]
    _no(monkeypatch, "faster_whisper")
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    seen["model"] = _Model
    return seen


class TestOpenAIWhisper:
    def test_loads_and_transcribes_through_a_temp_file_it_removes(self, openai_whisper):
        from hfl.engine.whisper_engine import WhisperEngine

        engine = WhisperEngine()
        engine.load("tiny")
        assert engine.backend == "openai_whisper" and engine.model_name == "tiny"
        openai_whisper["model"].result = {
            "text": "  hola mundo ",
            "language": "es",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hola"}, {"text": "mundo"}],
        }
        result = engine.transcribe(b"RIFFdata", language="es", include_segments=True)
        audio, language, path = openai_whisper["transcribed"][0]
        assert audio == b"RIFFdata" and language == "es" and path.endswith(".wav")
        assert not Path(path).exists()  # unlinked after use
        assert result.text == "hola mundo" and result.language == "es"
        assert [(s.start, s.end, s.text) for s in result.segments] == [
            (0.0, 1.5, "hola"),
            (0.0, 0.0, "mundo"),
        ]

    def test_an_empty_result_is_an_empty_transcript(self, openai_whisper):
        from hfl.engine.whisper_engine import WhisperEngine

        engine = WhisperEngine()
        engine.load("tiny")
        openai_whisper["model"].result = None
        result = engine.transcribe(b"x")
        assert result.text == "" and result.language is None and result.segments is None

    def test_local_files_only_refuses_a_model_not_on_disk(self, openai_whisper):
        from hfl.engine.whisper_engine import WhisperEngine

        with pytest.raises(FileNotFoundError, match="not on disk"):
            WhisperEngine().load("small", local_files_only=True)
        assert openai_whisper["loaded"] == []

    def test_local_files_only_loads_a_cached_checkpoint(self, openai_whisper, tmp_path):
        from hfl.engine.whisper_engine import WhisperEngine

        (tmp_path / "whisper").mkdir()
        (tmp_path / "whisper" / "small.pt").write_bytes(b"ckpt")
        engine = WhisperEngine()
        engine.load("small", local_files_only=True)
        assert openai_whisper["loaded"] == ["small"]
        engine.unload()
        assert not engine.is_loaded and engine.backend is None
        assert engine.model_name == "whisper"

    def test_no_backend_after_all_is_a_clear_error(self, monkeypatch):
        from hfl.engine import whisper_engine

        _no(monkeypatch, "faster_whisper", "whisper")
        monkeypatch.setattr(whisper_engine, "is_available", lambda: True)
        with pytest.raises(RuntimeError, match="no backend available"):
            whisper_engine.WhisperEngine().load("tiny")


# ======================================================================
# Diffusers
# ======================================================================


class _Image:
    size = (64, 32)

    def save(self, buf, format):  # noqa: A002
        buf.write(b"\x89PNG-" + format.encode())


def _diffusers(monkeypatch, *, cuda=False, mps=False):
    seen: dict = {}
    diffusers = types.ModuleType("diffusers")

    class _Pipeline:
        def to(self, device):
            seen["device"] = device

        def __call__(self, prompt, **kwargs):
            seen["call"] = {"prompt": prompt, **kwargs}
            return types.SimpleNamespace(images=[_Image()])

    class DiffusionPipeline:
        @staticmethod
        def from_pretrained(model, torch_dtype, local_files_only):
            seen["load"] = (model, torch_dtype, local_files_only)
            return _Pipeline()

    diffusers.DiffusionPipeline = DiffusionPipeline  # type: ignore[attr-defined]
    torch = _torch(cuda=cuda, mps=mps)

    class Generator:
        def __init__(self, device):
            self.device, self.seed = device, None

        def manual_seed(self, seed):
            self.seed = seed
            return self

        def initial_seed(self):
            return 1234

    torch.Generator = Generator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return seen


class TestDiffusers:
    def test_available_only_with_diffusers_and_torch(self, monkeypatch):
        from hfl.engine.diffusers_engine import is_available

        _no(monkeypatch, "diffusers")
        assert is_available() is False
        monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
        _no(monkeypatch, "torch")
        assert is_available() is False
        _diffusers(monkeypatch)
        assert is_available() is True

    def test_load_without_the_extra_says_what_to_install(self, monkeypatch):
        from hfl.engine.diffusers_engine import DiffusersEngine

        _no(monkeypatch, "diffusers")
        with pytest.raises(RuntimeError, match="imagegen"):
            DiffusersEngine().load("sdxl")

    @pytest.mark.parametrize(
        ("cuda", "mps", "device", "dtype"),
        [
            (True, False, "cuda", "float16"),
            (False, True, "mps", "float16"),
            (False, False, "cpu", "float32"),
        ],
    )
    def test_load_picks_the_device_and_dtype(self, monkeypatch, cuda, mps, device, dtype):
        from hfl.engine.diffusers_engine import DiffusersEngine

        seen = _diffusers(monkeypatch, cuda=cuda, mps=mps)
        engine = DiffusersEngine()
        engine.load("stabilityai/sdxl-turbo", local_files_only=True)
        assert seen["load"] == ("stabilityai/sdxl-turbo", dtype, True)
        assert seen["device"] == device and engine.is_loaded
        assert engine.model_name == "stabilityai/sdxl-turbo"

    def test_generate_with_a_seed_returns_a_png(self, monkeypatch):
        from hfl.engine.diffusers_engine import DiffusersEngine

        seen = _diffusers(monkeypatch)
        engine = DiffusersEngine()
        engine.load("m", device="cpu")
        result = engine.generate("a cat", negative_prompt="dogs", width=64, height=32, steps=2,
                                 guidance_scale=1.0, seed=7)  # fmt: skip
        call = seen["call"]
        assert call["prompt"] == "a cat" and call["negative_prompt"] == "dogs"
        assert call["num_inference_steps"] == 2 and call["generator"].seed == 7
        assert base64.b64decode(result.image_png_base64) == b"\x89PNG-PNG"
        assert (result.seed, result.width, result.height) == (7, 64, 32)

    def test_generate_without_a_seed_reports_the_one_used(self, monkeypatch):
        from hfl.engine.diffusers_engine import DiffusersEngine

        _diffusers(monkeypatch)
        engine = DiffusersEngine()
        engine.load("m", device="cpu")
        assert engine.generate("x").seed == 1234

    def test_generate_needs_a_model_and_unload_drops_it(self, monkeypatch):
        from hfl.engine.diffusers_engine import DiffusersEngine

        _diffusers(monkeypatch)
        engine = DiffusersEngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate("x")
        engine.load("m", device="cpu")
        engine.unload()
        assert not engine.is_loaded and engine.model_name == "diffusers"
