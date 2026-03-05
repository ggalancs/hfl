# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for TTS engines (Bark and Coqui)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hfl.engine.base import AudioEngine, AudioResult, TTSConfig


class TestBarkEngine:
    """Tests for BarkEngine class."""

    def test_initialization(self):
        """Should initialize with default values."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()

        assert engine.is_loaded is False
        assert engine.model_name == ""

    def test_is_loaded_false_initially(self):
        """Should not be loaded initially."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        assert engine.is_loaded is False

    def test_implements_audio_engine(self):
        """Should implement AudioEngine interface."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        assert isinstance(engine, AudioEngine)

    def test_supported_voices(self):
        """Should list supported voices."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        voices = engine.supported_voices

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "v2/en_speaker_0" in voices

    def test_supported_languages(self):
        """Should list supported languages."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        languages = engine.supported_languages

        assert isinstance(languages, list)
        assert "en" in languages
        assert "es" in languages

    def test_synthesize_without_load_raises(self):
        """Should raise error if synthesize called before load."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()

        with pytest.raises(RuntimeError, match="not loaded"):
            engine.synthesize("Hello")

    @patch("hfl.engine.bark_engine.BarkEngine._encode_wav")
    def test_synthesize_with_mock_pipeline(self, mock_encode):
        """Should synthesize audio with mocked pipeline."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()

        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "audio": np.zeros(24000, dtype=np.float32),  # 1 second at 24kHz
            "sampling_rate": 24000,
        }
        engine._pipeline = mock_pipeline
        engine._model_name = "test-model"

        # Mock encoding
        mock_encode.return_value = b"RIFF...wav data..."

        config = TTSConfig(format="wav", sample_rate=24000)
        result = engine.synthesize("Hello world", config)

        assert isinstance(result, AudioResult)
        assert result.format == "wav"
        assert result.sample_rate == 24000
        mock_pipeline.assert_called_once_with("Hello world")

    @patch("hfl.engine.bark_engine.torch", create=True)
    def test_unload(self, mock_torch):
        """Should unload model."""
        from hfl.engine.bark_engine import BarkEngine

        # Mock torch.cuda
        mock_torch.cuda.is_available.return_value = False

        engine = BarkEngine()
        engine._pipeline = MagicMock()
        engine._model_name = "test"

        engine.unload()

        assert engine._pipeline is None
        assert engine._model_name == ""
        assert engine.is_loaded is False

    def test_encode_wav_manual(self):
        """Should encode WAV manually when soundfile not available."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()

        # Create a simple sine wave
        sample_rate = 22050
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        wav_bytes = engine._encode_wav_manual(audio, sample_rate)

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert wav_bytes[12:16] == b"fmt "

    def test_adjust_speed(self):
        """Should adjust audio speed."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        audio = np.ones(1000, dtype=np.float32)

        # Speed up 2x should halve length
        faster = engine._adjust_speed(audio, 2.0)
        assert len(faster) == 500

        # Slow down 0.5x should double length
        slower = engine._adjust_speed(audio, 0.5)
        assert len(slower) == 2000

    def test_synthesize_stream(self):
        """Should yield audio chunks."""
        from hfl.engine.bark_engine import BarkEngine

        engine = BarkEngine()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "audio": np.zeros(24000, dtype=np.float32),
            "sampling_rate": 24000,
        }
        engine._pipeline = mock_pipeline
        engine._model_name = "test"
        engine._sample_rate = 24000

        # Patch synthesize directly to avoid torch import issues
        mock_result = MagicMock()
        mock_result.audio = b"x" * 2048

        with patch.object(engine, "synthesize", return_value=mock_result):
            chunks = list(engine.synthesize_stream("Hello"))

        assert len(chunks) == 2  # 2048 bytes / 1024 chunk size
        assert all(isinstance(c, bytes) for c in chunks)


class TestCoquiEngine:
    """Tests for CoquiEngine class."""

    def test_initialization(self):
        """Should initialize with default values."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()

        assert engine.is_loaded is False
        assert engine.model_name == ""

    def test_is_loaded_false_initially(self):
        """Should not be loaded initially."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()
        assert engine.is_loaded is False

    def test_implements_audio_engine(self):
        """Should implement AudioEngine interface."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()
        assert isinstance(engine, AudioEngine)

    def test_synthesize_without_load_raises(self):
        """Should raise error if synthesize called before load."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()

        with pytest.raises(RuntimeError, match="not loaded"):
            engine.synthesize("Hello")

    @patch("hfl.engine.coqui_engine.torch", create=True)
    def test_unload(self, mock_torch):
        """Should unload model."""
        from hfl.engine.coqui_engine import CoquiEngine

        # Mock torch.cuda
        mock_torch.cuda.is_available.return_value = False

        engine = CoquiEngine()
        engine._tts = MagicMock()
        engine._model_name = "test"

        engine.unload()

        assert engine._tts is None
        assert engine._model_name == ""
        assert engine.is_loaded is False

    @patch("hfl.engine.coqui_engine.CoquiEngine._encode_wav")
    def test_synthesize_with_mock_tts(self, mock_encode):
        """Should synthesize audio with mocked TTS."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()

        # Mock the TTS object
        mock_tts = MagicMock()
        mock_tts.tts.return_value = np.zeros(22050, dtype=np.float32)
        engine._tts = mock_tts
        engine._model_name = "test-model"
        engine._sample_rate = 22050

        # Mock encoding
        mock_encode.return_value = b"RIFF...wav data..."

        config = TTSConfig(format="wav", sample_rate=22050)
        result = engine.synthesize("Hello world", config)

        assert isinstance(result, AudioResult)
        assert result.format == "wav"

    def test_supported_voices_xtts(self):
        """Should return XTTS voices when loaded."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()
        engine._model_name = "xtts_v2"

        # Mock TTS with speakers attribute
        mock_tts = MagicMock()
        mock_tts.speakers = ["speaker_1", "speaker_2", "speaker_3"]
        engine._tts = mock_tts

        voices = engine.supported_voices

        assert isinstance(voices, list)
        assert len(voices) == 3
        assert "speaker_1" in voices

    def test_supported_languages_xtts(self):
        """Should return XTTS languages when loaded."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()
        engine._model_name = "xtts_v2"

        # Mock TTS with languages attribute
        mock_tts = MagicMock()
        mock_tts.languages = ["en", "es", "fr", "de"]
        engine._tts = mock_tts

        languages = engine.supported_languages

        assert isinstance(languages, list)
        assert "en" in languages
        assert len(languages) == 4

    def test_supported_voices_not_loaded(self):
        """Should return default voice when not loaded."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()

        voices = engine.supported_voices

        assert voices == ["default"]

    def test_supported_languages_not_loaded(self):
        """Should return default language when not loaded."""
        from hfl.engine.coqui_engine import CoquiEngine

        engine = CoquiEngine()

        languages = engine.supported_languages

        assert languages == ["en"]


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = TTSConfig()

        assert config.voice == "default"
        assert config.speed == 1.0
        assert config.language == "en"
        assert config.sample_rate == 22050
        assert config.format == "wav"

    def test_custom_values(self):
        """Should accept custom values."""
        config = TTSConfig(
            voice="v2/en_speaker_1",
            speed=1.5,
            language="es",
            sample_rate=44100,
            format="mp3",
        )

        assert config.voice == "v2/en_speaker_1"
        assert config.speed == 1.5
        assert config.language == "es"
        assert config.sample_rate == 44100
        assert config.format == "mp3"


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_creation(self):
        """Should create AudioResult."""
        result = AudioResult(
            audio=b"audio bytes",
            sample_rate=22050,
            duration=1.5,
            format="wav",
        )

        assert result.audio == b"audio bytes"
        assert result.sample_rate == 22050
        assert result.duration == 1.5
        assert result.format == "wav"
        assert result.metadata == {}

    def test_with_metadata(self):
        """Should include metadata."""
        result = AudioResult(
            audio=b"data",
            sample_rate=24000,
            duration=2.0,
            format="mp3",
            metadata={"model": "bark", "voice": "en_speaker_0"},
        )

        assert result.metadata["model"] == "bark"
        assert result.metadata["voice"] == "en_speaker_0"
