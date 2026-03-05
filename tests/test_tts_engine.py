# SPDX-License-Identifier: HRUL-1.0
"""Tests for TTS engine classes and related functionality."""

import numpy as np
import pytest

from hfl.converter.formats import ModelType, detect_model_type
from hfl.engine.base import (
    AudioEngine,
    AudioResult,
    TTSConfig,
)

# =============================================================================
# TTSConfig Tests
# =============================================================================


class TestTTSConfig:
    """Tests for TTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        assert config.voice == "default"
        assert config.speed == 1.0
        assert config.language == "en"
        assert config.sample_rate == 22050
        assert config.format == "wav"

    def test_custom_values(self):
        """Test custom configuration values."""
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

    def test_extreme_speed_values(self):
        """Test edge case speed values."""
        slow = TTSConfig(speed=0.25)
        fast = TTSConfig(speed=4.0)
        assert slow.speed == 0.25
        assert fast.speed == 4.0


# =============================================================================
# AudioResult Tests
# =============================================================================


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_create_result(self):
        """Test creating an audio result."""
        audio_data = b"\x00\x00" * 1000
        result = AudioResult(
            audio=audio_data,
            sample_rate=22050,
            duration=2.5,
            format="wav",
        )
        assert result.audio == audio_data
        assert result.sample_rate == 22050
        assert result.duration == 2.5
        assert result.format == "wav"
        assert result.metadata == {}

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = AudioResult(
            audio=b"\x00",
            sample_rate=24000,
            duration=1.0,
            format="mp3",
            metadata={"model": "bark-small", "voice": "default"},
        )
        assert result.metadata["model"] == "bark-small"
        assert result.metadata["voice"] == "default"


# =============================================================================
# AudioEngine Interface Tests
# =============================================================================


class TestAudioEngineInterface:
    """Tests for the AudioEngine abstract interface."""

    def test_is_abstract_class(self):
        """Test that AudioEngine is abstract."""
        with pytest.raises(TypeError):
            AudioEngine()

    def test_required_methods(self):
        """Test that required methods are defined."""
        assert hasattr(AudioEngine, "load")
        assert hasattr(AudioEngine, "unload")
        assert hasattr(AudioEngine, "synthesize")
        assert hasattr(AudioEngine, "synthesize_stream")

    def test_required_properties(self):
        """Test that required properties are defined."""
        assert hasattr(AudioEngine, "model_name")
        assert hasattr(AudioEngine, "is_loaded")

    def test_optional_properties_have_defaults(self):
        """Test that optional properties have default implementations."""
        # These should be accessible on the class
        assert hasattr(AudioEngine, "supported_voices")
        assert hasattr(AudioEngine, "supported_languages")


# =============================================================================
# Concrete AudioEngine Implementation for Testing
# =============================================================================


class MockAudioEngine(AudioEngine):
    """Mock AudioEngine implementation for testing."""

    def __init__(self):
        self._loaded = False
        self._name = ""
        self._voices = ["default", "voice1", "voice2"]
        self._languages = ["en", "es", "fr"]

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True
        self._name = str(model_path)

    def unload(self) -> None:
        self._loaded = False
        self._name = ""

    def synthesize(self, text: str, config: TTSConfig | None = None) -> AudioResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        config = config or TTSConfig()

        # Generate fake audio (1 second of silence at given sample rate)
        num_samples = config.sample_rate
        audio_data = b"\x00\x00" * num_samples

        return AudioResult(
            audio=audio_data,
            sample_rate=config.sample_rate,
            duration=1.0,
            format=config.format,
            metadata={"text": text, "voice": config.voice},
        )

    def synthesize_stream(self, text: str, config: TTSConfig | None = None):
        result = self.synthesize(text, config)
        chunk_size = 1024
        audio = result.audio
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supported_voices(self) -> list[str]:
        return self._voices

    @property
    def supported_languages(self) -> list[str]:
        return self._languages


class TestMockAudioEngine:
    """Tests for the mock audio engine implementation."""

    def test_load_and_unload(self):
        """Test loading and unloading a model."""
        engine = MockAudioEngine()
        assert not engine.is_loaded

        engine.load("/path/to/model")
        assert engine.is_loaded
        assert engine.model_name == "/path/to/model"

        engine.unload()
        assert not engine.is_loaded

    def test_synthesize_requires_loaded_model(self):
        """Test that synthesize requires a loaded model."""
        engine = MockAudioEngine()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.synthesize("Hello")

    def test_synthesize_basic(self):
        """Test basic text synthesis."""
        engine = MockAudioEngine()
        engine.load("/path/to/model")

        result = engine.synthesize("Hello world")

        assert isinstance(result, AudioResult)
        assert result.format == "wav"
        assert result.sample_rate == 22050
        assert result.duration > 0
        assert len(result.audio) > 0

    def test_synthesize_with_config(self):
        """Test synthesis with custom config."""
        engine = MockAudioEngine()
        engine.load("/path/to/model")

        config = TTSConfig(
            voice="voice1",
            language="es",
            sample_rate=44100,
            format="mp3",
        )
        result = engine.synthesize("Hola mundo", config)

        assert result.sample_rate == 44100
        assert result.format == "mp3"
        assert result.metadata["voice"] == "voice1"

    def test_synthesize_stream(self):
        """Test streaming synthesis."""
        engine = MockAudioEngine()
        engine.load("/path/to/model")

        chunks = list(engine.synthesize_stream("Hello"))

        assert len(chunks) > 0
        total_size = sum(len(c) for c in chunks)
        assert total_size > 0

    def test_supported_voices(self):
        """Test getting supported voices."""
        engine = MockAudioEngine()
        engine.load("/path/to/model")

        voices = engine.supported_voices
        assert "default" in voices
        assert "voice1" in voices

    def test_supported_languages(self):
        """Test getting supported languages."""
        engine = MockAudioEngine()
        engine.load("/path/to/model")

        languages = engine.supported_languages
        assert "en" in languages
        assert "es" in languages


# =============================================================================
# ModelType Detection Tests
# =============================================================================


class TestModelTypeDetection:
    """Tests for model type detection."""

    def test_model_type_enum_values(self):
        """Test ModelType enum values."""
        assert ModelType.LLM.value == "llm"
        assert ModelType.TTS.value == "tts"
        assert ModelType.UNKNOWN.value == "unknown"

    def test_detect_tts_model_from_config(self, tmp_path):
        """Test detecting TTS model from config.json."""
        import json

        # Create a mock TTS model directory
        model_dir = tmp_path / "bark-model"
        model_dir.mkdir()

        config = {
            "architectures": ["BarkModel"],
            "model_type": "bark",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.TTS

    def test_detect_llm_model_from_config(self, tmp_path):
        """Test detecting LLM model from config.json."""
        import json

        model_dir = tmp_path / "llm-model"
        model_dir.mkdir()

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.LLM

    def test_detect_gguf_as_llm(self, tmp_path):
        """Test that GGUF files are detected as LLM."""
        model_dir = tmp_path / "gguf-model"
        model_dir.mkdir()
        (model_dir / "model.gguf").write_bytes(b"GGUF")

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.LLM

    def test_detect_unknown_without_config(self, tmp_path):
        """Test unknown detection without config."""
        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.UNKNOWN

    def test_detect_speecht5_as_tts(self, tmp_path):
        """Test detecting SpeechT5 as TTS."""
        import json

        model_dir = tmp_path / "speecht5-model"
        model_dir.mkdir()

        config = {
            "architectures": ["SpeechT5ForTextToSpeech"],
            "model_type": "speecht5",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.TTS


# =============================================================================
# TTS Engine Selection Tests
# =============================================================================


class TestTTSEngineSelection:
    """Tests for TTS engine selection logic."""

    def test_is_bark_model_by_name(self, tmp_path):
        """Test Bark model detection by name."""
        from hfl.engine.selector import _is_bark_model

        bark_dir = tmp_path / "suno-bark-small"
        bark_dir.mkdir()

        assert _is_bark_model(bark_dir)

    def test_is_bark_model_by_config(self, tmp_path):
        """Test Bark model detection by config."""
        import json

        from hfl.engine.selector import _is_bark_model

        model_dir = tmp_path / "my-model"
        model_dir.mkdir()

        config = {"architectures": ["BarkModel"]}
        (model_dir / "config.json").write_text(json.dumps(config))

        assert _is_bark_model(model_dir)

    def test_is_coqui_model(self, tmp_path):
        """Test Coqui model detection."""
        from hfl.engine.selector import _is_coqui_model

        # Test by path pattern
        assert _is_coqui_model(tmp_path / "tts_models/en/ljspeech/vits")
        assert _is_coqui_model(tmp_path / "xtts-v2")
        assert not _is_coqui_model(tmp_path / "regular-model")


# =============================================================================
# WAV Encoding Tests
# =============================================================================


class TestWAVEncoding:
    """Tests for WAV audio encoding."""

    def test_wav_header_structure(self):
        """Test that WAV encoding produces valid headers."""
        import struct

        # Create simple audio data
        audio = np.zeros(44100, dtype=np.float32)  # 1 second of silence

        # Use manual encoding from BarkEngine
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        sample_rate = 44100
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_bytes)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        wav_data = header + audio_bytes

        # Verify header
        assert wav_data[:4] == b"RIFF"
        assert wav_data[8:12] == b"WAVE"
        assert wav_data[12:16] == b"fmt "

    def test_audio_normalization(self):
        """Test that audio is properly normalized."""
        # Test clipping
        audio = np.array([2.0, -2.0, 0.5, -0.5])
        clipped = np.clip(audio, -1.0, 1.0)

        assert clipped[0] == 1.0
        assert clipped[1] == -1.0
        assert clipped[2] == 0.5
        assert clipped[3] == -0.5


# =============================================================================
# API Route Tests (basic structure)
# =============================================================================


# =============================================================================
# Extended ModelType Tests
# =============================================================================


class TestModelTypeFunctions:
    """Tests for model type utility functions."""

    def test_is_model_type_supported_llm(self):
        """Test that LLM is supported."""
        from hfl.converter.formats import ModelType, is_model_type_supported

        assert is_model_type_supported(ModelType.LLM) is True

    def test_is_model_type_supported_tts(self):
        """Test that TTS is supported."""
        from hfl.converter.formats import ModelType, is_model_type_supported

        # TTS is now supported
        assert is_model_type_supported(ModelType.TTS) is True

    def test_is_model_type_supported_stt(self):
        """Test that STT is not supported."""
        from hfl.converter.formats import ModelType, is_model_type_supported

        assert is_model_type_supported(ModelType.STT) is False

    def test_is_model_type_supported_image_gen(self):
        """Test that image generation is not supported."""
        from hfl.converter.formats import ModelType, is_model_type_supported

        assert is_model_type_supported(ModelType.IMAGE_GEN) is False

    def test_is_model_type_supported_unknown(self):
        """Test that unknown is not supported."""
        from hfl.converter.formats import ModelType, is_model_type_supported

        assert is_model_type_supported(ModelType.UNKNOWN) is False

    def test_get_model_type_display_name(self):
        """Test getting display names for model types."""
        from hfl.converter.formats import ModelType, get_model_type_display_name

        assert get_model_type_display_name(ModelType.LLM) == "LLM (Text Generation)"
        assert get_model_type_display_name(ModelType.TTS) == "TTS (Text-to-Speech)"
        assert get_model_type_display_name(ModelType.STT) == "STT (Speech-to-Text)"
        assert get_model_type_display_name(ModelType.IMAGE_GEN) == "Image Generation"


class TestExtendedModelTypeDetection:
    """Tests for extended model type detection."""

    def test_detect_whisper_as_stt(self, tmp_path):
        """Test detecting Whisper as STT."""
        import json

        model_dir = tmp_path / "whisper-model"
        model_dir.mkdir()

        config = {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.STT

    def test_detect_vit_as_image_class(self, tmp_path):
        """Test detecting ViT as image classification."""
        import json

        model_dir = tmp_path / "vit-model"
        model_dir.mkdir()

        config = {
            "architectures": ["ViTForImageClassification"],
            "model_type": "vit",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.IMAGE_CLASS

    def test_detect_detr_as_object_detection(self, tmp_path):
        """Test detecting DETR as object detection."""
        import json

        model_dir = tmp_path / "detr-model"
        model_dir.mkdir()

        config = {
            "architectures": ["DetrForObjectDetection"],
            "model_type": "detr",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.OBJECT_DETECT

    def test_detect_bert_as_embedding(self, tmp_path):
        """Test detecting BERT as embedding model."""
        import json

        model_dir = tmp_path / "bert-model"
        model_dir.mkdir()

        config = {
            "architectures": ["BertModel"],
            "model_type": "bert",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.EMBEDDING

    def test_detect_llava_as_visual_qa(self, tmp_path):
        """Test detecting LLaVA as visual QA."""
        import json

        model_dir = tmp_path / "llava-model"
        model_dir.mkdir()

        config = {
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.VISUAL_QA

    def test_detect_by_pipeline_tag(self, tmp_path):
        """Test detection by pipeline_tag field."""
        import json

        model_dir = tmp_path / "asr-model"
        model_dir.mkdir()

        config = {
            "pipeline_tag": "automatic-speech-recognition",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.STT

    def test_detect_sentence_transformers(self, tmp_path):
        """Test detection of sentence-transformers models."""
        import json

        model_dir = tmp_path / "st-model"
        model_dir.mkdir()

        # Create sentence_bert_config.json
        (model_dir / "sentence_bert_config.json").write_text("{}")
        (model_dir / "config.json").write_text(json.dumps({}))

        model_type = detect_model_type(model_dir)
        assert model_type == ModelType.EMBEDDING


# =============================================================================
# TTS API Schema Tests
# =============================================================================


class TestTTSAPISchemas:
    """Tests for TTS API request/response schemas."""

    def test_openai_tts_request_defaults(self):
        """Test OpenAI TTS request default values."""
        from hfl.api.routes_tts import OpenAITTSRequest

        req = OpenAITTSRequest(model="bark", input="Hello")
        assert req.voice == "alloy"
        assert req.response_format == "mp3"
        assert req.speed == 1.0

    def test_native_tts_request_defaults(self):
        """Test native TTS request default values."""
        from hfl.api.routes_tts import NativeTTSRequest

        req = NativeTTSRequest(model="bark", text="Hello")
        assert req.voice == "default"
        assert req.language == "en"
        assert req.format == "wav"
        assert req.speed == 1.0
        assert req.sample_rate == 22050
        assert req.stream is False

    def test_format_mapping(self):
        """Test format name mapping."""
        from hfl.api.routes_tts import _map_openai_format

        assert _map_openai_format("mp3") == "mp3"
        assert _map_openai_format("wav") == "wav"
        assert _map_openai_format("opus") == "ogg"
        assert _map_openai_format("unknown") == "wav"

    def test_content_type_mapping(self):
        """Test content type mapping."""
        from hfl.api.routes_tts import _format_to_content_type

        assert _format_to_content_type("wav") == "audio/wav"
        assert _format_to_content_type("mp3") == "audio/mpeg"
        assert _format_to_content_type("ogg") == "audio/ogg"
        assert _format_to_content_type("unknown") == "audio/wav"
