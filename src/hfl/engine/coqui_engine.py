# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Coqui TTS engine.

Coqui TTS is an open-source text-to-speech library with support for
multiple models including XTTS-v2, VITS, and more.

Supported models:
  - tts_models/multilingual/multi-dataset/xtts_v2 (~4GB VRAM)
  - tts_models/en/ljspeech/vits (~200MB)
  - tts_models/en/ljspeech/tacotron2-DDC (~400MB)
"""

import io
from typing import Iterator

import numpy as np

from hfl.engine.base import AudioEngine, AudioResult, TTSConfig


class CoquiEngine(AudioEngine):
    """TTS engine using Coqui TTS library."""

    def __init__(self):
        self._tts = None
        self._model_name: str = ""
        self._sample_rate: int = 22050

    def load(self, model_path: str, **kwargs) -> None:
        """Load Coqui TTS model.

        Args:
            model_path: Model name (e.g., "tts_models/multilingual/multi-dataset/xtts_v2")
                       or path to local model directory
            **kwargs:
                gpu: Whether to use GPU (default: auto-detect)
                progress_bar: Show download progress (default: True)
        """
        try:
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError(
                "Coqui engine requires coqui-tts.\n\n"
                "Install with:\n"
                "  pip install hfl[coqui]\n\n"
                "Or directly:\n"
                "  pip install coqui-tts"
            ) from e

        gpu = kwargs.get("gpu")
        if gpu is None:
            gpu = self._has_cuda()

        progress_bar = kwargs.get("progress_bar", True)

        # Load model
        self._tts = TTS(model_path, progress_bar=progress_bar, gpu=gpu)
        self._model_name = model_path

        # Get sample rate from model config
        if hasattr(self._tts, "synthesizer") and self._tts.synthesizer is not None:
            if hasattr(self._tts.synthesizer, "output_sample_rate"):
                self._sample_rate = self._tts.synthesizer.output_sample_rate
            elif hasattr(self._tts.synthesizer, "tts_config"):
                self._sample_rate = getattr(
                    self._tts.synthesizer.tts_config, "audio", {}
                ).get("sample_rate", 22050)

    def unload(self) -> None:
        """Release model from memory."""
        if self._tts is not None:
            del self._tts
            self._tts = None
            self._model_name = ""

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def synthesize(self, text: str, config: TTSConfig | None = None) -> AudioResult:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize
            config: TTS configuration

        Returns:
            AudioResult with WAV audio bytes
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = config or TTSConfig()

        # Prepare synthesis parameters
        kwargs = {}

        # Language (for multilingual models like XTTS)
        if self._is_multilingual():
            kwargs["language"] = config.language

        # Voice/speaker
        if config.voice != "default" and self._supports_speakers():
            kwargs["speaker"] = config.voice

        # Speed (Coqui uses speed parameter for some models)
        if config.speed != 1.0:
            kwargs["speed"] = config.speed

        # Generate audio
        wav = self._tts.tts(text=text, **kwargs)

        # Convert to numpy array if needed
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        # Get actual sample rate
        sample_rate = self._sample_rate

        # Resample if needed
        if config.sample_rate != sample_rate:
            wav = self._resample(wav, sample_rate, config.sample_rate)
            sample_rate = config.sample_rate

        # Calculate duration
        duration = len(wav) / sample_rate

        # Encode to requested format
        audio_bytes = self._encode_audio(wav, sample_rate, config.format)

        return AudioResult(
            audio=audio_bytes,
            sample_rate=sample_rate,
            duration=duration,
            format=config.format,
            metadata={
                "model": self._model_name,
                "language": config.language,
                "voice": config.voice,
            },
        )

    def synthesize_stream(
        self, text: str, config: TTSConfig | None = None
    ) -> Iterator[bytes]:
        """Stream audio synthesis.

        Note: Most Coqui models don't support native streaming,
        so this synthesizes full audio and yields chunks.

        For XTTS, we could potentially use the streaming API in the future.

        Args:
            text: Text to synthesize
            config: TTS configuration

        Yields:
            Audio data chunks
        """
        # Check if XTTS streaming is available
        if self._supports_streaming():
            yield from self._stream_xtts(text, config)
        else:
            # Fallback: synthesize and chunk
            result = self.synthesize(text, config)
            chunk_size = 1024
            audio = result.audio
            for i in range(0, len(audio), chunk_size):
                yield audio[i : i + chunk_size]

    def _stream_xtts(self, text: str, config: TTSConfig | None = None) -> Iterator[bytes]:
        """Stream synthesis using XTTS streaming API."""
        config = config or TTSConfig()

        # XTTS streaming requires different handling
        try:
            chunks = self._tts.tts_to_file(
                text=text,
                language=config.language,
                speaker=config.voice if config.voice != "default" else None,
                split_sentences=True,
            )
            # This is a simplified version - actual XTTS streaming
            # would use the lower-level streaming API
            yield from chunks
        except Exception:
            # Fallback to non-streaming
            result = self.synthesize(text, config)
            chunk_size = 1024
            audio = result.audio
            for i in range(0, len(audio), chunk_size):
                yield audio[i : i + chunk_size]

    @property
    def is_loaded(self) -> bool:
        return self._tts is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_voices(self) -> list[str]:
        """Get list of available speakers/voices."""
        if not self.is_loaded:
            return ["default"]

        if hasattr(self._tts, "speakers") and self._tts.speakers:
            return list(self._tts.speakers)

        return ["default"]

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        if not self.is_loaded:
            return ["en"]

        if hasattr(self._tts, "languages") and self._tts.languages:
            return list(self._tts.languages)

        return ["en"]

    def _is_multilingual(self) -> bool:
        """Check if the loaded model supports multiple languages."""
        if not self.is_loaded:
            return False
        return hasattr(self._tts, "is_multi_lingual") and self._tts.is_multi_lingual

    def _supports_speakers(self) -> bool:
        """Check if the loaded model supports multiple speakers."""
        if not self.is_loaded:
            return False
        return hasattr(self._tts, "speakers") and bool(self._tts.speakers)

    def _supports_streaming(self) -> bool:
        """Check if the model supports streaming synthesis."""
        # Currently only XTTS supports streaming
        return "xtts" in self._model_name.lower()

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import torch
            import torchaudio

            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            resampled = resampler(audio_tensor)
            return resampled.squeeze().numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def _encode_audio(
        self, audio: np.ndarray, sample_rate: int, fmt: str
    ) -> bytes:
        """Encode numpy audio array to bytes in the specified format."""
        # Ensure audio is float32 and in range [-1, 1]
        audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

        if fmt == "wav":
            return self._encode_wav(audio, sample_rate)
        elif fmt == "mp3":
            return self._encode_mp3(audio, sample_rate)
        elif fmt == "ogg":
            return self._encode_ogg(audio, sample_rate)
        else:
            return self._encode_wav(audio, sample_rate)

    def _encode_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio to WAV format."""
        try:
            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            return self._encode_wav_manual(audio, sample_rate)

    def _encode_wav_manual(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Manual WAV encoding without soundfile."""
        import struct

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # WAV header
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
            1,  # PCM
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        return header + audio_bytes

    def _encode_mp3(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio to MP3 format."""
        try:
            from pydub import AudioSegment

            audio_int16 = (audio * 32767).astype(np.int16)

            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )

            buffer = io.BytesIO()
            segment.export(buffer, format="mp3")
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            raise ImportError(
                "MP3 encoding requires pydub.\n"
                "Install with: pip install pydub\n"
                "Also requires ffmpeg to be installed."
            )

    def _encode_ogg(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Encode audio to OGG format."""
        try:
            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="OGG", subtype="VORBIS")
            buffer.seek(0)
            return buffer.read()
        except ImportError:
            raise ImportError(
                "OGG encoding requires soundfile.\n"
                "Install with: pip install soundfile"
            )


# Convenience function to list available Coqui models
def list_coqui_models() -> dict:
    """List all available Coqui TTS models.

    Returns:
        Dictionary with model categories and names
    """
    try:
        from TTS.utils.manage import ModelManager

        manager = ModelManager()
        return {
            "tts_models": manager.list_tts_models(),
            "vocoder_models": manager.list_vocoder_models(),
        }
    except ImportError:
        return {"error": "coqui-tts not installed"}
