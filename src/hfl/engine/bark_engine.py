# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
Bark TTS engine using HuggingFace Transformers.

Bark is a transformer-based text-to-audio model created by Suno.
It can generate highly realistic, multilingual speech as well as other audio.

Supported models:
  - suno/bark-small (~1GB VRAM)
  - suno/bark (~2GB VRAM)
"""

import io
from typing import Iterator

import numpy as np

from hfl.engine.base import AudioEngine, AudioResult, TTSConfig


class BarkEngine(AudioEngine):
    """TTS engine using Bark via HuggingFace Transformers."""

    def __init__(self):
        self._pipeline = None
        self._model_name: str = ""
        self._sample_rate: int = 24000  # Bark outputs at 24kHz

    def load(self, model_path: str, **kwargs) -> None:
        """Load Bark model.

        Args:
            model_path: Path to model directory or HuggingFace model ID
            **kwargs:
                device: Device to use ("cuda", "cpu", "mps", "auto")
                dtype: torch dtype (torch.float16, torch.float32)
        """
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "Bark engine requires transformers.\n\n"
                "Install with:\n"
                "  pip install hfl[tts]\n\n"
                "Or directly:\n"
                "  pip install transformers torch torchaudio"
            ) from e

        device = kwargs.get("device", "auto")
        if device == "auto":
            device = self._detect_device()

        # torch_dtype for memory efficiency
        torch_dtype = kwargs.get("dtype")
        if torch_dtype is None:
            import torch

            torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        self._pipeline = pipeline(
            "text-to-audio",
            model=model_path,
            device=device if device != "cpu" else -1,
            torch_dtype=torch_dtype,
        )
        self._model_name = model_path
        self._sample_rate = 24000  # Bark native sample rate

    def unload(self) -> None:
        """Release model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._model_name = ""

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # Ignore any torch import/runtime errors
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

        # Generate audio
        output = self._pipeline(text)
        audio_array = output["audio"]
        sampling_rate = output["sampling_rate"]

        # Handle multi-dimensional output (stereo or batched)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.squeeze()
        if len(audio_array.shape) > 1:
            # If still multi-dimensional, take first channel
            audio_array = audio_array[0]

        # Resample if needed
        if config.sample_rate != sampling_rate:
            audio_array = self._resample(audio_array, sampling_rate, config.sample_rate)
            sampling_rate = config.sample_rate

        # Apply speed adjustment if not 1.0
        if config.speed != 1.0:
            audio_array = self._adjust_speed(audio_array, config.speed)

        # Calculate duration
        duration = len(audio_array) / sampling_rate

        # Encode to requested format
        audio_bytes = self._encode_audio(audio_array, sampling_rate, config.format)

        return AudioResult(
            audio=audio_bytes,
            sample_rate=sampling_rate,
            duration=duration,
            format=config.format,
            metadata={"model": self._model_name},
        )

    def synthesize_stream(
        self, text: str, config: TTSConfig | None = None
    ) -> Iterator[bytes]:
        """Stream audio synthesis.

        Note: Bark doesn't natively support streaming, so this synthesizes
        the full audio and yields it in chunks.

        Args:
            text: Text to synthesize
            config: TTS configuration

        Yields:
            Audio data chunks
        """
        result = self.synthesize(text, config)

        # Yield in chunks of ~1024 bytes
        chunk_size = 1024
        audio = result.audio
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_voices(self) -> list[str]:
        """Bark supports voice presets."""
        return [
            "v2/en_speaker_0",
            "v2/en_speaker_1",
            "v2/en_speaker_2",
            "v2/en_speaker_3",
            "v2/en_speaker_4",
            "v2/en_speaker_5",
            "v2/en_speaker_6",
            "v2/en_speaker_7",
            "v2/en_speaker_8",
            "v2/en_speaker_9",
            "v2/es_speaker_0",
            "v2/es_speaker_1",
            "v2/fr_speaker_0",
            "v2/de_speaker_0",
            "v2/zh_speaker_0",
            "v2/ja_speaker_0",
        ]

    @property
    def supported_languages(self) -> list[str]:
        """Bark supports multiple languages."""
        return ["en", "es", "fr", "de", "zh", "ja", "ko", "pl", "pt", "ru", "tr", "it"]

    def _detect_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import torch
            import torchaudio

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            resampled = resampler(audio_tensor)
            return resampled.squeeze().numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            import numpy as np

            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio playback speed."""
        import numpy as np

        # Simple time stretching via interpolation
        new_length = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)

    def _encode_audio(
        self, audio: np.ndarray, sample_rate: int, fmt: str
    ) -> bytes:
        """Encode numpy audio array to bytes in the specified format."""
        import numpy as np

        # Ensure audio is in the correct range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        if fmt == "wav":
            return self._encode_wav(audio, sample_rate)
        elif fmt == "mp3":
            return self._encode_mp3(audio, sample_rate)
        elif fmt == "ogg":
            return self._encode_ogg(audio, sample_rate)
        else:
            # Default to WAV
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
            # Fallback: manual WAV encoding
            return self._encode_wav_manual(audio, sample_rate)

    def _encode_wav_manual(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Manual WAV encoding without soundfile."""
        import struct

        import numpy as np

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
            16,  # Subchunk1Size
            1,  # AudioFormat (PCM)
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
        """Encode audio to MP3 format (requires pydub or ffmpeg)."""
        try:
            import numpy as np
            from pydub import AudioSegment

            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)

            # Create AudioSegment
            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
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
