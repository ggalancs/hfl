# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""TTS API schemas.

These schemas define the request/response formats for Text-to-Speech endpoints.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    PCM = "pcm"


class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request schema.

    Compatible with OpenAI's /v1/audio/speech endpoint.

    Attributes:
        model: TTS model to use for synthesis
        input: Text to synthesize (max 4096 characters)
        voice: Voice to use for synthesis
        response_format: Audio output format
        speed: Speed multiplier (0.25 to 4.0)
    """

    model: str = Field(
        ...,
        description="TTS model to use",
        examples=["bark-small", "coqui-tts"],
    )
    input: str = Field(
        ...,
        description="Text to synthesize",
        max_length=4096,
    )
    voice: str = Field(
        default="alloy",
        max_length=128,
        description="Voice to use for synthesis",
        examples=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio output format",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed multiplier (0.25 to 4.0)",
    )


class NativeTTSRequest(BaseModel):
    """Native HFL/Ollama-style TTS request schema.

    Used with HFL's native /api/tts endpoint.

    Attributes:
        model: TTS model to use for synthesis
        text: Text to synthesize (max 4096 characters)
        voice: Voice/speaker to use
        language: Language code (e.g., 'en', 'es', 'fr')
        speed: Speed multiplier (0.25 to 4.0)
        sample_rate: Output sample rate in Hz
        format: Audio output format
        stream: Whether to stream audio chunks
    """

    model: str = Field(
        ...,
        description="TTS model to use",
        examples=["bark-small", "coqui-tts"],
    )
    text: str = Field(
        ...,
        description="Text to synthesize",
        max_length=4096,
    )
    voice: str = Field(
        default="default",
        max_length=128,
        description="Voice/speaker to use",
    )
    language: str = Field(
        default="en",
        max_length=32,
        description="Language code",
        examples=["en", "es", "fr", "de", "it"],
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed multiplier (0.25 to 4.0)",
    )
    sample_rate: int = Field(
        default=22050,
        description="Output sample rate in Hz",
        examples=[16000, 22050, 44100, 48000],
    )
    format: Literal["wav", "mp3", "ogg"] = Field(
        default="wav",
        description="Audio output format",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream audio chunks",
    )


class TTSModelInfo(BaseModel):
    """TTS model information for listing endpoints.

    Attributes:
        id: Model identifier
        object: Object type (always "model")
        owned_by: Model owner/organization
        capabilities: Model capabilities dict
    """

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    owned_by: str = Field(..., description="Model owner/organization")
    capabilities: dict = Field(
        default_factory=dict,
        description="Model capabilities",
    )
