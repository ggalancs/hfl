# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""
TTS API endpoints.

Implements:
  - OpenAI-compatible: POST /v1/audio/speech
  - Native HFL: POST /api/tts
"""

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from hfl.api.helpers import run_with_timeout
from hfl.api.schemas import NativeTTSRequest, OpenAITTSRequest, TTSModelInfo
from hfl.core.container import get_registry
from hfl.engine.base import TTSConfig

if TYPE_CHECKING:
    from hfl.api.state import ServerState

router = APIRouter(tags=["TTS"])


# =============================================================================
# Helpers
# =============================================================================


def _get_state() -> "ServerState":
    """Get the singleton server state."""
    from hfl.api.state import get_state

    return get_state()


async def _ensure_tts_model_loaded(model_name: str) -> None:
    """Load the TTS model if not already in memory (thread-safe)."""
    state = _get_state()

    # Fast path: model already loaded
    if state.is_tts_loaded():
        if state.current_tts_model and state.current_tts_model.name == model_name:
            return

    # Look up model in registry
    registry = get_registry()
    manifest = registry.get(model_name)
    if not manifest:
        raise HTTPException(404, f"TTS model not found: {model_name}")

    # Verify it's a TTS model
    from pathlib import Path

    from hfl.converter.formats import ModelType, detect_model_type

    model_path = Path(manifest.local_path)
    model_type = detect_model_type(model_path)

    if model_type != ModelType.TTS:
        raise HTTPException(
            400,
            f"Model '{model_name}' is not a TTS model (detected: {model_type.value})",
        )

    # Select and load engine
    from hfl.engine.selector import select_tts_engine

    engine = select_tts_engine(model_path)
    engine.load(manifest.local_path)

    # Thread-safe state update
    await state.set_tts_engine(engine, manifest)


def _map_openai_format(fmt: str) -> str:
    """Map OpenAI format names to internal format names."""
    mapping = {
        "mp3": "mp3",
        "opus": "ogg",  # We use OGG Vorbis, close enough
        "aac": "mp3",  # Fallback to MP3
        "flac": "wav",  # Fallback to WAV
        "wav": "wav",
        "pcm": "wav",  # PCM in WAV container
    }
    return mapping.get(fmt, "wav")


def _format_to_content_type(fmt: str) -> str:
    """Get MIME type for audio format."""
    types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
    }
    return types.get(fmt, "audio/wav")


# =============================================================================
# OpenAI-Compatible Endpoint
# =============================================================================


@router.post("/v1/audio/speech")
async def openai_tts(req: OpenAITTSRequest) -> Response:
    """
    OpenAI-compatible TTS endpoint.

    POST /v1/audio/speech
    {
        "model": "bark-small",
        "input": "Hello, world!",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0
    }

    Returns: Audio binary data
    """
    await _ensure_tts_model_loaded(req.model)
    state = _get_state()
    assert state.tts_engine is not None  # Guaranteed by _ensure_tts_model_loaded

    # Map OpenAI format to internal format
    audio_format = _map_openai_format(req.response_format)

    # Create TTS config
    config = TTSConfig(
        voice=req.voice,
        speed=req.speed,
        format=audio_format,
    )

    # Synthesize with timeout
    result = await run_with_timeout(
        state.tts_engine.synthesize, req.input, config, operation="tts_synthesize"
    )

    return Response(
        content=result.audio,
        media_type=_format_to_content_type(result.format),
        headers={
            "Content-Disposition": f'attachment; filename="speech.{result.format}"',
            "X-Audio-Duration": str(result.duration),
            "X-Audio-Sample-Rate": str(result.sample_rate),
        },
    )


# =============================================================================
# Native HFL Endpoints
# =============================================================================


@router.post("/api/tts", response_model=None)
async def native_tts(req: NativeTTSRequest) -> Response | StreamingResponse:
    """
    Native HFL TTS endpoint.

    POST /api/tts
    {
        "model": "bark-small",
        "text": "Hello, world!",
        "language": "en",
        "voice": "default",
        "format": "wav",
        "stream": false
    }

    Returns:
        - If stream=false: Audio binary data
        - If stream=true: Streaming audio chunks
    """
    await _ensure_tts_model_loaded(req.model)
    state = _get_state()
    assert state.tts_engine is not None  # Guaranteed by _ensure_tts_model_loaded

    config = TTSConfig(
        voice=req.voice,
        speed=req.speed,
        language=req.language,
        sample_rate=req.sample_rate,
        format=req.format,
    )

    if req.stream:
        return StreamingResponse(
            state.tts_engine.synthesize_stream(req.text, config),
            media_type=_format_to_content_type(req.format),
            headers={
                "X-Audio-Sample-Rate": str(req.sample_rate),
            },
        )

    # Synthesize with timeout
    result = await run_with_timeout(
        state.tts_engine.synthesize, req.text, config, operation="tts_synthesize"
    )

    return Response(
        content=result.audio,
        media_type=_format_to_content_type(result.format),
        headers={
            "Content-Disposition": f'attachment; filename="speech.{result.format}"',
            "X-Audio-Duration": str(result.duration),
            "X-Audio-Sample-Rate": str(result.sample_rate),
        },
    )


@router.get("/api/tts/voices")
async def list_voices(model: str | None = None) -> dict[str, Any]:
    """
    List available voices for a TTS model.

    GET /api/tts/voices?model=bark-small

    Returns:
        {
            "voices": ["v2/en_speaker_0", "v2/en_speaker_1", ...],
            "languages": ["en", "es", "fr", ...]
        }
    """
    if model:
        await _ensure_tts_model_loaded(model)
        state = _get_state()
        assert state.tts_engine is not None  # Guaranteed by _ensure_tts_model_loaded

        return {
            "model": model,
            "voices": state.tts_engine.supported_voices,
            "languages": state.tts_engine.supported_languages,
        }

    # No model specified, return generic info
    return {
        "message": "Specify a model to get available voices",
        "example": "/api/tts/voices?model=bark-small",
    }


@router.get("/v1/audio/models")
async def list_tts_models() -> dict[str, Any]:
    """
    List available TTS models.

    GET /v1/audio/models

    Returns: List of TTS models in OpenAI format
    """
    from pathlib import Path

    from hfl.converter.formats import ModelType, detect_model_type

    registry = get_registry()
    all_models = registry.list_all()

    tts_models = []
    for m in all_models:
        model_type = detect_model_type(Path(m.local_path))
        if model_type == ModelType.TTS:
            tts_models.append(
                TTSModelInfo(
                    id=m.name,
                    owned_by=m.repo_id.split("/")[0] if "/" in m.repo_id else "local",
                    capabilities={
                        "tts": True,
                        "voices": True,
                    },
                )
            )

    return {
        "object": "list",
        "data": [m.model_dump() for m in tts_models],
    }
