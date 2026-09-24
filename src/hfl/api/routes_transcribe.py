# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Speech-to-text: ``POST /api/transcribe`` and OpenAI's
``POST /v1/audio/transcriptions`` (Whisper-backed).

``model`` is a Whisper size or a Hub repo id, which the backend would
download on first use. Only a caller who may fetch models (the owner:
:func:`~hfl.api.admin_guard.may_fetch_models`) can trigger that download;
anyone else is served models already on disk, or told the model is not on
this server. The load and the inference run off the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from hfl.engine.whisper_engine import WhisperEngine, WhisperResult, is_available
from hfl.hub.local_cache import whisper_available_locally

logger = logging.getLogger(__name__)
router = APIRouter(tags=["STT"])


_MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB
_DEFAULT_MODEL = "small"
# OpenAI's model names; they mean "the default Whisper" here.
_OPENAI_MODELS = frozenset({"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"})
_FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})


async def _read_audio(file: UploadFile) -> bytes:
    # SEC: read a bounded amount. This route is exempt from the global
    # request-body limit (audio files are legitimately larger than a JSON
    # body), and the size check used to run *after* an unbounded
    # ``file.read()`` — so a caller could make the server buffer an
    # arbitrarily large upload, spilling it to the temp directory via
    # Starlette's SpooledTemporaryFile and then pulling the whole thing into
    # a bytes object, before being told 413. Reading one byte past the limit
    # is enough to detect an oversized upload without ever holding more than
    # the limit in memory.
    audio = await file.read(_MAX_AUDIO_BYTES + 1)
    if len(audio) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio exceeds {_MAX_AUDIO_BYTES} bytes",
        )
    return audio


def _transcribe_sync(
    audio: bytes,
    model: str,
    language: str | None,
    include_segments: bool,
    local_only: bool,
) -> WhisperResult:
    engine = WhisperEngine()
    try:
        engine.load(model, local_files_only=local_only)
    except Exception:
        logger.exception("Whisper load failed: %s", model)
        raise HTTPException(status_code=500, detail="whisper load failed") from None
    try:
        return engine.transcribe(audio, language=language, include_segments=include_segments)
    except Exception:
        logger.exception("Whisper transcribe failed")
        raise HTTPException(status_code=500, detail="transcription failed") from None
    finally:
        engine.unload()


async def _transcribe(
    request: Request,
    file: UploadFile,
    model: str,
    language: str | None,
    include_segments: bool,
) -> WhisperResult:
    if not is_available():
        raise HTTPException(
            status_code=501,
            detail="Whisper backend not installed. `pip install 'hfl[stt]'`.",
        )
    from hfl.api.admin_guard import may_fetch_models

    local_only = not may_fetch_models(request)
    if local_only and not await asyncio.to_thread(whisper_available_locally, model):
        raise HTTPException(
            status_code=404,
            detail={
                "error": (
                    f"{model} is not on this server, and only the server's owner "
                    "can download models."
                ),
                "code": "model_not_local",
            },
        )
    audio = await _read_audio(file)
    return await asyncio.to_thread(
        _transcribe_sync, audio, model, language, include_segments, local_only
    )


@router.post("/api/transcribe", response_model=None)
async def api_transcribe(
    request: Request,
    model: str = Form(
        _DEFAULT_MODEL,
        description="Whisper size (tiny/base/small/medium/large-v3) or HF repo id.",
    ),
    language: str | None = Form(None),
    include_segments: bool = Form(False),
    file: UploadFile = File(...),
) -> dict[str, Any] | JSONResponse:
    """Transcribe an uploaded audio file.

    Body is ``multipart/form-data`` with a ``file`` field carrying
    wav / mp3 / ogg / flac / m4a. Max 100 MB. Additional form fields
    mirror the Whisper SDK names: ``model`` (size or repo id),
    ``language`` (ISO code, optional), ``include_segments`` (attach
    per-segment timestamps).
    """
    result = await _transcribe(request, file, model, language, include_segments)
    envelope: dict[str, Any] = {
        "text": result.text,
        "language": result.language,
        "duration_s": result.duration_s,
        "model": model,
    }
    if result.segments is not None:
        envelope["segments"] = [
            {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
        ]
    return envelope


def _timestamp(seconds: float, separator: str) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _subtitles(result: WhisperResult, *, vtt: bool) -> str:
    separator = "." if vtt else ","
    blocks = []
    for index, segment in enumerate(result.segments or [], start=1):
        timing = f"{_timestamp(segment.start, separator)} --> {_timestamp(segment.end, separator)}"
        cue = f"{timing}\n{segment.text.strip()}\n"
        blocks.append(cue if vtt else f"{index}\n{cue}")
    body = "\n".join(blocks)
    return f"WEBVTT\n\n{body}" if vtt else body


@router.post("/v1/audio/transcriptions", response_model=None, tags=["OpenAI"])
async def openai_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
) -> dict[str, Any] | PlainTextResponse:
    """OpenAI-compatible transcription. ``whisper-1`` (and OpenAI's other
    names) mean the default local Whisper; any size or repo id works too.
    ``prompt`` and ``temperature`` are accepted and not used."""
    if response_format not in _FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of {', '.join(sorted(_FORMATS))}",
        )
    whisper_model = _DEFAULT_MODEL if model in _OPENAI_MODELS else model
    with_segments = response_format in ("verbose_json", "srt", "vtt")
    result = await _transcribe(request, file, whisper_model, language, with_segments)
    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format in ("srt", "vtt"):
        return PlainTextResponse(_subtitles(result, vtt=response_format == "vtt"))
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": result.duration_s,
            "text": result.text,
            "segments": [
                {"id": i, "start": s.start, "end": s.end, "text": s.text}
                for i, s in enumerate(result.segments or [])
            ],
        }
    return {"text": result.text}


__all__ = ["router"]
