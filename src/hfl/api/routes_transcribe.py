# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""``POST /api/transcribe`` — Whisper-backed speech-to-text (Phase 16)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from hfl.engine.whisper_engine import WhisperEngine, is_available

logger = logging.getLogger(__name__)
router = APIRouter(tags=["STT"])


_MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("/api/transcribe", response_model=None)
async def api_transcribe(
    model: str = Form(
        "small",
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
    if not is_available():
        raise HTTPException(
            status_code=501,
            detail="Whisper backend not installed. `pip install 'hfl[stt]'`.",
        )
    audio = await file.read()
    if len(audio) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio exceeds {_MAX_AUDIO_BYTES} bytes",
        )
    engine = WhisperEngine()
    try:
        engine.load(model)
    except Exception:
        logger.exception("Whisper load failed: %s", model)
        raise HTTPException(status_code=500, detail="whisper load failed")

    try:
        result = engine.transcribe(
            audio,
            language=language,
            include_segments=include_segments,
        )
    except Exception:
        logger.exception("Whisper transcribe failed")
        raise HTTPException(status_code=500, detail="transcription failed")
    finally:
        engine.unload()

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


__all__ = ["router"]
