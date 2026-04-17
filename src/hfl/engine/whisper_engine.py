# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Whisper ASR engine (Phase 16 — V2 row 16).

Adapter around ``faster-whisper`` (preferred — CPU/GPU both fast)
or ``openai-whisper`` (fallback). The whole module is optional: a
``pip install 'hfl'`` that doesn't pull the ``[stt]`` extra still
imports cleanly and ``is_available()`` returns False.

Usage:

    engine = WhisperEngine()
    engine.load("small")      # model size or repo id
    result = engine.transcribe(audio_bytes, language="en")

The ``transcribe`` entry point is synchronous and thread-safe under
the assumption that the caller serialises through HFL's dispatcher
(the same constraint we apply to LlamaCppEngine).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["WhisperEngine", "WhisperResult", "is_available"]


def is_available() -> bool:
    """True iff either Whisper backend is importable."""
    for module in ("faster_whisper", "whisper"):
        try:
            __import__(module)
            return True
        except ImportError:
            continue
    return False


@dataclass
class WhisperSegment:
    """One transcribed segment with timing metadata."""

    start: float
    end: float
    text: str


@dataclass
class WhisperResult:
    """Return value of ``WhisperEngine.transcribe``."""

    text: str
    language: str | None = None
    duration_s: float = 0.0
    segments: list[WhisperSegment] | None = None


class WhisperEngine:
    """Backend-agnostic wrapper around faster-whisper / openai-whisper."""

    def __init__(self) -> None:
        self._backend: str | None = None
        self._model: Any = None
        self._model_name: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(
        self,
        model: str,
        *,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        """Load a Whisper model by size or HF repo id.

        ``model`` is one of ``tiny``/``base``/``small``/``medium``/
        ``large-v3``, or a HuggingFace repo id.
        """
        if not is_available():
            raise RuntimeError("No Whisper backend installed. `pip install 'hfl[stt]'` adds one.")
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(
                model,
                device=device or "auto",
                compute_type=compute_type or "auto",
            )
            self._backend = "faster_whisper"
            self._model_name = model
            logger.info("Whisper loaded (faster_whisper): %s", model)
            return
        except ImportError:
            pass
        try:
            import whisper  # type: ignore

            self._model = whisper.load_model(model)
            self._backend = "openai_whisper"
            self._model_name = model
            logger.info("Whisper loaded (openai): %s", model)
        except ImportError as exc:
            raise RuntimeError("Whisper install failed: no backend available") from exc

    def unload(self) -> None:
        self._model = None
        self._backend = None
        self._model_name = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str | None:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._model_name or "whisper"

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        include_segments: bool = False,
    ) -> WhisperResult:
        if not self.is_loaded:
            raise RuntimeError("Whisper engine not loaded")
        start = time.perf_counter()
        if self._backend == "faster_whisper":
            text, lang, segments = self._run_faster_whisper(audio, language)
        else:
            text, lang, segments = self._run_openai_whisper(audio, language)
        return WhisperResult(
            text=text,
            language=lang,
            duration_s=time.perf_counter() - start,
            segments=segments if include_segments else None,
        )

    def _run_faster_whisper(
        self,
        audio: bytes,
        language: str | None,
    ) -> tuple[str, str | None, list[WhisperSegment]]:

        # faster-whisper accepts file paths and bytesio; pass an
        # in-memory path-like to avoid hitting disk.
        segments_iter, info = self._model.transcribe(
            io.BytesIO(audio),
            language=language,
            beam_size=1,
        )
        out_segments: list[WhisperSegment] = []
        chunks: list[str] = []
        for seg in segments_iter:
            out_segments.append(WhisperSegment(start=seg.start, end=seg.end, text=seg.text))
            chunks.append(seg.text)
        return "".join(chunks).strip(), info.language, out_segments

    def _run_openai_whisper(
        self,
        audio: bytes,
        language: str | None,
    ) -> tuple[str, str | None, list[WhisperSegment]]:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio)
            tmp.flush()
            result = self._model.transcribe(tmp.name, language=language)
        text = (result or {}).get("text", "").strip()
        lang = (result or {}).get("language")
        out_segments = [
            WhisperSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
            )
            for seg in (result or {}).get("segments", []) or []
        ]
        return text, lang, out_segments
