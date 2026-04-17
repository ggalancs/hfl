# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Embedding engines (Ollama / OpenAI parity).

Embedding models produce dense vectors instead of tokens; they are
the foundation of RAG pipelines (LangChain, LlamaIndex), semantic
search, clustering and retrieval. Supporting them is the P0-1 item
of ``OLLAMA_PARITY_PLAN.md``.

Architecture:

- :class:`EmbeddingEngine` (abstract) — one method, ``embed``, that
  takes ``list[str]`` and returns ``list[list[float]]``. Matches
  Ollama's ``/api/embed`` contract.
- :class:`LlamaCppEmbeddingEngine` — llama-cpp-python with
  ``embedding=True``. The same library that serves LLMs, so no new
  runtime dep is pulled in for users of the ``[llama]`` extra.
- :class:`TransformersEmbeddingEngine` — sentence-transformers-style
  mean-pooling over transformer hidden states. Requires the
  ``[transformers]`` extra.

Both adapters share the same ``truncate`` and ``dimensions``
semantics so routing can be done at the router layer without
special-casing.

None of the code in this module does blocking network I/O. The
routes layer is responsible for wrapping ``engine.embed(...)`` in
``asyncio.to_thread`` — mirroring the LLM backend contract.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover — typing-only
    pass


@dataclass
class EmbeddingResult:
    """Output of a single embed call.

    Attributes:
        embeddings: One vector per input string, same order as input.
        total_tokens: Total tokens consumed across all inputs
            (``prompt_eval_count`` in Ollama / ``usage.prompt_tokens``
            in OpenAI). ``0`` when the backend can't report it.
        model: Name of the model that produced the vectors (used to
            populate the response envelope).
    """

    embeddings: list[list[float]]
    total_tokens: int = 0
    model: str = ""


class EmbeddingEngine(ABC):
    """Base class for embedding backends.

    Concrete implementations (LlamaCpp, Transformers, vLLM when vLLM
    gains embedding support) override ``load`` / ``unload`` /
    ``embed``. The interface is intentionally minimal because
    clients of embedding servers care about exactly one thing:
    "give me vectors back for these strings, fast".
    """

    def __init__(self) -> None:
        self._loaded = False
        self._model_path = ""

    @property
    def is_loaded(self) -> bool:
        """True when the engine holds a live model and can serve embeds."""
        return self._loaded

    @property
    def model_name(self) -> str:
        """Path / identifier of the loaded model."""
        return self._model_path

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load the model from ``model_path``.

        Kwargs are backend-specific (``n_ctx``, ``n_gpu_layers`` for
        llama-cpp; ``device``, ``trust_remote_code`` for
        Transformers, etc.).
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model and any GPU / Metal / CUDA resources."""

    @abstractmethod
    def embed(
        self,
        inputs: list[str],
        *,
        truncate: bool = True,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """Produce embeddings for a batch of input strings.

        Args:
            inputs: Strings to embed. Empty list is rejected by the
                router; engines may assume non-empty.
            truncate: When True (Ollama's default), inputs longer
                than the model's context are truncated to fit
                instead of raising. When False, an oversized input
                raises ``ValueError``.
            dimensions: Optional Matryoshka-style truncation of the
                output vectors. When None, the model's native
                dimension is returned. Values > native dimension
                raise ``ValueError``.

        Returns:
            :class:`EmbeddingResult` with one vector per input in the
            same order.

        Raises:
            RuntimeError: Model not loaded.
            ValueError: Inputs list empty or invalid ``dimensions``.
        """


# ----------------------------------------------------------------------
# LLama.cpp adapter
# ----------------------------------------------------------------------


class LlamaCppEmbeddingEngine(EmbeddingEngine):
    """Embedding engine backed by llama-cpp-python.

    Supports the full GGUF embedding catalogue: nomic-embed-text-v1.5,
    bge-*, e5-*, mxbai-embed-*, jina-embeddings-v2-*, etc. The same
    library that powers :class:`hfl.engine.llama_cpp.LlamaCppEngine`
    gains ``embedding=True`` at construction time to switch from
    causal generation to pooled-embedding mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self._llm: Any | None = None
        self._n_embd: int | None = None  # Native embedding dimension

    def load(self, model_path: str, **kwargs: Any) -> None:
        from llama_cpp import Llama  # Deferred — optional dep

        # n_ctx default: 8192 is a safe upper bound for modern
        # embedding models (BGE-M3 supports 8192 tokens). Callers
        # can override via kwargs.
        n_ctx = kwargs.pop("n_ctx", 8192)
        self._llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            verbose=kwargs.pop("verbose", False),
            **kwargs,
        )
        # Cache native dimension for the dimensions= validator.
        self._n_embd = int(getattr(self._llm, "n_embd", lambda: 0)())
        self._model_path = model_path
        self._loaded = True
        logger.info(
            "Loaded embedding model %s (n_embd=%d, n_ctx=%d)",
            model_path,
            self._n_embd or 0,
            n_ctx,
        )

    def unload(self) -> None:
        if self._llm is not None:
            # llama-cpp-python releases memory when the Llama instance
            # goes out of scope; explicit close() for newer versions.
            close = getattr(self._llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover — defensive
                    logger.warning("llama_cpp close() raised", exc_info=True)
            self._llm = None
        self._loaded = False
        self._model_path = ""

    def embed(
        self,
        inputs: list[str],
        *,
        truncate: bool = True,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")
        if not inputs:
            raise ValueError("inputs must be a non-empty list")
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("dimensions must be a positive integer")
            if self._n_embd and dimensions > self._n_embd:
                raise ValueError(
                    f"dimensions ({dimensions}) exceeds model's native size ({self._n_embd})"
                )

        vectors: list[list[float]] = []
        total_tokens = 0

        for text in inputs:
            # llama-cpp returns either a bare list[float] or a list
            # of lists depending on version; normalise to list[float].
            raw = self._llm.embed(text, truncate=truncate)
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                vec = raw[0]
            else:
                vec = list(raw)  # type: ignore[arg-type]

            if dimensions is not None and len(vec) > dimensions:
                vec = vec[:dimensions]

            vectors.append([float(x) for x in vec])

            # Token accounting: tokenize once to record usage. Free
            # if the backend exposes it; otherwise approximate with
            # word count / 0.75 (matches OpenAI's 4-chars-per-token
            # heuristic).
            try:
                tokens = self._llm.tokenize(text.encode("utf-8"))
                total_tokens += len(tokens)
            except Exception:  # pragma: no cover — defensive
                total_tokens += max(1, len(text) // 4)

        return EmbeddingResult(
            embeddings=vectors,
            total_tokens=total_tokens,
            model=self._model_path,
        )


# ----------------------------------------------------------------------
# Transformers adapter
# ----------------------------------------------------------------------


class TransformersEmbeddingEngine(EmbeddingEngine):
    """Embedding engine backed by sentence-transformers-style pooling.

    Loads the HuggingFace ``AutoModel`` + ``AutoTokenizer`` and
    mean-pools the last hidden state over the attention mask — the
    de-facto standard used by sentence-transformers and most
    embedding leaderboards (MTEB). Supports any BERT-like encoder,
    including multilingual models.

    Requires the ``[transformers]`` extra.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._n_embd: int | None = None
        self._device: str = "cpu"

    def load(self, model_path: str, **kwargs: Any) -> None:
        # Deferred import — optional dep.
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = kwargs.pop("device", None)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        trust_remote_code = kwargs.pop("trust_remote_code", False)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code).to(
            device
        )
        self._model.eval()
        self._device = device
        self._model_path = model_path
        # Pick up native embedding dimension from the model config.
        hidden_size = getattr(self._model.config, "hidden_size", None)
        if hidden_size is None:
            # Some models expose it via ``d_model`` (T5 family) or
            # ``embedding_size`` (Electra). Fall back gracefully.
            hidden_size = getattr(self._model.config, "d_model", None) or getattr(
                self._model.config, "embedding_size", None
            )
        self._n_embd = int(hidden_size) if hidden_size else None
        self._loaded = True
        logger.info(
            "Loaded embedding model %s on %s (hidden_size=%s)",
            model_path,
            device,
            self._n_embd,
        )

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._model_path = ""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover — defensive
            pass

    def embed(
        self,
        inputs: list[str],
        *,
        truncate: bool = True,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        if not inputs:
            raise ValueError("inputs must be a non-empty list")
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("dimensions must be a positive integer")
            if self._n_embd and dimensions > self._n_embd:
                raise ValueError(
                    f"dimensions ({dimensions}) exceeds model's native size ({self._n_embd})"
                )

        import torch

        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=truncate,
            return_tensors="pt",
        ).to(self._device)

        total_tokens = int(encoded["attention_mask"].sum().item())

        with torch.no_grad():
            outputs = self._model(**encoded)

        # Mean-pool over the non-padding tokens.
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        # L2-normalise for cosine-friendly consumption (matches
        # sentence-transformers default and what LangChain expects).
        norms = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        pooled = pooled / norms

        vectors = pooled.cpu().tolist()
        if dimensions is not None:
            vectors = [v[:dimensions] for v in vectors]

        return EmbeddingResult(
            embeddings=vectors,
            total_tokens=total_tokens,
            model=self._model_path,
        )
