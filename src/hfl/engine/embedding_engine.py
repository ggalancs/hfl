# SPDX-License-Identifier: Apache-2.0
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
from typing import TYPE_CHECKING, Any, cast

from hfl.engine.embedding_pooling import POOLING_STRATEGIES, Pooling, pool

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
        pooling: str = "mean",
    ) -> EmbeddingResult:
        """Produce embeddings for a batch of input strings.

        Args:
            inputs: Strings to embed. Empty list is rejected by the
                router; engines may assume non-empty.
            pooling: How token embeddings collapse into one vector —
                ``mean``, ``cls`` or ``last``. A model trained with CLS
                pooling and served with mean pooling returns vectors that
                are not obviously wrong, just quietly worse at retrieval,
                which is why this is explicit rather than inferred. An
                engine that cannot honour the request must raise rather
                than silently pool some other way.
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
        # An encoder takes a whole input in one batch: with llama.cpp's
        # default of 512, an 846-token input (well within the context) was
        # refused, or cut to 512 with ``truncate`` (measured).
        kwargs.setdefault("n_batch", n_ctx)
        kwargs.setdefault("n_ubatch", n_ctx)
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
        pooling: str = "mean",
    ) -> EmbeddingResult:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Model not loaded")
        if pooling != "mean":
            # llama.cpp pools inside the C library and hands back one
            # vector per input; the token matrix never reaches Python, so
            # CLS or last-token pooling cannot be applied after the fact.
            # Accepting the argument and ignoring it would return
            # mean-pooled vectors labelled as something else.
            raise ValueError(
                f"pooling={pooling!r} is not available on the llama.cpp embedding "
                "backend, which pools internally. Serve this model through the "
                "transformers backend, or request pooling='mean'."
            )
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
            # normalize: unit length, as Ollama and OpenAI return them
            # (llama-cpp-python's default is not to: measured norms of 5+).
            raw = self._llm.embed(text, truncate=truncate, normalize=True)
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                vec = raw[0]
            else:
                vec = list(raw)

            if dimensions is not None and len(vec) > dimensions:
                vec = vec[:dimensions]
                # Matryoshka (ENG-5): the full vector is unit-norm, so
                # slicing alone yields norm < 1. Re-normalise the truncated
                # vector to keep it unit-norm for cosine/IP consumers.
                norm = sum(x * x for x in vec) ** 0.5
                if norm > 0:
                    vec = [x / norm for x in vec]

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
# llama-server adapter
# ----------------------------------------------------------------------


def _unit(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


class LlamaServerEmbeddingEngine(EmbeddingEngine):
    """GGUF embeddings through a ``llama-server --embeddings`` process, for
    an HFL without llama-cpp-python — Homebrew's, which serves GGUF chat
    models through Homebrew's llama.cpp and could not embed with them.

    Same contract as :class:`LlamaCppEmbeddingEngine`: unit-length vectors,
    the model's own pooling (so only ``pooling="mean"`` is accepted, as
    there), ``truncate`` and ``dimensions``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._proc: Any | None = None
        self._client: Any | None = None
        self._n_embd: int | None = None
        self._n_ctx = 0

    def load(self, model_path: str, **kwargs: Any) -> None:
        from pathlib import Path

        from hfl.config import config
        from hfl.converter.gguf_header import read_fields
        from hfl.engine.llama_server import binary, start_server

        exe = binary()
        if exe is None:
            raise RuntimeError(
                "GGUF embeddings need llama-cpp-python or llama.cpp's llama-server: "
                "install one (e.g. `brew install llama.cpp`)"
            )
        try:
            arch = read_fields(model_path, {"general.architecture"}).get("general.architecture")
            keys = {f"{arch}.context_length", f"{arch}.embedding_length"}
            fields = read_fields(model_path, keys)
        except (OSError, ValueError):
            fields = {}
        trained = fields.get(f"{arch}.context_length")
        # 8192 at most, as the llama-cpp-python engine; a whole input is one
        # batch for an encoder, so the batch is the context.
        n_ctx = int(kwargs.get("n_ctx") or min(int(trained or 8192), 8192))
        argv = [
            exe, "-m", model_path, "--host", "127.0.0.1", "--embeddings",
            "-c", str(n_ctx), "-b", str(n_ctx), "-ub", str(n_ctx), "-np", "1",
            "-ngl", "999", "--no-webui", "--no-slots",
        ]  # fmt: skip
        log_dir = config.home_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-server-embed-{Path(model_path).stem}.log"
        timeout = float(getattr(config, "model_load_timeout", 600) or 600)
        self._proc, self._client = start_server(argv, model_path, log_path, timeout)
        length = fields.get(f"{arch}.embedding_length")
        self._n_embd = int(length) if isinstance(length, int) else None
        self._n_ctx = n_ctx
        self._model_path = model_path
        self._loaded = True
        logger.info("Loaded embedding model %s through llama-server", model_path)

    def unload(self) -> None:
        from hfl.engine.llama_server import stop_server

        if self._client is not None:
            self._client.close()
            self._client = None
        proc, self._proc = self._proc, None
        stop_server(proc)
        self._loaded = False
        self._model_path = ""

    def _fit(self, text: str, truncate: bool) -> str:
        """``text`` cut to the context when ``truncate``; else an error."""
        assert self._client is not None
        tokens = self._client.post("/tokenize", json={"content": text}).json()["tokens"]
        limit = self._n_ctx - 2  # room for the encoder's own CLS/SEP
        if len(tokens) <= limit:
            return text
        if not truncate:
            raise ValueError(f"input of {len(tokens)} tokens exceeds the model's context ({limit})")
        cut = self._client.post("/detokenize", json={"tokens": tokens[:limit]})
        return str(cut.json()["content"])

    def embed(
        self,
        inputs: list[str],
        *,
        truncate: bool = True,
        dimensions: int | None = None,
        pooling: str = "mean",
    ) -> EmbeddingResult:
        if not self._loaded or self._client is None:
            raise RuntimeError("Model not loaded")
        if pooling != "mean":
            raise ValueError(
                f"pooling={pooling!r} is not available on the llama.cpp embedding "
                "backend, which pools internally. Serve this model through the "
                "transformers backend, or request pooling='mean'."
            )
        if not inputs:
            raise ValueError("inputs must be a non-empty list")
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("dimensions must be a positive integer")
            if self._n_embd and dimensions > self._n_embd:
                raise ValueError(
                    f"dimensions ({dimensions}) exceeds model's native size ({self._n_embd})"
                )
        texts = [self._fit(text, truncate) for text in inputs]
        response = self._client.post("/v1/embeddings", json={"input": texts})
        if response.status_code != 200:
            raise RuntimeError(f"llama-server could not embed: HTTP {response.status_code}")
        body = response.json()
        rows = sorted(body["data"], key=lambda row: row["index"])
        vectors = []
        for row in rows:
            vec = [float(x) for x in row["embedding"]]
            if dimensions is not None and len(vec) > dimensions:
                vec = vec[:dimensions]
            vectors.append(_unit(vec))
        usage = body.get("usage") or {}
        return EmbeddingResult(
            embeddings=vectors,
            total_tokens=int(usage.get("prompt_tokens") or 0),
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

        from hfl.security import remote_code_allowed

        # Gate trust_remote_code behind operator opt-in (HFL_ALLOW_REMOTE_CODE);
        # an untrusted caller must never be able to execute model-repo Python.
        trust_remote_code = bool(kwargs.pop("trust_remote_code", False)) and remote_code_allowed()
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
        pooling: str = "mean",
    ) -> EmbeddingResult:
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(
                f"unknown pooling {pooling!r}; expected one of {', '.join(POOLING_STRATEGIES)}"
            )
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

        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        if pooling == "mean":
            # Kept as the original tensor path rather than routed through
            # ``pool``: it is the default, it is what every existing vector
            # was produced with, and a rewrite would risk moving the last
            # bit of a number for no gain.
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            # CLS / last-token go through the shared implementation so the
            # strategies have exactly one definition in the codebase.
            masks = encoded["attention_mask"].tolist()
            rows = [
                pool(seq.tolist(), row_mask, cast("Pooling", pooling))
                for seq, row_mask in zip(last_hidden, masks)
            ]
            pooled = torch.tensor(rows, dtype=last_hidden.dtype, device=last_hidden.device)
        # Matryoshka (ENG-5): truncate BEFORE the final L2-normalisation so
        # the returned vector is unit-norm at the requested dimensionality.
        # Normalising first and slicing afterwards leaves norm < 1, which
        # breaks cosine/dot-product consumers (FAISS inner-product indexes,
        # LangChain) that assume unit vectors — the exact contract
        # text-embedding-3 / MRL define.
        if dimensions is not None:
            pooled = pooled[:, :dimensions]
        # L2-normalise for cosine-friendly consumption (matches
        # sentence-transformers default and what LangChain expects).
        norms = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        pooled = pooled / norms

        vectors = pooled.cpu().tolist()

        return EmbeddingResult(
            embeddings=vectors,
            total_tokens=total_tokens,
            model=self._model_path,
        )
