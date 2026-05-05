# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Central configuration for hfl."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _parse_ollama_host_env() -> tuple[str | None, int | None]:
    """Parse the Ollama-compatible ``OLLAMA_HOST`` value.

    Ollama documents three accepted shapes for the variable:

    - ``"0.0.0.0"`` — host only (port stays at the default)
    - ``"0.0.0.0:11434"`` — explicit ``host:port``
    - ``":11434"`` — port only (host stays at the default)

    HFL respects all three so a drop-in replacement of an Ollama
    install works without re-reading the docs. Returns
    ``(host_or_None, port_or_None)``: the corresponding HFL setting
    overrides only the components actually present in the env value.
    A value that doesn't parse cleanly (``"abc:xyz"``) yields
    ``(None, None)`` and the caller falls back to defaults.
    """
    raw = os.environ.get("OLLAMA_HOST")
    if not raw:
        return None, None
    raw = raw.strip()
    if not raw:
        return None, None

    if ":" in raw:
        host_part, _, port_part = raw.rpartition(":")
        host = host_part or None
        try:
            port: int | None = int(port_part) if port_part else None
        except ValueError:
            return None, None
        return host, port

    # No colon — treat as host (the most common Ollama usage).
    return raw, None


def _parse_cors_origins_env() -> list[str] | None:
    """Read ``HFL_ORIGINS`` / ``OLLAMA_ORIGINS`` into a clean list.

    Returns ``None`` (i.e. "no env override") when neither variable is
    set, so the defaults defined on the dataclass remain authoritative.
    Otherwise splits on commas, strips whitespace, drops empties.
    """
    raw = os.environ.get("HFL_ORIGINS") or os.environ.get("OLLAMA_ORIGINS")
    if raw is None:
        return None
    items = [piece.strip() for piece in raw.split(",")]
    return [piece for piece in items if piece]


@dataclass
class SLOConfig:
    """Service Level Objective configuration.

    Defines performance targets for monitoring and alerting.
    These values are used by health checks to determine service status.
    """

    # Availability: target uptime percentage (0.0 - 1.0)
    availability_target: float = 0.999  # 99.9%

    # Latency targets in milliseconds for API endpoints
    # P50 = median, P95 = 95th percentile, P99 = 99th percentile
    latency_p50_ms: float = 100.0  # 100ms
    latency_p95_ms: float = 500.0  # 500ms
    latency_p99_ms: float = 1000.0  # 1s

    # Health check latency (should be very fast)
    health_latency_ms: float = 50.0  # 50ms

    # Error rate: maximum acceptable error rate (0.0 - 1.0)
    error_rate_target: float = 0.01  # 1%

    # Throughput: minimum requests per second (0 = no minimum)
    min_throughput_rps: float = 0.0

    # Model loading time limit in seconds
    model_load_time_limit: float = 60.0  # 1 minute

    # Memory usage threshold (0.0 - 1.0) for warnings
    memory_warning_threshold: float = 0.8  # 80%
    memory_critical_threshold: float = 0.95  # 95%

    # Time window for SLI calculations (seconds)
    sli_window_seconds: int = 300  # 5 minutes

    def validate(self) -> list[str]:
        """Validate SLO configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if not 0.0 <= self.availability_target <= 1.0:
            errors.append("availability_target must be between 0.0 and 1.0")

        if not 0.0 <= self.error_rate_target <= 1.0:
            errors.append("error_rate_target must be between 0.0 and 1.0")

        if self.latency_p50_ms < 0:
            errors.append("latency_p50_ms must be non-negative")

        if self.latency_p95_ms < self.latency_p50_ms:
            errors.append("latency_p95_ms should be >= latency_p50_ms")

        if self.latency_p99_ms < self.latency_p95_ms:
            errors.append("latency_p99_ms should be >= latency_p95_ms")

        if not 0.0 <= self.memory_warning_threshold <= 1.0:
            errors.append("memory_warning_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.memory_critical_threshold <= 1.0:
            errors.append("memory_critical_threshold must be between 0.0 and 1.0")

        if self.memory_warning_threshold >= self.memory_critical_threshold:
            errors.append("memory_warning_threshold should be < memory_critical_threshold")

        return errors


@dataclass
class HFLConfig:
    """Global application configuration."""

    # Root directory (~/.hfl by default)
    home_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("HFL_HOME", Path.home() / ".hfl"))
    )

    # Subdirectories
    @property
    def models_dir(self) -> Path:
        return self.home_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.home_dir / "cache"

    @property
    def registry_path(self) -> Path:
        return self.home_dir / "models.json"

    @property
    def llama_cpp_dir(self) -> Path:
        """Directory where llama.cpp is cloned/compiled for conversion."""
        return self.home_dir / "tools" / "llama.cpp"

    # Server bind address. Resolution order:
    #   1. ``HFL_HOST`` (explicit)
    #   2. host part of ``OLLAMA_HOST`` ("0.0.0.0" or "0.0.0.0:11434"
    #      or ":11434" — the host slot may be empty)
    #   3. Default ``"127.0.0.1"``
    host: str = field(
        default_factory=lambda: (
            os.environ.get("HFL_HOST") or _parse_ollama_host_env()[0] or "127.0.0.1"
        )
    )
    # Server port. Resolution order:
    #   1. ``HFL_PORT`` (explicit)
    #   2. port part of ``OLLAMA_HOST``
    #   3. ``OLLAMA_PORT`` (some Ollama deployments split host/port)
    #   4. Default ``11434``
    port: int = field(
        default_factory=lambda: int(
            os.environ.get("HFL_PORT")
            or (str(_parse_ollama_host_env()[1]) if _parse_ollama_host_env()[1] else "")
            or os.environ.get("OLLAMA_PORT")
            or "11434"
        )
    )

    # Security - CORS
    # By default, CORS is restrictive (same-origin only).
    # Set cors_allow_all=True for development or use explicit origins.
    # NOTE: allow_credentials must be False when cors_allow_all=True
    #
    # Operators can override the allow-list via env without editing
    # config: ``HFL_ORIGINS`` / ``OLLAMA_ORIGINS`` accepts a
    # comma-separated list. ``"*"`` (the only entry, or as a member)
    # is recognised and flips ``cors_allow_all`` on. Empty / unset
    # falls back to the documented same-origin default.
    cors_origins: list[str] = field(default_factory=lambda: _parse_cors_origins_env() or [])
    cors_allow_all: bool = field(default_factory=lambda: "*" in (_parse_cors_origins_env() or []))
    cors_allow_credentials: bool = False  # Must be False with wildcard origin
    cors_allow_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_allow_headers: list[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-Request-ID"]
    )

    # Security - Rate Limiting (configurable via HFL_RATE_LIMIT_* env vars)
    rate_limit_enabled: bool = field(
        default_factory=lambda: os.environ.get("HFL_RATE_LIMIT_ENABLED", "true").lower() == "true"
    )
    rate_limit_requests: int = field(
        default_factory=lambda: int(os.environ.get("HFL_RATE_LIMIT_REQUESTS", "60"))
    )
    rate_limit_window: int = field(
        default_factory=lambda: int(os.environ.get("HFL_RATE_LIMIT_WINDOW", "60"))
    )

    # LLM Inference (0 = auto-detect from model's GGUF metadata)
    default_ctx_size: int = field(
        default_factory=lambda: int(os.environ.get("HFL_DEFAULT_CTX_SIZE", "0"))
    )
    default_n_gpu_layers: int = -1  # -1 = all layers to GPU
    default_threads: int = 0  # 0 = auto-detect

    # KV cache dtype (Phase 11 P1 — OLLAMA_PARITY_PLAN_V2 row 9).
    # Quantising the KV cache halves / quarters its VRAM footprint
    # at the cost of some accuracy, mirroring Ollama's
    # ``OLLAMA_KV_CACHE_TYPE``. One of: ``"f16"`` (default, no
    # quantisation), ``"q8_0"`` (half VRAM, negligible quality
    # loss), ``"q4_0"`` (quarter VRAM, visible on small models).
    kv_cache_type: str = field(default_factory=lambda: os.environ.get("HFL_KV_CACHE_TYPE", "f16"))

    # Default keep-alive duration applied to /api/chat and /api/generate
    # requests that did *not* set ``keep_alive`` themselves. Matches
    # Ollama's ``OLLAMA_KEEP_ALIVE`` (default "5m"). Per-request values
    # always win — the global only kicks in when the client omits the
    # field. Accepts the Ollama duration grammar ("5m", "30s", "0",
    # "-1" for never-expire, etc.); validated lazily by the existing
    # ``parse_keep_alive`` so a bad value here fails the first
    # request, not the import.
    keep_alive_default: str | None = field(
        default_factory=lambda: (
            os.environ.get("HFL_KEEP_ALIVE") or os.environ.get("OLLAMA_KEEP_ALIVE") or "5m"
        )
    )

    # Prefix cache across requests (Phase 11 P1 — V2 row 10). When
    # True, the engine reuses KV-cache state when the current prompt
    # shares a prefix with a recently-served one. Only the latest-N
    # prefixes are retained; ``prompt_cache_max_entries`` caps the
    # LRU.
    prompt_cache_enabled: bool = field(
        default_factory=lambda: os.environ.get("HFL_PROMPT_CACHE_ENABLED", "true").lower() == "true"
    )
    prompt_cache_max_entries: int = field(
        default_factory=lambda: int(os.environ.get("HFL_PROMPT_CACHE_MAX_ENTRIES", "32"))
    )

    # TTS defaults
    default_tts_sample_rate: int = 22050
    default_tts_format: str = "wav"  # wav, mp3, ogg

    # Timeouts (seconds)
    model_load_timeout: float = 300.0  # 5 minutes
    generation_timeout: float = 600.0  # 10 minutes
    download_timeout: float = 3600.0  # 1 hour
    conversion_timeout: float = 7200.0  # 2 hours
    api_request_timeout: float = 120.0  # 2 minutes

    # Maximum request body size (bytes) — prevents DoS by oversized prompts.
    # Default 10 MiB: comfortably fits legitimate multi-turn conversations
    # (a 128k-token prompt at ~4 chars/token is ~512 KB) while rejecting
    # obvious abuse. Override with HFL_MAX_REQUEST_BYTES=0 to disable.
    max_request_bytes: int = field(
        default_factory=lambda: int(os.environ.get("HFL_MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))
    )

    # Inference dispatcher (spec §5.3 — concurrency / queueing).
    # Llama.cpp and transformers-GPU share a single non-reentrant model
    # instance; concurrent requests must be serialized by a bounded
    # in-server queue. ``queue_max_inflight`` is the number of requests
    # that may run at once, ``queue_max_size`` is how many more may wait
    # in line before further requests are rejected with 429, and
    # ``queue_acquire_timeout_seconds`` caps how long a caller may wait
    # for a slot before giving up with 503.
    queue_enabled: bool = field(
        default_factory=lambda: os.environ.get("HFL_QUEUE_ENABLED", "true").lower() == "true"
    )
    # Resolution order for the in-flight slot count, in priority:
    #   1. ``HFL_QUEUE_MAX_INFLIGHT`` (explicit, the original key)
    #   2. ``HFL_NUM_PARALLEL`` (Ollama-equivalent name; what most
    #      operators reach for first, since Ollama exposes the same
    #      knob as ``OLLAMA_NUM_PARALLEL``)
    #   3. ``OLLAMA_NUM_PARALLEL`` (drop-in replacement of an Ollama
    #      install whose env vars are already set)
    #   4. Default ``1`` — preserves V1 behaviour (single-flight).
    queue_max_inflight: int = field(
        default_factory=lambda: int(
            os.environ.get("HFL_QUEUE_MAX_INFLIGHT")
            or os.environ.get("HFL_NUM_PARALLEL")
            or os.environ.get("OLLAMA_NUM_PARALLEL")
            or "1"
        )
    )
    # Max queue depth, with the same Ollama-fallback chain as the
    # in-flight cap.
    queue_max_size: int = field(
        default_factory=lambda: int(
            os.environ.get("HFL_QUEUE_MAX_SIZE")
            or os.environ.get("HFL_MAX_QUEUE")
            or os.environ.get("OLLAMA_MAX_QUEUE")
            or "16"
        )
    )
    queue_acquire_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("HFL_QUEUE_ACQUIRE_TIMEOUT", "60"))
    )

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    # Streaming queue timeouts (seconds). These previously lived as
    # magic numbers inside ``engine/async_wrapper.py``,
    # ``engine/vllm_engine.py``, and ``api/streaming.py``. Centralising
    # them here lets operators tune streaming backpressure without a
    # code change.
    stream_queue_put_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HFL_STREAM_QUEUE_PUT_TIMEOUT", "60"))
    )
    stream_queue_get_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HFL_STREAM_QUEUE_GET_TIMEOUT", "30"))
    )
    # vLLM-specific: time allowed for the error sentinel to reach the
    # consumer when the worker thread failed. Shorter than the regular
    # put timeout because a dying stream shouldn't wait for
    # backpressure relief.
    vllm_error_put_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HFL_VLLM_ERROR_PUT_TIMEOUT", "10"))
    )
    # vLLM worker-thread join timeout during shutdown.
    vllm_shutdown_join_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HFL_VLLM_SHUTDOWN_JOIN_TIMEOUT", "5"))
    )
    # SQLite registry backend: busy-timeout (seconds) waiting for a
    # lock before raising ``OperationalError``.
    registry_sqlite_busy_timeout: float = field(
        default_factory=lambda: float(os.environ.get("HFL_REGISTRY_SQLITE_TIMEOUT", "30"))
    )

    # Service Level Objectives (SLOs)
    slo: SLOConfig = field(default_factory=SLOConfig)

    # HuggingFace
    # PRIVACY (R6 - Legal Audit): hf_token is read ONLY from environment variable.
    # It is NEVER persisted to disk, NEVER stored in models.json or any config file.
    # Token is held in memory only for the duration of the process.
    #
    # Known limitation (platform, not a bug): Python strings are
    # immutable, so once ``hf_token`` is assigned the original bytes
    # cannot be reliably overwritten — a memory dump of a running
    # process will still contain the secret. This is true for every
    # Python program reading a secret into a ``str``. Mitigations:
    # (a) prefer ``huggingface-cli login`` which writes a token file
    # with 0600 permissions and does not require setting an env var,
    # (b) restrict the env var's scope to the systemd unit / shell
    # session that launches ``hfl``, (c) run HFL under an unprivileged
    # user whose memory cannot be inspected by other processes.
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        errors = self.slo.validate()
        if errors:
            import logging

            logging.getLogger(__name__).warning("Invalid SLO configuration: %s", "; ".join(errors))

        # CORS: the wildcard + credentials combination is rejected by every
        # modern browser (W3C Fetch §3.2.1), but Starlette/FastAPI will
        # still accept it and emit the headers — a silent misconfiguration.
        # Reject at construction time so the error surfaces on import/boot
        # rather than after the server has been serving broken CORS for
        # hours. ``cors_origins=["*"]`` explicit value is treated the same
        # as ``cors_allow_all=True``.
        wildcard = self.cors_allow_all or self.cors_origins == ["*"]
        if wildcard and self.cors_allow_credentials:
            raise ValueError(
                "CORS misconfiguration: cors_allow_credentials=True is not compatible "
                "with wildcard origins (cors_allow_all=True or cors_origins=['*']). "
                "Browsers will reject the response; either set explicit origins or "
                "disable credentials."
            )

    def ensure_dirs(self):
        """Creates the necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Initialize registry if it doesn't exist
        if not self.registry_path.exists():
            self.registry_path.write_text("[]")


# Global instance
config = HFLConfig()


def _safe_ensure_dirs() -> None:
    """Ensure directories exist, ignoring errors in read-only environments."""
    try:
        config.ensure_dirs()
    except OSError:
        pass  # Read-only filesystem or test environment


_safe_ensure_dirs()
