# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Central configuration for hfl."""

import os
from dataclasses import dataclass, field
from pathlib import Path


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

    # Server (configurable via HFL_HOST / HFL_PORT env vars)
    host: str = field(default_factory=lambda: os.environ.get("HFL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("HFL_PORT", "11434")))

    # Security - CORS
    # By default, CORS is restrictive (same-origin only).
    # Set cors_allow_all=True for development or use explicit origins.
    # NOTE: allow_credentials must be False when cors_allow_all=True
    cors_origins: list[str] = field(default_factory=list)  # Empty = same-origin only
    cors_allow_all: bool = False  # Explicit opt-in for "*" (all origins)
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

    # TTS defaults
    default_tts_sample_rate: int = 22050
    default_tts_format: str = "wav"  # wav, mp3, ogg

    # Timeouts (seconds)
    model_load_timeout: float = 300.0  # 5 minutes
    generation_timeout: float = 600.0  # 10 minutes
    download_timeout: float = 3600.0  # 1 hour
    conversion_timeout: float = 7200.0  # 2 hours
    api_request_timeout: float = 120.0  # 2 minutes

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
    queue_max_inflight: int = field(
        default_factory=lambda: int(os.environ.get("HFL_QUEUE_MAX_INFLIGHT", "1"))
    )
    queue_max_size: int = field(
        default_factory=lambda: int(os.environ.get("HFL_QUEUE_MAX_SIZE", "16"))
    )
    queue_acquire_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("HFL_QUEUE_ACQUIRE_TIMEOUT", "60"))
    )

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    # Service Level Objectives (SLOs)
    slo: SLOConfig = field(default_factory=SLOConfig)

    # HuggingFace
    # PRIVACY (R6 - Legal Audit): hf_token is read ONLY from environment variable.
    # It is NEVER persisted to disk, NEVER stored in models.json or any config file.
    # Token is held in memory only for the duration of the process.
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        errors = self.slo.validate()
        if errors:
            import logging

            logging.getLogger(__name__).warning("Invalid SLO configuration: %s", "; ".join(errors))

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
