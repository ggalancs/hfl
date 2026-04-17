# syntax=docker/dockerfile:1.7
# HFL production image.
#
# Two-stage build: the builder compiles the C extensions (llama-cpp,
# accelerated BLAS) once, the runtime stage copies the wheel and the
# venv into a slim Python 3.12 image so the shipped layer stays
# ~850 MB instead of ~2 GB.
#
# Build args:
#   HFL_EXTRAS    — pip extras to install into the venv. Defaults to
#                   "llama" (CPU-only GGUF). Pass "all" for the full
#                   surface, or a comma-separated subset like
#                   "llama,transformers,mcp".
#   PYTHON_VERSION — pinned to 3.12 to match CI.
#
# Runtime:
#   docker run --rm -p 11434:11434 ghcr.io/ggalancs/hfl:latest serve
#
# HFL home is ``/var/lib/hfl`` — mount it to persist models.

ARG PYTHON_VERSION=3.12
ARG HFL_EXTRAS=llama

# --------------------------------------------------------------------
# Stage 1: builder
# --------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG HFL_EXTRAS

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: compilers + cmake for llama-cpp-python, git for
# occasional pip-from-VCS pins, curl for healthchecks in base image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Build a venv then install HFL into it. Using a venv (not the system
# site-packages) keeps the runtime copy simple.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip wheel \
 && /opt/venv/bin/pip install ".[${HFL_EXTRAS}]"

# --------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="hfl" \
      org.opencontainers.image.description="Run HuggingFace models locally (Ollama-compatible)" \
      org.opencontainers.image.source="https://github.com/ggalancs/hfl" \
      org.opencontainers.image.licenses="HRUL-1.0"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HFL_HOME=/var/lib/hfl \
    HFL_HOST=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 hfl \
 && useradd --system --uid 1000 --gid hfl --home /var/lib/hfl --shell /bin/bash hfl \
 && mkdir -p /var/lib/hfl \
 && chown -R hfl:hfl /var/lib/hfl

COPY --from=builder /opt/venv /opt/venv

USER hfl
WORKDIR /var/lib/hfl

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:11434/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "hfl"]
CMD ["serve", "--host", "0.0.0.0", "--port", "11434"]
