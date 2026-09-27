# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
#
# The local audit on Linux, in a container: CPU only, no GPU.
#
#   docker build -t hfl-audit-linux -f audit/linux/Dockerfile .
#   docker run --rm -v hfl-audit-work:/work hfl-audit-linux --setup
#   docker run --rm -v hfl-audit-work:/work hfl-audit-linux --only A,B,D,E
#
# The repository is copied in; everything the audit makes (venvs, models,
# REPORT.md) stays in the /work volume.
FROM python:3.12-slim-bookworm

# git + a C/C++ toolchain + cmake: what a package without a wheel for this
# platform builds from source with.
# espeak-ng: speech to test transcription with. libportaudio2: the [audio] extra.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl espeak-ng libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv httpx websockets
# llama-cpp-python from its project's generic CPU wheels, as HFL's own image:
# built here it targets this CPU, and on arm64 under Docker Desktop gcc 12
# fails on the fp16 intrinsics ("target specific option mismatch").
ENV UV_EXTRA_INDEX_URL=https://abetlen.github.io/llama-cpp-python/whl/cpu
# uv's cache in the work volume, next to the venvs, so it hard-links each
# package into every venv instead of copying it: on Linux pip's torch
# carries CUDA (~5 GB), and with the cache in the container's own layer every
# extra's venv held a copy — the volume grew to 45 GB and filled Docker's disk.
ENV UV_CACHE_DIR=/work/.uv-cache UV_LINK_MODE=hardlink

WORKDIR /src
COPY . /src
ENTRYPOINT ["python", "-u", "audit/local_audit.py", "--work", "/work"]
