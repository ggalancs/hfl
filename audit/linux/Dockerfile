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
FROM python:3.12-slim-trixie

# git + a C/C++ toolchain + cmake: what a package without a wheel for this
# platform builds from source with.
# espeak-ng: speech to test transcription with. libportaudio2: the [audio] extra.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl espeak-ng libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv httpx websockets
# llama.cpp's llama-server, the official release (the build Homebrew ships on
# the Mac audit), for the checks that need it (E2, E14, D21, D22, D33, D54,
# D55); checksums pinned, a mismatch stops the build.
ARG TARGETARCH
ARG LLAMA_BUILD=b10964
RUN set -e; \
    case "$TARGETARCH" in \
      arm64) arch=arm64; sum=5f0e9c95d970892e43380f82ebcab960edfd20a1cd0f7abffa13b29fdb924949 ;; \
      amd64) arch=x64;   sum=9abf88aea48a55d0f80edb1ee20220b186848cca0b4e919d71518cfd7ca67443 ;; \
      *) echo "no llama.cpp build for $TARGETARCH"; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-${arch}.tar.gz"; \
    echo "$sum  /tmp/llama.tgz" | sha256sum -c -; \
    mkdir -p /opt/llama && tar xzf /tmp/llama.tgz -C /opt/llama --strip-components=1; \
    rm /tmp/llama.tgz
ENV PATH=/opt/llama:$PATH
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
