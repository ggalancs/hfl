# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Inference engine module for hfl."""

import os

# Transformers loads weights on a pool of threads, and with safetensors 0.8.0
# two of them can deadlock on a lazily-initialised value in its Rust binding
# (a pyo3 once-cell set up with the GIL released, awaited by a thread holding
# it). Seen once in HFL's server — the whole process froze, its own timeouts
# included, since the GIL was held — and reported elsewhere with the same
# versions; never reproduced here (0 in ~20,000 loads), no fixed release yet.
# Loading on one thread removes the condition; measured no slower for 0.5B
# and 1.1B models. An explicit HF_DEACTIVATE_ASYNC_LOAD from the user wins.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
