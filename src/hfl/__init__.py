# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""hfl - Download, run and try any Hugging Face model on your own machine."""

__version__ = "0.21.0"

# Every Hub API call gets a connect timeout. Without it, a network that drops
# packets instead of refusing them hung each Hub command forever. Imports
# nothing heavy: it arms a hook for when huggingface_hub loads.
from hfl.hub.timeouts import install_hub_timeouts as _install_hub_timeouts  # noqa: E402

_install_hub_timeouts()
