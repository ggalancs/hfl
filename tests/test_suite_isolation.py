# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""The test suite never reads or writes the developer's real ``~/.hfl``.

Isolation used to be opt-in (the ``temp_config`` fixture): a test that
forgot it registered its fake models in the real registry, where they
showed up in ``hfl list`` — five modelfile tests did exactly that.
The root conftest now points ``HFL_HOME`` at a throwaway directory for
the whole session, so forgetting is harmless.
"""

from __future__ import annotations

from pathlib import Path

REAL_HOME = (Path.home() / ".hfl").resolve()


def test_the_default_home_is_not_the_real_one():
    from hfl.config import HFLConfig

    assert HFLConfig().home_dir.resolve() != REAL_HOME


def test_the_global_config_is_not_the_real_one():
    import hfl.config

    assert hfl.config.config.home_dir.resolve() != REAL_HOME


def test_a_registry_without_temp_config_stays_out_of_the_real_home():
    from hfl.models.registry import ModelRegistry

    assert ModelRegistry().path.resolve().parent != REAL_HOME
