# SPDX-License-Identifier: HRUL-1.0
"""Extended tests for engine selector module."""


import pytest

from hfl.engine.selector import (
    MissingDependencyError,
    _create_engine,
)


class TestCreateEngineUnknown:
    """Tests for _create_engine with unknown backend."""

    def test_create_unknown_engine_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _create_engine("unknown-backend")

        assert "unknown" in str(exc_info.value).lower()


class TestHasCudaFunction:
    """Tests for _has_cuda function."""

    def test_has_cuda_function_exists(self):
        """Test _has_cuda function exists and can be imported."""
        from hfl.engine.selector import _has_cuda

        # Function should be callable
        assert callable(_has_cuda)


class TestMissingDependencyError:
    """Tests for MissingDependencyError exception."""

    def test_missing_dependency_error_message(self):
        """Test MissingDependencyError message."""
        error = MissingDependencyError("Test dependency missing")

        assert "Test dependency missing" in str(error)

    def test_missing_dependency_error_inheritance(self):
        """Test MissingDependencyError is an Exception."""
        error = MissingDependencyError("Test")

        assert isinstance(error, Exception)
