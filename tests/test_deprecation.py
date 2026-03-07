# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for deprecation utilities."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from hfl.api.deprecation import (
    add_deprecation_headers,
    deprecated_endpoint,
    format_sunset_date,
)


class TestAddDeprecationHeaders:
    """Tests for add_deprecation_headers function."""

    def test_basic_deprecation(self):
        """Should add Deprecation: true header."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(response)

        assert response.headers["Deprecation"] == "true"

    def test_deprecation_with_date(self):
        """Should add timestamp-based deprecation header."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(response, deprecated_at="2026-01-15T00:00:00")

        assert response.headers["Deprecation"].startswith("@")
        # Extract timestamp and verify it's a valid number
        timestamp = int(response.headers["Deprecation"][1:])
        assert timestamp > 0

    def test_sunset_header(self):
        """Should add Sunset header in HTTP date format."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(response, sunset="2027-06-01T00:00:00")

        assert "Sunset" in response.headers
        assert "2027" in response.headers["Sunset"]
        assert "Jun" in response.headers["Sunset"]

    def test_alternative_link(self):
        """Should add Link header for alternative."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(response, alternative="/v2/chat/completions")

        assert "Link" in response.headers
        assert "/v2/chat/completions" in response.headers["Link"]
        assert 'rel="successor-version"' in response.headers["Link"]

    def test_all_headers_combined(self):
        """Should add all headers when all params provided."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(
            response,
            deprecated_at="2026-01-01T00:00:00",
            sunset="2027-01-01T00:00:00",
            alternative="/v2/endpoint",
        )

        assert "Deprecation" in response.headers
        assert "Sunset" in response.headers
        assert "Link" in response.headers

    def test_invalid_date_falls_back_to_true(self):
        """Should fall back to Deprecation: true for invalid date."""
        response = MagicMock()
        response.headers = {}

        add_deprecation_headers(response, deprecated_at="not-a-date")

        assert response.headers["Deprecation"] == "true"


class TestDeprecatedEndpointDecorator:
    """Tests for deprecated_endpoint decorator."""

    @pytest.mark.asyncio
    async def test_decorator_returns_result(self):
        """Decorator should not modify the return value."""

        @deprecated_endpoint()
        async def my_endpoint():
            return {"data": "value"}

        result = await my_endpoint()
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_decorator_adds_warning_to_dict(self):
        """Decorator should add warning to dict response."""

        @deprecated_endpoint(message="This endpoint is deprecated")
        async def my_endpoint():
            return {"data": "value"}

        result = await my_endpoint()
        assert result["_deprecation_warning"] == "This endpoint is deprecated"

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        """Decorator should preserve the function name."""

        @deprecated_endpoint()
        async def my_unique_endpoint():
            return {}

        assert my_unique_endpoint.__name__ == "my_unique_endpoint"


class TestFormatSunsetDate:
    """Tests for format_sunset_date function."""

    def test_formats_date_correctly(self):
        """Should format date as HTTP date."""
        dt = datetime(2027, 1, 15, 12, 30, 45)
        result = format_sunset_date(dt)

        assert "2027" in result
        assert "Jan" in result
        assert "GMT" in result

    def test_day_of_week_included(self):
        """Should include day of week."""
        dt = datetime(2027, 1, 1, 0, 0, 0)  # Friday
        result = format_sunset_date(dt)

        assert "Fri" in result
