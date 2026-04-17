# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for ``hfl.utils.duration.parse_keep_alive``.

The parser accepts the full matrix of Ollama-compatible values:
number, Go-duration string, ``0``, ``-1``, ``None``. Getting any of
these wrong means silently different behaviour from Ollama — e.g.
``"5m"`` parsed as 5 seconds would unload the model 60× too fast.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from hfl.utils.duration import (
    NEVER_EXPIRE,
    UNLOAD_AFTER,
    InvalidKeepAliveError,
    is_never_expire,
    is_unload_immediately,
    parse_keep_alive,
)


class TestNone:
    def test_none_returns_none(self):
        """Missing field → None (use default idle timeout)."""
        assert parse_keep_alive(None) is None

    def test_empty_string_returns_none(self):
        assert parse_keep_alive("") is None
        assert parse_keep_alive("   ") is None


class TestNumeric:
    def test_zero_is_unload_immediately(self):
        assert parse_keep_alive(0) == UNLOAD_AFTER
        assert parse_keep_alive(0.0) == UNLOAD_AFTER

    def test_minus_one_is_never_expire(self):
        assert parse_keep_alive(-1) == NEVER_EXPIRE
        assert parse_keep_alive(-1.0) == NEVER_EXPIRE

    def test_positive_int_is_seconds(self):
        assert parse_keep_alive(10) == timedelta(seconds=10)
        assert parse_keep_alive(3600) == timedelta(hours=1)

    def test_positive_float_is_fractional_seconds(self):
        assert parse_keep_alive(1.5) == timedelta(seconds=1.5)

    def test_negative_values_besides_minus_one_are_rejected(self):
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(-2)
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(-0.5)


class TestGoDurationString:
    @pytest.mark.parametrize(
        "s,expected",
        [
            ("5s", timedelta(seconds=5)),
            ("30s", timedelta(seconds=30)),
            ("5m", timedelta(minutes=5)),
            ("1h", timedelta(hours=1)),
            ("1h30m", timedelta(hours=1, minutes=30)),
            ("2h15m10s", timedelta(hours=2, minutes=15, seconds=10)),
            ("500ms", timedelta(milliseconds=500)),
            ("0.5s", timedelta(seconds=0.5)),
        ],
    )
    def test_common_durations(self, s, expected):
        assert parse_keep_alive(s) == expected

    def test_ollama_default_five_minutes(self):
        """``"5m"`` is Ollama's documented default — pin it explicitly."""
        assert parse_keep_alive("5m") == timedelta(minutes=5)

    def test_plain_number_string_is_seconds(self):
        assert parse_keep_alive("10") == timedelta(seconds=10)
        assert parse_keep_alive("3600") == timedelta(hours=1)

    def test_zero_string_is_unload(self):
        assert parse_keep_alive("0") == UNLOAD_AFTER

    def test_minus_one_string_is_never(self):
        assert parse_keep_alive("-1") == NEVER_EXPIRE

    def test_minus_one_with_unit_is_never(self):
        """``"-1s"`` decodes as NEVER_EXPIRE because total_seconds==1
        and leading sign is negative (mirrors Ollama's parser).
        """
        assert parse_keep_alive("-1s") == NEVER_EXPIRE

    def test_microsecond_units_parsed(self):
        assert parse_keep_alive("500us") == timedelta(microseconds=500)
        assert parse_keep_alive("500µs") == timedelta(microseconds=500)
        assert parse_keep_alive("1000ns") == timedelta(microseconds=1)


class TestInvalidStrings:
    @pytest.mark.parametrize(
        "s",
        [
            "5minutes",  # unit not recognised
            "1d",  # days not supported by Ollama
            "abc",
            "1h abc",
            "-2m",  # negative durations besides -1 are rejected
        ],
    )
    def test_rejects_bad_format(self, s):
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(s)

    def test_rejects_bool(self):
        """Booleans are ints in Python — explicit rejection to prevent
        ``True``/``False`` silently meaning 1/0 seconds."""
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(True)
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(False)

    def test_rejects_other_types(self):
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive([5])  # type: ignore[arg-type]
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive({"m": 5})  # type: ignore[arg-type]


class TestTimedeltaPassthrough:
    def test_timedelta_returned_unchanged(self):
        td = timedelta(minutes=3)
        assert parse_keep_alive(td) is td

    def test_never_expire_sentinel_passthrough(self):
        assert parse_keep_alive(NEVER_EXPIRE) == NEVER_EXPIRE

    def test_negative_timedelta_rejected(self):
        with pytest.raises(InvalidKeepAliveError):
            parse_keep_alive(timedelta(seconds=-5))


class TestSentinelPredicates:
    def test_is_never_expire_true(self):
        assert is_never_expire(NEVER_EXPIRE)
        assert is_never_expire(parse_keep_alive(-1))
        assert is_never_expire(parse_keep_alive("-1"))

    def test_is_never_expire_false(self):
        assert not is_never_expire(None)
        assert not is_never_expire(timedelta(seconds=300))
        assert not is_never_expire(UNLOAD_AFTER)

    def test_is_unload_immediately(self):
        assert is_unload_immediately(UNLOAD_AFTER)
        assert is_unload_immediately(parse_keep_alive(0))
        assert is_unload_immediately(parse_keep_alive("0"))

    def test_is_unload_immediately_false(self):
        assert not is_unload_immediately(None)
        assert not is_unload_immediately(NEVER_EXPIRE)
        assert not is_unload_immediately(timedelta(seconds=1))
