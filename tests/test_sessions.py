# SPDX-License-Identifier: HRUL-1.0
# Copyright (c) 2026 Gabriel Galán Pelayo
"""Tests for session persistence (Phase 18 P3 — V2 row 36)."""

from __future__ import annotations

import pytest

from hfl.core import sessions


class TestNameValidation:
    def test_valid_names(self, temp_config):
        for name in ["abc", "A_1", "x.y-z", "a" * 64]:
            session = sessions.ChatSession(name=name, model="m")
            sessions.save_session(session)

    def test_rejects_traversal(self, temp_config):
        for name in ["../evil", "a/b", ".hidden", "with space"]:
            with pytest.raises(sessions.InvalidSessionNameError):
                sessions.save_session(sessions.ChatSession(name=name, model="m"))

    def test_rejects_over_length(self, temp_config):
        name = "a" * 65
        with pytest.raises(sessions.InvalidSessionNameError):
            sessions.save_session(sessions.ChatSession(name=name, model="m"))


class TestRoundTrip:
    def test_save_load(self, temp_config):
        session = sessions.ChatSession(
            name="chat1",
            model="llama3.3",
            options={"temperature": 0.7},
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello!"},
            ],
            system="You are helpful.",
        )
        sessions.save_session(session)
        loaded = sessions.load_session("chat1")
        assert loaded.name == "chat1"
        assert loaded.model == "llama3.3"
        assert loaded.options == {"temperature": 0.7}
        assert loaded.messages[0]["role"] == "user"
        assert loaded.system == "You are helpful."

    def test_save_refreshes_updated_at(self, temp_config):
        s = sessions.ChatSession(name="r", model="m")
        first_updated = s.updated_at
        sessions.save_session(s)
        second_updated = s.updated_at
        assert second_updated >= first_updated


class TestListDelete:
    def test_list_sorts_by_updated_desc(self, temp_config):
        import time

        a = sessions.ChatSession(name="a", model="m")
        sessions.save_session(a)
        time.sleep(1.1)
        b = sessions.ChatSession(name="b", model="m")
        sessions.save_session(b)
        listed = sessions.list_sessions()
        assert [s.name for s in listed[:2]] == ["b", "a"]

    def test_list_skips_malformed(self, temp_config):
        (sessions.sessions_dir() / "broken.json").write_text("not-json")
        listed = sessions.list_sessions()
        assert all(s.name != "broken" for s in listed)

    def test_delete_returns_false_for_missing(self, temp_config):
        assert sessions.delete_session("ghost") is False

    def test_delete_removes_file(self, temp_config):
        sessions.save_session(sessions.ChatSession(name="bye", model="m"))
        assert sessions.delete_session("bye") is True
        assert sessions.delete_session("bye") is False


class TestUnknownSession:
    def test_load_missing_raises(self, temp_config):
        with pytest.raises(sessions.SessionNotFoundError):
            sessions.load_session("ghost")
