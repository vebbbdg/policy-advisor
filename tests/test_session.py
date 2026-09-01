"""
Unit tests for the session manager.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.session import SessionManager


class TestSessionManager:
    """Test suite for SessionManager."""

    def setup_method(self):
        self.sm = SessionManager()

    def test_create_session(self):
        sid = self.sm.create_session()
        assert sid is not None
        # 完整uuid4：36字符（含4个连字符），不可枚举
        assert len(sid) == 36
        assert self.sm.session_exists(sid)

    def test_create_session_with_title(self):
        sid = self.sm.create_session(title="My Chat")
        sessions = self.sm.list_sessions()
        assert sessions[0]["title"] == "My Chat"

    def test_add_and_get_messages(self):
        sid = self.sm.create_session()
        self.sm.add_message(sid, "user", "Hello")
        self.sm.add_message(sid, "assistant", "Hi there!")

        msgs = self.sm.get_messages(sid)
        # system prompt + user + assistant
        assert len(msgs) == 3
        assert msgs[1]["content"] == "Hello"
        assert msgs[2]["content"] == "Hi there!"

    def test_auto_title_from_first_message(self):
        sid = self.sm.create_session()
        self.sm.add_message(sid, "user", "What is Python programming?")
        sessions = self.sm.list_sessions()
        assert "Python" in sessions[0]["title"]

    def test_reset_session(self):
        sid = self.sm.create_session()
        self.sm.add_message(sid, "user", "Hello")
        self.sm.reset_session(sid)

        msgs = self.sm.get_messages(sid)
        assert len(msgs) == 1  # only system prompt
        assert msgs[0]["role"] == "system"

    def test_delete_session(self):
        sid = self.sm.create_session()
        assert self.sm.session_exists(sid)
        self.sm.delete_session(sid)
        assert not self.sm.session_exists(sid)

    def test_list_sessions_ordered(self):
        s1 = self.sm.create_session("First")
        s2 = self.sm.create_session("Second")
        sessions = self.sm.list_sessions()
        # Most recent first
        assert sessions[0]["id"] == s2
        assert sessions[1]["id"] == s1

    def test_get_nonexistent_session(self):
        assert self.sm.get_messages("nonexistent") == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
