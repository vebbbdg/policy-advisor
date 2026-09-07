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
        # 每个测试用独立的内存库：互不干扰，也不碰生产 data/app.db
        self.sm = SessionManager("sqlite:///:memory:")

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

    def test_persistence_across_instances(self, tmp_path):
        """阶段 3.1 核心：数据落盘后，新建实例（模拟服务重启）仍能读到"""
        db_url = f"sqlite:///{tmp_path / 'persist.db'}"

        # 第一个实例：写入数据后释放文件句柄（Windows 上 tmp_path 自动清理需要）
        sm1 = SessionManager(db_url)
        sid = sm1.create_session("Persisted")
        sm1.add_message(sid, "user", "hello")
        sm1.engine.dispose()

        # 第二个实例：全新对象、同一个库文件 —— 模拟重启
        sm2 = SessionManager(db_url)
        assert sm2.session_exists(sid), "重启后会话丢失"
        msgs = sm2.get_messages(sid)
        assert len(msgs) == 2, "重启后消息丢失（应剩 system + user）"
        assert msgs[1]["content"] == "hello"
        assert sm2.list_sessions()[0]["title"] == "Persisted"
        sm2.engine.dispose()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
