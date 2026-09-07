"""
速率限制测试（阶段 3.3-B）：/api/chat-stream 每 IP 限 CHAT_RATE_LIMIT 次/小时。

关键设计：用假流式模型 + 内存会话库 + use_rag=false，
全程零 DeepSeek API 调用、零真实向量检索，只验证限流机制本身。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """隔离被测应用：假模型 + 内存会话库，避免烧 API、避免污染生产 data/app.db"""
    import main
    from core.session import SessionManager

    monkeypatch.setattr(main, "session_manager", SessionManager("sqlite:///:memory:"))

    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    class _FakeModel:
        def stream(self, messages):
            yield _FakeChunk("hello")

    monkeypatch.setattr(main, "model", _FakeModel())

    # 清空限流计数，避免重复运行/其他测试相互干扰（内存存储支持 reset）
    try:
        main.app.state.limiter.reset()
    except Exception:
        pass

    return TestClient(main.app)


@pytest.fixture
def auth_headers():
    """非访客 token：避开访客 5 条配额，纯粋测速率限制（否则第 6 条就撞配额而非第 21 条撞限流）"""
    from core.auth import issue_login_token
    token, _ = issue_login_token("ratelimit@test.com")
    return {"Authorization": f"Bearer {token}"}


def test_chat_stream_rate_limit(client, auth_headers):
    """连发 limit+1 次（带非访客 token）：前 limit 次放行(200)，第 limit+1 次被限流(429)"""
    import main

    # 从配置解析限额，避免把 20 写死（默认 CHAT_RATE_LIMIT="20/hour"）
    limit = int(main.CHAT_RATE_LIMIT.split("/")[0])
    payload = {"message": "hi", "use_rag": False}

    for i in range(limit):
        resp = client.post("/api/chat-stream", json=payload, headers=auth_headers)
        assert resp.status_code == 200, f"第 {i + 1} 次应放行，实际 {resp.status_code}"

    blocked = client.post("/api/chat-stream", json=payload, headers=auth_headers)
    assert blocked.status_code == 429, f"第 {limit + 1} 次应被限流，实际 {blocked.status_code}"
