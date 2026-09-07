"""
RAG 引擎集成测试：索引 → 计数 → 清空 全链路
使用隔离的临时向量库，不触碰生产 data/vectordb。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.rag import RAGEngine


def _release(engine):
    """释放 Chroma 客户端资源（Windows 上避免句柄占用导致清理失败）"""
    try:
        client = engine.vector_store._client
        if hasattr(client, "clear_system_cache"):
            client.clear_system_cache()
        else:
            client._system.stop()
    except Exception:
        pass


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("vectordb")
    eng = RAGEngine(persist_directory=str(tmp_dir))
    yield eng
    _release(eng)


def test_add_count_and_clear(engine, tmp_path):
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "Vector databases store embeddings for similarity search. "
        "Chroma is an open-source vector database.",
        encoding="utf-8",
    )

    chunk_count = engine.add_document(str(doc), "sample.txt")
    assert chunk_count >= 1
    assert engine.get_document_count() == chunk_count

    # 回归测试：新版 Chroma 不接受空 where，清空必须按 id 删除
    engine.clear_all()
    assert engine.get_document_count() == 0

    # 空库重复清空不应报错
    engine.clear_all()
    assert engine.get_document_count() == 0


def test_retrieve_by_mode_routes(engine, monkeypatch):
    """阶段 3.3-A：RETRIEVER_MODE 决定分派到哪种检索；默认 hybrid，未知值降级 hybrid"""
    calls = []
    monkeypatch.setattr(engine, "retrieve", lambda q, top_k=3: calls.append("dense") or [])
    monkeypatch.setattr(engine, "retrieve_hybrid", lambda q, top_k=3, **kw: calls.append("hybrid") or [])
    monkeypatch.setattr(engine, "retrieve_reranked", lambda q, top_k=3, **kw: calls.append("rerank") or [])

    # 未设环境变量 → 默认 hybrid
    monkeypatch.delenv("RETRIEVER_MODE", raising=False)
    engine.retrieve_by_mode("q")
    assert calls[-1] == "hybrid"

    # 三种模式（含大小写与首尾空格）正确路由
    for mode, expected in [("dense", "dense"), ("hybrid", "hybrid"),
                           ("rerank", "rerank"), ("DENSE", "dense"), (" Hybrid ", "hybrid")]:
        monkeypatch.setenv("RETRIEVER_MODE", mode)
        engine.retrieve_by_mode("q")
        assert calls[-1] == expected, f"RETRIEVER_MODE={mode!r} 应路由到 {expected}"

    # 未知值 → 降级 hybrid（不因配置笔误中断）
    monkeypatch.setenv("RETRIEVER_MODE", "bogus")
    engine.retrieve_by_mode("q")
    assert calls[-1] == "hybrid"
