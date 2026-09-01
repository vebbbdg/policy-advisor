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
