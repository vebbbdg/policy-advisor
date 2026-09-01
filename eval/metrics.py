"""
检索评估指标（纯函数，无模型依赖，可单独单测）
- chunk_matches: 归一化后判断 chunk 是否包含 ground-truth 片段
- recall_at_k:   top-k 中是否命中（单条 ground-truth 场景下为 0/1）
- mrr:           首次命中排名的倒数，未命中为 0
- precision_at_k: top-k 中命中占比（重叠切片可能多次命中）
"""
from typing import List


def _normalize(text: str) -> str:
    """统一小写并压缩空白，避免换行/多余空格造成误判"""
    return " ".join(text.lower().split())


def chunk_matches(chunk_text: str, snippet: str) -> bool:
    """chunk 命中定义：归一化后的 chunk 文本包含 ground-truth 片段"""
    return _normalize(snippet) in _normalize(chunk_text)


def recall_at_k(retrieved: List[str], snippet: str, k: int) -> float:
    """top-k 召回：命中返回 1.0，否则 0.0"""
    top_k = retrieved[:k]
    return 1.0 if any(chunk_matches(c, snippet) for c in top_k) else 0.0


def mrr(retrieved: List[str], snippet: str) -> float:
    """平均倒数排名（单条 ground-truth 时即该查询的 RR）"""
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk_matches(chunk, snippet):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[str], snippet: str, k: int) -> float:
    """top-k 精确率：命中数 / k"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if chunk_matches(c, snippet))
    return hits / len(top_k)
