"""
检索评估指标（纯函数，无模型依赖，可单独单测）
- chunk_matches: 归一化后判断 chunk 是否包含任一 ground-truth 片段
- recall_at_k:   top-k 中是否命中（单条 ground-truth 场景下为 0/1）
- mrr:           首次命中排名的倒数，未命中为 0
- precision_at_k: top-k 中命中占比（重叠切片可能多次命中）

snippet 支持单个字符串或字符串列表：多源政策语料中，同一问题常有多个
语义等价的正确答案片段（如 USCIS 主页与政策手册各有一句），任一命中即算召回。
"""
from typing import List, Union

# ground-truth 片段：单句或多句（任一命中即算）
Snippet = Union[str, List[str]]


def _normalize(text: str) -> str:
    """统一小写并压缩空白，避免换行/多余空格造成误判"""
    return " ".join(text.lower().split())


def chunk_matches(chunk_text: str, snippet: Snippet) -> bool:
    """chunk 命中定义：归一化后的 chunk 文本包含任一 ground-truth 片段"""
    candidates = [snippet] if isinstance(snippet, str) else snippet
    normalized_chunk = _normalize(chunk_text)
    return any(_normalize(s) in normalized_chunk for s in candidates)


def recall_at_k(retrieved: List[str], snippet: Snippet, k: int) -> float:
    """top-k 召回：命中返回 1.0，否则 0.0"""
    top_k = retrieved[:k]
    return 1.0 if any(chunk_matches(c, snippet) for c in top_k) else 0.0


def mrr(retrieved: List[str], snippet: Snippet) -> float:
    """平均倒数排名（单条 ground-truth 时即该查询的 RR）"""
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk_matches(chunk, snippet):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[str], snippet: Snippet, k: int) -> float:
    """top-k 精确率：命中数 / k"""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for c in top_k if chunk_matches(c, snippet))
    return hits / len(top_k)
