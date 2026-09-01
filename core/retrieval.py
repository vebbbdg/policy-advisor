"""
检索融合工具（纯函数，无模型/向量库依赖，可单独单测）
- tokenize:      BM25 用的轻量分词
- rrf_fuse:      Reciprocal Rank Fusion，融合多路检索的排名
- top_by_scores: 按分数降序取 top-k（reranker 打分后用）
"""
import re
from collections import defaultdict
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """小写化并按单词切分，供 BM25 使用"""
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[str]:
    """
    Reciprocal Rank Fusion：融合多路排名结果
    - rankings: 多路检索结果，每路是按相关性降序的唯一键列表
    - k: 平滑常数（RRF 论文默认 60，抑制头部排名的过度影响）
    返回按融合分数降序的键列表。
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] += 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def top_by_scores(items: List, scores: List[float], k: int) -> List:
    """
    按分数降序取前 k 个 item（reranker 对 (query, passage) 打分后精排用）。
    同分时保持原顺序（稳定排序），避免候选集顺序带来的随机性。
    """
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    return [items[i] for i in order[:k]]
