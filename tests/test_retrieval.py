"""
Unit tests for the retrieval fusion utilities (tokenize / rrf_fuse).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval import tokenize, rrf_fuse, top_by_scores


class TestTokenize:
    def test_lowercase_and_split(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestRRFFuse:
    def test_consensus_winner_ranks_first(self):
        # doc1 is rank-1 in both rankings -> highest fused score
        fused = rrf_fuse([["doc1", "doc2"], ["doc1", "doc3"]])
        assert fused[0] == "doc1"

    def test_single_ranking_preserves_order(self):
        assert rrf_fuse([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_doc_in_one_ranking_still_included(self):
        fused = rrf_fuse([["a", "b"], ["c", "d"]])
        assert set(fused) == {"a", "b", "c", "d"}

    def test_two_rank1_beats_one_rank1_and_one_rank2(self):
        # x: rank1+rank1 vs y: rank1+rank2 -> x must win
        fused = rrf_fuse([["x", "y"], ["x", "y"]])
        assert fused == ["x", "y"]

    def test_empty_input(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []

    def test_score_formula(self):
        # rank-1 doc with k=60 gets 1/61 per ranking
        fused = rrf_fuse([["only"]], k=60)
        assert fused == ["only"]


class TestTopByScores:
    def test_sorts_by_score_descending(self):
        items = ["a", "b", "c"]
        scores = [0.2, 0.9, 0.5]
        assert top_by_scores(items, scores, k=3) == ["b", "c", "a"]

    def test_truncates_to_k(self):
        items = ["a", "b", "c"]
        scores = [0.1, 0.3, 0.2]
        assert top_by_scores(items, scores, k=2) == ["b", "c"]

    def test_ties_keep_original_order(self):
        items = ["a", "b", "c"]
        scores = [0.5, 0.5, 0.5]
        assert top_by_scores(items, scores, k=3) == ["a", "b", "c"]

    def test_empty_input(self):
        assert top_by_scores([], [], k=3) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
