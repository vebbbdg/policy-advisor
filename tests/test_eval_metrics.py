"""
Unit tests for RAG retrieval evaluation metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import chunk_matches, recall_at_k, mrr, precision_at_k


SNIPPET = "The most common similarity metrics are cosine similarity and Euclidean distance."


class TestChunkMatches:
    def test_exact_contains(self):
        assert chunk_matches("prefix " + SNIPPET + " suffix", SNIPPET)

    def test_case_insensitive(self):
        assert chunk_matches(SNIPPET.upper(), SNIPPET)

    def test_whitespace_normalized(self):
        # newlines / extra spaces must not break matching
        mangled = SNIPPET.replace(" ", "  \n ")
        assert chunk_matches(mangled, SNIPPET)

    def test_no_match(self):
        assert not chunk_matches("completely unrelated text", SNIPPET)


class TestRecallAtK:
    def test_hit_within_top_k(self):
        retrieved = ["unrelated", SNIPPET, "other"]
        assert recall_at_k(retrieved, SNIPPET, k=3) == 1.0

    def test_hit_outside_top_k(self):
        retrieved = ["a", "b", "c", SNIPPET]
        assert recall_at_k(retrieved, SNIPPET, k=3) == 0.0
        assert recall_at_k(retrieved, SNIPPET, k=4) == 1.0

    def test_empty_retrieval(self):
        assert recall_at_k([], SNIPPET, k=3) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr([SNIPPET, "x"], SNIPPET) == 1.0

    def test_third_position(self):
        assert mrr(["a", "b", SNIPPET], SNIPPET) == 1 / 3

    def test_miss_returns_zero(self):
        assert mrr(["a", "b"], SNIPPET) == 0.0


class TestPrecisionAtK:
    def test_single_hit_in_top3(self):
        assert precision_at_k(["a", SNIPPET, "b"], SNIPPET, k=3) == 1 / 3

    def test_multiple_hits_via_overlapping_chunks(self):
        assert precision_at_k([SNIPPET, SNIPPET, "b"], SNIPPET, k=3) == 2 / 3

    def test_empty_returns_zero(self):
        assert precision_at_k([], SNIPPET, k=3) == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
