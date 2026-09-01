"""
Unit tests for the sliding window memory module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory import keep_recent_messages


class TestSlidingWindowMemory:
    """Test suite for keep_recent_messages function."""

    def test_empty_list_returns_empty(self):
        assert keep_recent_messages([]) == []

    def test_system_prompt_preserved(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = keep_recent_messages(msgs, max_pairs=10)
        assert result[0]["role"] == "system"
        assert len(result) == 3

    def test_truncation_keeps_recent(self):
        # 1 system + 15 pairs (30 messages) = 31 total
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(15):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        result = keep_recent_messages(msgs, max_pairs=10)
        # system + 10 pairs = 21 messages
        assert len(result) == 21
        assert result[0]["role"] == "system"
        # Should contain q5 through q14 (most recent 10 pairs)
        assert result[-2]["content"] == "q14"
        assert result[1]["content"] == "q5"

    def test_no_system_prompt(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = keep_recent_messages(msgs, max_pairs=10)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_exactly_max_pairs(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        result = keep_recent_messages(msgs, max_pairs=10)
        assert len(result) == 21  # 1 + 20

    def test_custom_max_pairs(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        result = keep_recent_messages(msgs, max_pairs=3)
        assert len(result) == 7  # 1 + 6
        assert result[-2]["content"] == "q9"
        assert result[1]["content"] == "q7"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
