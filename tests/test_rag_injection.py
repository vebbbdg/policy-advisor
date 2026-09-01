"""
Regression tests for RAG context injection.
Verifies that injecting retrieved context into the system prompt
does NOT mutate the original system message stored in the session,
which previously caused the system prompt to grow unboundedly
across multiple RAG turns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory import keep_recent_messages, inject_rag_prompt
from core.session import SessionManager, SYSTEM_PROMPT


RAG_PROMPT = (
    "Use the following retrieved context to answer the user's question. "
    "\n\nRetrieved context:\n[Document 1 from notes.txt] some fake context"
)


class TestRAGPromptInjection:
    """Regression suite: system prompt must stay constant across RAG turns."""

    def test_multi_turn_rag_keeps_system_prompt_constant(self):
        """Simulate 5 RAG turns; stored system prompt length must never change."""
        sm = SessionManager()
        sid = sm.create_session()
        baseline_len = len(SYSTEM_PROMPT)

        for i in range(5):
            # 1. User message lands in session storage
            sm.add_message(sid, "user", f"question {i}")

            # 2. Build model input: fetch -> trim -> inject RAG context
            messages = sm.get_messages(sid)
            optimized = keep_recent_messages(messages, max_pairs=15)
            optimized = inject_rag_prompt(optimized, RAG_PROMPT)

            # The messages sent to the model DO contain the RAG context
            assert RAG_PROMPT in optimized[0]["content"]

            # 3. Assistant reply lands in session storage
            sm.add_message(sid, "assistant", f"answer {i}")

            # The stored system message must be untouched after every turn
            stored_system = sm.get_messages(sid)[0]
            assert stored_system["content"] == SYSTEM_PROMPT
            assert len(stored_system["content"]) == baseline_len

    def test_injection_builds_new_system_dict(self):
        """Injected system message must be a new dict, not the original object."""
        original = {"role": "system", "content": "base prompt"}
        msgs = [original, {"role": "user", "content": "q"}]

        result = inject_rag_prompt(msgs, RAG_PROMPT)

        assert result[0] is not original
        assert original["content"] == "base prompt"  # original untouched
        assert result[0]["content"] == "base prompt\n\n" + RAG_PROMPT

    def test_injection_without_system_message(self):
        """No system message -> list content unchanged."""
        msgs = [{"role": "user", "content": "hi"}]
        result = inject_rag_prompt(msgs, RAG_PROMPT)
        assert result == msgs


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
