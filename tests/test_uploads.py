"""
Unit tests for the upload security module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.uploads import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    validate_upload,
    safe_stored_name,
)


class TestValidateUpload:
    """Extension whitelist + size limit enforcement."""

    @pytest.mark.parametrize("filename", ["a.pdf", "notes.txt", "report.DOCX", "d.doc"])
    def test_allowed_extensions_pass(self, filename):
        validate_upload(filename, size=1024)  # should not raise

    def test_disallowed_extension_rejected(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            validate_upload("evil.exe", size=10)

    def test_no_extension_rejected(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            validate_upload("noext", size=10)

    def test_empty_filename_rejected(self):
        with pytest.raises(ValueError, match="No file provided"):
            validate_upload("", size=10)
        with pytest.raises(ValueError, match="No file provided"):
            validate_upload(None, size=10)

    def test_size_limit_enforced(self):
        validate_upload("a.pdf", size=MAX_UPLOAD_BYTES)  # boundary: allowed
        with pytest.raises(ValueError, match="too large"):
            validate_upload("a.pdf", size=MAX_UPLOAD_BYTES + 1)

    def test_whitelist_matches_rag_loaders(self):
        """Whitelist must not drift from rag.py loader support."""
        assert ALLOWED_EXTENSIONS == {".pdf", ".txt", ".docx", ".doc"}


class TestSafeStoredName:
    """Stored filenames must be collision-free and traversal-safe."""

    def test_unique_names_for_same_input(self):
        assert safe_stored_name("a.pdf") != safe_stored_name("a.pdf")

    def test_keeps_original_basename(self):
        assert safe_stored_name("report.pdf").endswith("_report.pdf")

    def test_strips_path_components(self):
        # Path traversal attempt must not escape the upload directory
        stored = safe_stored_name("../../etc/passwd")
        assert "/" not in stored and "\\" not in stored
        assert ".." not in stored.split("_", 1)[1] or stored.endswith("_passwd")
        assert stored.endswith("_passwd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
