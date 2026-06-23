"""
Upload Pipeline Tests
======================
Tests for PDF upload validation, processing, status polling, and session scoping.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from app.services.upload import (
    UploadValidationError,
    chunk_text,
    validate_pdf,
)
from app.services.upload_store import UploadJob, UploadStore


# ──────────────────────────────────────────────
# UploadStore tests
# ──────────────────────────────────────────────

class TestUploadStore:
    def test_create_and_get_job(self):
        store = UploadStore()
        job = store.create_job("job-1", "sess-1", "test.pdf")
        assert job.job_id == "job-1"
        assert job.status == "queued"
        assert store.get_job("job-1") is job

    def test_get_nonexistent_job_returns_none(self):
        store = UploadStore()
        assert store.get_job("nonexistent") is None

    def test_session_jobs(self):
        store = UploadStore()
        store.create_job("j1", "sess-A", "a.pdf")
        store.create_job("j2", "sess-B", "b.pdf")
        store.create_job("j3", "sess-A", "c.pdf")

        sess_a = store.get_session_jobs("sess-A")
        assert len(sess_a) == 2
        assert all(j.session_id == "sess-A" for j in sess_a)

    def test_remove_job(self):
        store = UploadStore()
        store.create_job("j1", "s1", "f.pdf")
        removed = store.remove_job("j1")
        assert removed is not None
        assert store.get_job("j1") is None

    def test_active_jobs_count(self):
        store = UploadStore()
        j1 = store.create_job("j1", "s1", "a.pdf")
        j2 = store.create_job("j2", "s1", "b.pdf")
        j1.complete(10)
        assert store.active_jobs == 1


# ──────────────────────────────────────────────
# UploadJob tests
# ──────────────────────────────────────────────

class TestUploadJob:
    def test_complete(self):
        job = UploadJob(job_id="j1", session_id="s1", filename="f.pdf")
        job.complete(42)
        assert job.status == "ready"
        assert job.chunks_created == 42
        assert job.completed_at is not None

    def test_fail(self):
        job = UploadJob(job_id="j1", session_id="s1", filename="f.pdf")
        job.fail("bad pdf", "invalid_file_type")
        assert job.status == "failed"
        assert job.error == "bad pdf"
        assert job.error_code == "invalid_file_type"

    def test_to_dict(self):
        job = UploadJob(job_id="j1", session_id="s1", filename="f.pdf")
        d = job.to_dict()
        assert d["job_id"] == "j1"
        assert "status" in d
        assert "timings" in d


# ──────────────────────────────────────────────
# Chunking tests
# ──────────────────────────────────────────────

class TestChunking:
    def test_basic_chunking(self):
        pages = [{"page": 1, "text": "This is a sentence. This is another sentence. And a third one."}]
        docs, metas, ids = chunk_text(pages, "sess-1", "up-1", "test.pdf", chunk_size=50, chunk_overlap=10)

        assert len(docs) > 0
        assert all(m["source_type"] == "user_upload" for m in metas)
        assert all(m["session_id"] == "sess-1" for m in metas)
        assert all("upload_up-1" in i for i in ids)

    def test_empty_page(self):
        pages = [{"page": 1, "text": ""}]
        docs, metas, ids = chunk_text(pages, "s1", "u1", "f.pdf")
        assert len(docs) == 0

    def test_metadata_schema(self):
        pages = [{"page": 3, "text": "Hello world. This is content."}]
        docs, metas, ids = chunk_text(pages, "sess-1", "up-1", "doc.pdf", chunk_size=500)

        assert len(metas) == 1
        m = metas[0]
        assert m["source_type"] == "user_upload"
        assert m["file_name"] == "doc.pdf"
        assert m["page_number"] == "3"
        assert m["upload_id"] == "up-1"
        assert m["session_id"] == "sess-1"
        assert m["content_type"] == "user_upload"
        assert "uploaded_at" in m

    def test_large_text_chunked(self):
        # Generate text larger than chunk_size
        text = ". ".join([f"Sentence number {i}" for i in range(100)])
        pages = [{"page": 1, "text": text}]
        docs, metas, ids = chunk_text(pages, "s1", "u1", "f.pdf", chunk_size=100, chunk_overlap=20)

        assert len(docs) > 1
        # Each chunk should be roughly within chunk_size
        for doc in docs:
            # Allow some slack for sentence boundaries
            assert len(doc) < 200  # generous upper bound


# ──────────────────────────────────────────────
# Validation tests (using mocks for magic/fitz)
# ──────────────────────────────────────────────

class TestValidation:
    def test_wrong_file_type(self):
        with patch("app.services.upload.magic.from_buffer", return_value="image/png"):
            with pytest.raises(UploadValidationError) as exc:
                validate_pdf(b"fake data", "test.png")
            assert exc.value.error_code == "invalid_file_type"

    def test_file_too_large(self):
        with patch("app.services.upload.magic.from_buffer", return_value="application/pdf"):
            with patch("app.services.upload.settings") as mock_settings:
                mock_settings.max_upload_size_mb = 1
                mock_settings.max_upload_pages = 500
                big_data = b"x" * (2 * 1024 * 1024)  # 2MB
                with pytest.raises(UploadValidationError) as exc:
                    validate_pdf(big_data, "big.pdf")
                assert exc.value.error_code == "file_too_large"

    def test_encrypted_pdf(self):
        mock_doc = MagicMock()
        mock_doc.is_encrypted = True

        with patch("app.services.upload.magic.from_buffer", return_value="application/pdf"):
            with patch("app.services.upload.settings") as mock_settings:
                mock_settings.max_upload_size_mb = 25
                with patch("app.services.upload.fitz.open", return_value=mock_doc):
                    with pytest.raises(UploadValidationError) as exc:
                        validate_pdf(b"fake pdf", "encrypted.pdf")
                    assert exc.value.error_code == "encrypted_pdf"
