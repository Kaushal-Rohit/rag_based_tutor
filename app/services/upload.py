"""
Upload Processing Service
===========================
Handles the full PDF → chunks → indexed pipeline:

  1. Validate PDF (magic-byte, size, encryption, page count)
  2. Extract text per page (PyMuPDF)
  3. Chunk text (sentence-boundary-aware splitting)
  4. Embed chunks (reuse loaded SentenceTransformer)
  5. Index incrementally into FAISS + ChromaDB (no full reindex)

Each stage updates the UploadJob status and logs timing through the
existing structured logging system.
"""

import os
import re
import time
import uuid
from datetime import datetime, timezone

import fitz  # PyMuPDF
import magic
import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.upload_store import UploadJob

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 1. Validation
# ──────────────────────────────────────────────

class UploadValidationError(Exception):
    """Raised when PDF validation fails."""

    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code


def validate_pdf(file_bytes: bytes, filename: str) -> fitz.Document:
    """
    Validate an uploaded file before processing.

    Checks:
      - Magic-byte file type (must be application/pdf)
      - File size (must be under configured cap)
      - Not encrypted / password-protected
      - Page count under configured cap

    Returns:
        An open PyMuPDF Document on success.

    Raises:
        UploadValidationError with a specific error_code on any failure.
    """
    # Magic-byte check
    mime = magic.from_buffer(file_bytes[:2048], mime=True)
    if mime != "application/pdf":
        raise UploadValidationError(
            f"Invalid file type: expected application/pdf, got {mime}. "
            f"Rename tricks won't work — we check the actual file bytes.",
            error_code="invalid_file_type",
        )

    # Size check
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise UploadValidationError(
            f"File too large: {size_mb:.1f} MB exceeds the "
            f"{settings.max_upload_size_mb} MB limit.",
            error_code="file_too_large",
        )

    # Open with PyMuPDF
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise UploadValidationError(
            f"Cannot read PDF: {e}",
            error_code="unreadable_pdf",
        )

    # Encrypted check
    if doc.is_encrypted:
        doc.close()
        raise UploadValidationError(
            "This PDF is password-protected. Please remove the password "
            "and re-upload, or upload an unencrypted version.",
            error_code="encrypted_pdf",
        )

    # Page count check
    if doc.page_count > settings.max_upload_pages:
        doc.close()
        raise UploadValidationError(
            f"PDF has {doc.page_count} pages, which exceeds the "
            f"{settings.max_upload_pages} page limit.",
            error_code="too_many_pages",
        )

    return doc


# ──────────────────────────────────────────────
# 2. Text extraction
# ──────────────────────────────────────────────

def extract_text(
    doc: fitz.Document, upload_id: str
) -> tuple[list[dict], list[str]]:
    """
    Extract text from each page of the PDF.

    Returns:
        tuple of (page_texts, warnings)
        page_texts: list of {"page": int, "text": str}
        warnings: list of warning messages for pages that failed
    """
    page_texts = []
    warnings = []

    for page_num in range(doc.page_count):
        try:
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                page_texts.append({"page": page_num + 1, "text": text})
        except Exception as e:
            warning = f"Page {page_num + 1}: extraction failed — {e}"
            warnings.append(warning)
            logger.warning(
                f"[upload:{upload_id}] {warning}",
                extra={"stream": "ingestion"},
            )

    return page_texts, warnings


# ──────────────────────────────────────────────
# 3. Chunking
# ──────────────────────────────────────────────

# Sentence boundary pattern: split on . ! ? followed by whitespace or end
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def chunk_text(
    page_texts: list[dict],
    session_id: str,
    upload_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[list[str], list[dict], list[str]]:
    """
    Split extracted page texts into chunks using sentence-boundary-aware splitting.

    Uses the same metadata schema as the static NCERT corpus, plus upload-specific
    fields (source_type, file_name, page_number, upload_id, session_id).

    Returns:
        tuple of (documents, metadatas, ids)
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    documents = []
    metadatas = []
    ids = []
    uploaded_at = datetime.now(timezone.utc).isoformat()

    for page_info in page_texts:
        page_num = page_info["page"]
        text = page_info["text"]

        # Split into sentences
        sentences = _SENTENCE_SPLIT.split(text)

        current_chunk = ""
        chunk_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk_size, finalize current chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                _add_chunk(
                    documents, metadatas, ids,
                    current_chunk, page_num, chunk_idx,
                    session_id, upload_id, filename, uploaded_at,
                )
                chunk_idx += 1

                # Overlap: keep the tail of the current chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:]
                else:
                    current_chunk = ""

            # Append sentence
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

        # Finalize remaining text
        if current_chunk.strip():
            _add_chunk(
                documents, metadatas, ids,
                current_chunk.strip(), page_num, chunk_idx,
                session_id, upload_id, filename, uploaded_at,
            )

    return documents, metadatas, ids


def _add_chunk(
    documents, metadatas, ids,
    text, page_num, chunk_idx,
    session_id, upload_id, filename, uploaded_at,
):
    """Helper to append a single chunk with full metadata."""
    chunk_id = f"upload_{upload_id}_p{page_num}_{chunk_idx}"
    documents.append(text)
    metadatas.append({
        "chunk_id": chunk_id,
        "source_type": "user_upload",
        "file_name": filename,
        "page_number": str(page_num),
        "upload_id": upload_id,
        "session_id": session_id,
        "uploaded_at": uploaded_at,
        "subject": "",
        "class": "",
        "content_type": "user_upload",
    })
    ids.append(chunk_id)


# ──────────────────────────────────────────────
# 4. Indexing (incremental)
# ──────────────────────────────────────────────

def index_chunks(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
    embedding_model,
    faiss_index,
    chroma_collection,
    app_state,
) -> float:
    """
    Embed and index chunks incrementally into FAISS and ChromaDB.

    Appends to app.state.documents/metadatas/doc_ids in-place.
    Does NOT trigger a full reindex.

    Returns:
        Total indexing time in milliseconds.
    """
    t_start = time.perf_counter()

    # Embed
    embeddings = embedding_model.encode(documents, show_progress_bar=False)

    # FAISS incremental add
    if faiss_index is not None:
        faiss_index.add(np.array(embeddings, dtype=np.float32))

    # ChromaDB incremental add
    if chroma_collection is not None:
        chroma_collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )

    # Update in-memory document lists
    app_state.documents.extend(documents)
    app_state.metadatas.extend(metadatas)
    app_state.doc_ids.extend(ids)

    latency_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"Indexed {len(documents)} chunks incrementally ({latency_ms:.0f}ms)",
        extra={
            "stream": "ingestion",
            "context": {
                "chunks_indexed": len(documents),
                "latency_ms": latency_ms,
                "faiss_total": faiss_index.ntotal if faiss_index else 0,
                "chroma_total": chroma_collection.count() if chroma_collection else 0,
            },
        },
    )

    return latency_ms


# ──────────────────────────────────────────────
# 5. Full pipeline orchestrator
# ──────────────────────────────────────────────

def process_upload(
    job: UploadJob,
    file_bytes: bytes,
    app_state,
) -> None:
    """
    Run the full upload processing pipeline as a background task.

    Updates job.status at each stage and logs per-stage timing.
    On any failure, marks job as failed with a specific error.
    """
    upload_id = job.job_id

    try:
        # ── Stage 1: Validate ──
        t0 = time.perf_counter()
        doc = validate_pdf(file_bytes, job.filename)
        job.total_pages = doc.page_count
        job.timings["validate_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # ── Stage 2: Extract ──
        job.status = "extracting"
        t0 = time.perf_counter()
        page_texts, warnings = extract_text(doc, upload_id)
        doc.close()
        job.pages_processed = len(page_texts)
        job.warnings.extend(warnings)
        job.timings["extract_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if not page_texts:
            raise UploadValidationError(
                "No extractable text found — this looks like a scanned document; "
                "OCR isn't enabled yet.",
                error_code="no_extractable_text",
            )

        total_chars = sum(len(p["text"]) for p in page_texts)
        logger.info(
            f"[upload:{upload_id}] Extracted {len(page_texts)} pages, "
            f"{total_chars} chars ({job.timings['extract_ms']:.0f}ms)",
            extra={"stream": "ingestion"},
        )

        # ── Stage 3: Chunk ──
        job.status = "chunking"
        t0 = time.perf_counter()
        documents, metadatas, ids = chunk_text(
            page_texts, job.session_id, upload_id, job.filename,
        )
        job.timings["chunk_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        logger.info(
            f"[upload:{upload_id}] Created {len(documents)} chunks "
            f"({job.timings['chunk_ms']:.0f}ms)",
            extra={"stream": "ingestion"},
        )

        # ── Stage 4: Embed + Index ──
        job.status = "embedding"
        t0 = time.perf_counter()
        embed_and_index_ms = index_chunks(
            documents, metadatas, ids,
            app_state.embedding_model,
            app_state.faiss_index,
            app_state.chroma_collection,
            app_state,
        )
        job.timings["embed_index_ms"] = round(embed_and_index_ms, 1)

        # ── Stage 5: Save source PDF (optional, for reference) ──
        job.status = "indexing"
        os.makedirs(settings.upload_dir, exist_ok=True)
        pdf_path = os.path.join(settings.upload_dir, f"{upload_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        job.timings["save_ms"] = round((time.perf_counter() - t0) * 1000 - embed_and_index_ms, 1)

        # ── Done ──
        job.complete(chunks=len(documents))

        total_ms = sum(job.timings.values())
        logger.info(
            f"[upload:{upload_id}] Processing complete: {len(documents)} chunks "
            f"from {len(page_texts)} pages in {total_ms:.0f}ms total",
            extra={
                "stream": "ingestion",
                "context": {
                    "upload_id": upload_id,
                    "filename": job.filename,
                    "pages": len(page_texts),
                    "chunks": len(documents),
                    "timings": job.timings,
                },
            },
        )

    except UploadValidationError as e:
        job.fail(str(e), e.error_code)
        logger.error(
            f"[upload:{upload_id}] Validation failed: {e}",
            extra={"stream": "error"},
        )

    except Exception as e:
        job.fail(f"Unexpected error: {e}", "processing_error")
        logger.error(
            f"[upload:{upload_id}] Processing failed: {e}",
            extra={"stream": "error"},
            exc_info=True,
        )
