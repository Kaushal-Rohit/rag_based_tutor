"""
Ingest Endpoint
===============
``POST /api/v1/ingest`` — trigger re-indexing of documents.

Runs the full indexing pipeline (load documents → encode → build indices)
as a background task since it's long-running.
"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Request

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import IngestRequest, IngestResponse

logger = get_logger(__name__)
router = APIRouter()


def _run_ingestion(
    dataset_dir: str,
    faiss_output: str,
    chroma_path: str,
    embedding_model,
) -> None:
    """
    Synchronous ingestion task — runs in a background thread.

    Loads documents, encodes embeddings, and rebuilds both indices.
    """
    from app.services.indexer import (
        build_chroma_index,
        build_embeddings,
        build_faiss_index,
        load_documents,
    )

    logger.info(
        f"Ingestion started: dataset_dir={dataset_dir}",
        extra={"stream": "app"},
    )

    documents, metadatas, ids = load_documents(dataset_dir)
    embeddings = build_embeddings(documents, model=embedding_model)
    build_faiss_index(embeddings, faiss_output)
    build_chroma_index(documents, embeddings, metadatas, ids, chroma_path)

    logger.info(
        f"Ingestion complete: {len(documents)} documents, "
        f"{embeddings.shape[0]} vectors",
        extra={"stream": "app"},
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger re-indexing",
    description="Re-runs the indexing pipeline. Runs in the background; "
                "returns immediately with status.",
)
async def ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Trigger document re-indexing as a background task."""
    dataset_dir = body.dataset_dir or settings.dataset_dir
    faiss_output = settings.faiss_index_path
    chroma_path = settings.chroma_db_path

    logger.info(
        f"Ingest request received: dataset_dir={dataset_dir}, rebuild={body.rebuild}",
        extra={"stream": "access"},
    )

    # Count documents first (fast) to return an estimate
    from app.services.indexer import load_documents

    documents, _, _ = load_documents(dataset_dir)
    doc_count = len(documents)

    # Schedule actual indexing in background
    background_tasks.add_task(
        _run_ingestion,
        dataset_dir,
        faiss_output,
        chroma_path,
        request.app.state.embedding_model,
    )

    return IngestResponse(
        status="ingestion_started",
        documents_loaded=doc_count,
        vectors_indexed=0,  # will be populated when background task completes
    )
