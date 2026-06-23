"""
Health Check Endpoint
=====================
``GET /api/v1/health`` — liveness/readiness check.

Reports on:
  - Index loaded status (FAISS and/or ChromaDB)
  - Ollama server reachability
  - Configured model availability
  - Vector count
"""

from fastapi import APIRouter, Request

from app.core.logging_config import get_logger
from app.models.schemas import HealthResponse

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="Reports readiness of all subsystems: vector indices, embedding model, Ollama LLM.",
)
async def health_check(request: Request) -> HealthResponse:
    """Check liveness and readiness of all subsystems."""
    app = request.app

    # Index status
    index_loaded = False
    vector_count = 0
    if app.state.faiss_index is not None:
        index_loaded = True
        vector_count = app.state.faiss_index.ntotal
    if app.state.chroma_collection is not None:
        index_loaded = True
        if vector_count == 0:
            vector_count = app.state.chroma_collection.count()

    # Embedding model
    embedding_loaded = app.state.embedding_model is not None

    # Ollama checks
    llm = app.state.llm_engine
    ollama_reachable = await llm.check_connection()
    ollama_model_available = False
    if ollama_reachable:
        ollama_model_available = await llm.check_model_available()

    # Overall status
    if index_loaded and ollama_reachable and ollama_model_available and embedding_loaded:
        status = "healthy"
    elif index_loaded and embedding_loaded:
        status = "degraded"  # can serve cached/retrieval-only, but no LLM
    else:
        status = "unhealthy"

    logger.info(
        f"Health check: status={status}",
        extra={
            "stream": "access",
            "context": {
                "status": status,
                "index_loaded": index_loaded,
                "ollama_reachable": ollama_reachable,
                "model_available": ollama_model_available,
            },
        },
    )

    return HealthResponse(
        status=status,
        index_loaded=index_loaded,
        ollama_reachable=ollama_reachable,
        ollama_model_available=ollama_model_available,
        vector_count=vector_count,
        embedding_model_loaded=embedding_loaded,
    )
