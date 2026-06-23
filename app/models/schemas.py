"""
Pydantic Schemas
================
Request/response models for all API endpoints.
Provides explicit field validation and OpenAPI documentation.
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.models.enums import ClassLevel, ContentType, Subject, UserState


# ──────────────────────────────────────────────
# Shared / nested models
# ──────────────────────────────────────────────

class MetadataFilter(BaseModel):
    """Optional metadata filters for retrieval narrowing."""
    subject: Optional[Subject] = Field(None, description="Filter by subject")
    class_level: Optional[ClassLevel] = Field(
        None, alias="class", description="Filter by class level (9, 11, 12)"
    )
    content_type: Optional[ContentType] = Field(
        None, description="Filter by content type"
    )
    chapter_name: Optional[str] = Field(
        None, description="Filter by chapter name (exact match)"
    )

    model_config = {"populate_by_name": True}

    def to_filter_dict(self) -> dict:
        """Convert to a flat {field: value} dict for retriever consumption."""
        result = {}
        if self.subject is not None:
            result["subject"] = self.subject.value
        if self.class_level is not None:
            result["class"] = self.class_level.value
        if self.content_type is not None:
            result["content_type"] = self.content_type.value
        if self.chapter_name is not None:
            result["chapter_name"] = self.chapter_name
        return result


# ──────────────────────────────────────────────
# POST /api/v1/query
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the main RAG query endpoint."""
    query: str = Field(
        ..., min_length=1, max_length=2000,
        description="The user's question",
    )
    filters: Optional[MetadataFilter] = Field(
        None, description="Optional metadata filters to narrow retrieval"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity. "
                          "If omitted, a new session is created."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is Newton's second law of motion?",
                    "filters": {"subject": "Physics", "class": "12"},
                    "session_id": None,
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Non-streaming response for the RAG query endpoint."""
    answer: str
    session_id: str
    sentiment_score: float = Field(
        description="Rolling average sentiment polarity used for this query"
    )
    user_state: UserState = Field(
        description="Derived user comprehension state"
    )
    k_used: int = Field(description="Dynamic k value used for retrieval")
    chunks_retrieved: int = Field(
        description="Number of chunks returned from retrieval"
    )
    retrieval_latency_ms: float = Field(
        description="Total retrieval latency in milliseconds"
    )
    crag_applied: bool = Field(
        description="Whether CRAG pipeline was applied"
    )
    request_id: str = Field(
        description="Unique request ID for log correlation"
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="Source citations: [{filename, page, chunk_id, source_type}]"
    )


# ──────────────────────────────────────────────
# POST /api/v1/ingest
# ──────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request body for triggering re-indexing."""
    dataset_dir: Optional[str] = Field(
        None,
        description="Path to dataset directory. Defaults to configured path.",
    )
    rebuild: bool = Field(
        False,
        description="If true, delete existing indices and rebuild from scratch.",
    )


class IngestResponse(BaseModel):
    """Response from the ingest endpoint."""
    status: str = Field(description="Current ingestion status")
    documents_loaded: int = Field(
        description="Total documents loaded from all sources"
    )
    vectors_indexed: int = Field(
        description="Total vectors written to the index"
    )


# ──────────────────────────────────────────────
# GET /api/v1/health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str = Field(description="Overall health status: healthy | degraded | unhealthy")
    index_loaded: bool = Field(description="Whether vector indices are loaded in memory")
    ollama_reachable: bool = Field(description="Whether the Ollama server responds")
    ollama_model_available: bool = Field(
        description="Whether the configured LLM model is available in Ollama"
    )
    vector_count: int = Field(description="Number of vectors in the loaded index")
    embedding_model_loaded: bool = Field(
        description="Whether the embedding model is loaded"
    )


# ──────────────────────────────────────────────
# GET /api/v1/metrics
# ──────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """Response from the metrics endpoint."""
    total_queries: int = Field(description="Total queries served since startup")
    active_sessions: int = Field(description="Number of active conversation sessions")
    recent_retrieval_latencies_ms: list[float] = Field(
        description="Last N retrieval latencies in milliseconds"
    )
    average_retrieval_latency_ms: Optional[float] = Field(
        None, description="Mean retrieval latency across recent queries"
    )
    sentiment_distribution: dict[str, int] = Field(
        description="Count of queries by user state: {confused, neutral, clear}"
    )


# ──────────────────────────────────────────────
# POST /api/v1/upload
# ──────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response from the upload endpoint (202 Accepted)."""
    job_id: str = Field(description="Unique job ID for status polling")
    status: str = Field(description="Initial status (queued)")
    filename: str = Field(description="Original uploaded filename")


class UploadStatusResponse(BaseModel):
    """Response from the upload status endpoint."""
    job_id: str
    session_id: str
    filename: str
    status: str = Field(
        description="Processing stage: queued|extracting|chunking|embedding|indexing|ready|failed"
    )
    error: Optional[str] = None
    error_code: Optional[str] = None
    chunks_created: int = 0
    pages_processed: int = 0
    total_pages: int = 0
    timings: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = ""
    completed_at: Optional[str] = None


# ──────────────────────────────────────────────
# Structured error response (all endpoints)
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Consistent error schema returned by all endpoints on failure."""
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    request_id: str = Field(description="Request ID for log correlation")


# ──────────────────────────────────────────────
# Source citation (for notebook UI)
# ──────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A single source citation for a retrieved chunk."""
    filename: Optional[str] = Field(None, description="Source filename")
    page: Optional[str] = Field(None, description="Page number (if from upload)")
    chunk_id: str = Field(description="Chunk ID")
    source_type: str = Field(
        default="static", description="static or user_upload"
    )

