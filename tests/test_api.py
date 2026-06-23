"""
Integration Tests: API Endpoints
==================================
Tests the API endpoints using FastAPI's TestClient.

Note: These tests mock the heavy dependencies (Ollama, FAISS, ChromaDB)
to run without requiring the full infrastructure.
"""

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_app_state():
    """Create mock app state for all required dependencies."""
    state = MagicMock()

    # Mock embedding model
    state.embedding_model = MagicMock()

    # Mock FAISS index
    state.faiss_index = MagicMock()
    state.faiss_index.ntotal = 13917

    # Mock ChromaDB collection
    state.chroma_collection = MagicMock()
    state.chroma_collection.count.return_value = 13917

    # Mock retriever
    state.retriever = MagicMock()
    state.retriever.search.return_value = {
        "documents": ["Chunk 1 text", "Chunk 2 text"],
        "metadatas": [{"subject": "Physics"}, {"subject": "Physics"}],
        "ids": ["id1", "id2"],
        "distances": [0.1, 0.2],
        "latency": {"embedding_ms": 5.0, "search_ms": 1.0, "total_ms": 6.0},
    }

    # Mock LLM engine — all methods must be AsyncMock since they're awaited
    llm_mock = MagicMock()
    llm_mock.check_connection = AsyncMock(return_value=True)
    llm_mock.check_model_available = AsyncMock(return_value=True)
    llm_mock.generate = AsyncMock(return_value="This is a test answer.")
    llm_mock.close = AsyncMock()
    state.llm_engine = llm_mock

    # Mock CRAG — needs AsyncMock for async methods
    crag_mock = MagicMock()
    crag_mock.enabled = False
    crag_mock.rewrite_query = AsyncMock(return_value=("rewritten query", 10.0))
    crag_mock.grade_chunks = AsyncMock(return_value=(["chunk1"], 0, 5.0))
    crag_mock.run = AsyncMock()
    state.crag = crag_mock

    # Mock session manager
    from app.services.sentiment import ConversationManager, SessionManager

    state.session_manager = SessionManager(max_history=5)

    # Mock k selector
    from app.services.sentiment import DynamicKSelector

    state.k_selector = DynamicKSelector(base_k=5)

    # Mock metrics
    state.metrics = {
        "total_queries": 0,
        "retrieval_latencies": deque(maxlen=100),
        "sentiment_distribution": {"confused": 0, "neutral": 0, "clear": 0},
    }

    # Mock documents
    state.documents = ["doc1", "doc2"]
    state.metadatas = [{"subject": "Physics"}, {"subject": "Chemistry"}]
    state.doc_ids = ["id1", "id2"]
    state.chroma_client = MagicMock()

    # Mock upload store
    from app.services.upload_store import UploadStore
    state.upload_store = UploadStore()

    return state


@pytest.fixture
def client(mock_app_state):
    """Create a test client with mocked dependencies.

    Replaces the app's lifespan with a no-op so the real startup
    (model loading, Ollama connection, etc.) is skipped entirely.
    The mock state is injected after the no-op lifespan yields.
    """
    from contextlib import asynccontextmanager

    from app.main import app

    # Replace the real lifespan with a no-op that injects mock state
    @asynccontextmanager
    async def _test_lifespan(a):
        # Inject all mock state attributes onto app.state
        for attr in dir(mock_app_state):
            if not attr.startswith("_"):
                try:
                    setattr(a.state, attr, getattr(mock_app_state, attr))
                except Exception:
                    pass
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _test_lifespan

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Restore original lifespan so subsequent imports aren't affected
    app.router.lifespan_context = original_lifespan


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "index_loaded" in data
        assert "ollama_reachable" in data
        assert "ollama_model_available" in data
        assert "vector_count" in data
        assert "embedding_model_loaded" in data

    def test_health_reports_healthy(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["index_loaded"] is True
        assert data["vector_count"] == 13917


class TestMetricsEndpoint:
    """Tests for GET /api/v1/metrics."""

    def test_metrics_returns_200(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_metrics_response_schema(self, client):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert "total_queries" in data
        assert "active_sessions" in data
        assert "recent_retrieval_latencies_ms" in data
        assert "sentiment_distribution" in data

    def test_metrics_initial_values(self, client):
        response = client.get("/api/v1/metrics")
        data = response.json()
        assert data["total_queries"] == 0
        assert data["active_sessions"] == 0


class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    def test_query_returns_200(self, client):
        response = client.post(
            "/api/v1/query",
            json={"query": "What is Newton's second law?"},
        )
        assert response.status_code == 200

    def test_query_response_schema(self, client):
        response = client.post(
            "/api/v1/query",
            json={"query": "What is Newton's second law?"},
        )
        data = response.json()
        assert "answer" in data
        assert "session_id" in data
        assert "sentiment_score" in data
        assert "user_state" in data
        assert "k_used" in data
        assert "chunks_retrieved" in data
        assert "request_id" in data

    def test_query_with_filters(self, client):
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is electromagnetic induction?",
                "filters": {"subject": "Physics", "class": "12"},
            },
        )
        assert response.status_code == 200

    def test_query_invalid_subject_returns_422(self, client):
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query",
                "filters": {"subject": "InvalidSubject"},
            },
        )
        assert response.status_code == 422

    def test_query_empty_body_returns_422(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_increments_metrics(self, client, mock_app_state):
        client.post(
            "/api/v1/query",
            json={"query": "What is gravity?"},
        )
        assert mock_app_state.metrics["total_queries"] == 1

    def test_query_with_session_continuity(self, client):
        # First query creates a session
        r1 = client.post(
            "/api/v1/query",
            json={"query": "What is F=ma?", "session_id": "test-session"},
        )
        sid = r1.json()["session_id"]
        assert sid == "test-session"

        # Second query reuses the session
        r2 = client.post(
            "/api/v1/query",
            json={"query": "Explain more", "session_id": "test-session"},
        )
        assert r2.json()["session_id"] == "test-session"


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_redirects_to_notebook(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/notebook" in response.headers.get("location", "")

