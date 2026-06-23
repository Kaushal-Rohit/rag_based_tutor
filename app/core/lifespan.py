"""
Application Lifespan Manager
=============================
Handles startup and shutdown of all shared resources.

Loads the embedding model, FAISS index, ChromaDB client, and LLM engine
**once** at startup and stores them in ``app.state`` for reuse across requests.
"""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:
      1. Configure logging
      2. Load embedding model (GPU-aware)
      3. Load FAISS index (if backend requires it)
      4. Load ChromaDB collection (if backend requires it)
      5. Load document metadata (for FAISS post-filtering)
      6. Initialize LLM engine + health check
      7. Initialize CRAG pipeline
      8. Initialize session manager + metrics store

    Shutdown:
      - Close LLM engine HTTP client
      - Log session statistics
    """
    # ── 1. Logging ──
    setup_logging()
    logger.info("=" * 60, extra={"stream": "app"})
    logger.info("  Adaptive RAG Pipeline — Starting up", extra={"stream": "app"})
    logger.info("=" * 60, extra={"stream": "app"})

    # ── 2. Embedding model (GPU-aware) ──
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Loading embedding model: {settings.embedding_model_name} (device={device})",
        extra={"stream": "app"},
    )
    embedding_model = SentenceTransformer(settings.embedding_model_name, device=device)
    app.state.embedding_model = embedding_model

    # ── 3. Load document metadata (needed for FAISS post-filtering) ──
    from app.services.indexer import load_documents

    logger.info("Loading document metadata...", extra={"stream": "app"})
    documents, metadatas, ids = load_documents(settings.dataset_dir)
    app.state.documents = documents
    app.state.metadatas = metadatas
    app.state.doc_ids = ids

    # ── 4. FAISS index ──
    faiss_index = None
    backend = settings.retrieval_backend
    if backend in ("faiss", "both"):
        import faiss

        logger.info(
            f"Loading FAISS index from {settings.faiss_index_path}",
            extra={"stream": "app"},
        )
        faiss_index = faiss.read_index(settings.faiss_index_path)
        logger.info(
            f"FAISS loaded: {faiss_index.ntotal} vectors, dim={faiss_index.d}",
            extra={"stream": "app"},
        )
        assert faiss_index.ntotal == len(documents), (
            f"FAISS vectors ({faiss_index.ntotal}) != documents ({len(documents)}). "
            "The index may be stale — re-run ingestion."
        )
    app.state.faiss_index = faiss_index

    # ── 5. ChromaDB ──
    chroma_collection = None
    if backend in ("chroma", "both"):
        import chromadb

        logger.info(
            f"Loading ChromaDB from {settings.chroma_db_path}",
            extra={"stream": "app"},
        )
        chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        chroma_collection = chroma_client.get_collection(name="rag_collection")
        logger.info(
            f"ChromaDB loaded: {chroma_collection.count()} vectors",
            extra={"stream": "app"},
        )
        app.state.chroma_client = chroma_client
    app.state.chroma_collection = chroma_collection

    # ── 6. Retriever ──
    from app.services.retriever import DualRetriever

    app.state.retriever = DualRetriever(
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        chroma_collection=chroma_collection,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    # ── 7. LLM Engine + health check ──
    from app.services.llm_engine import AsyncLLMEngine

    llm_engine = AsyncLLMEngine()
    app.state.llm_engine = llm_engine

    ollama_ok = await llm_engine.check_connection()
    if not ollama_ok:
        logger.error(
            "CRITICAL: Cannot connect to Ollama at "
            f"{settings.ollama_base_url}. "
            "The API will start but /query will fail. "
            "Run 'ollama serve' and restart.",
            extra={"stream": "error"},
        )
    else:
        model_ok = await llm_engine.check_model_available()
        if not model_ok:
            logger.error(
                f"CRITICAL: Model '{settings.ollama_model}' not found in Ollama. "
                f"Run 'ollama pull {settings.ollama_model}' and restart.",
                extra={"stream": "error"},
            )
        else:
            logger.info(
                f"Ollama connected: model '{settings.ollama_model}' available",
                extra={"stream": "app"},
            )

    # ── 8. CRAG pipeline ──
    from app.services.crag import CRAGPipeline

    app.state.crag = CRAGPipeline(llm_engine)
    logger.info(
        f"CRAG pipeline: {'enabled' if settings.crag_enabled else 'disabled'}",
        extra={"stream": "app"},
    )

    # ── 9. Session manager + metrics ──
    from collections import deque

    from app.services.sentiment import DynamicKSelector, SessionManager

    app.state.session_manager = SessionManager(max_history=settings.max_history)
    app.state.k_selector = DynamicKSelector(base_k=settings.base_k)
    app.state.metrics = {
        "total_queries": 0,
        "retrieval_latencies": deque(maxlen=100),
        "sentiment_distribution": {"confused": 0, "neutral": 0, "clear": 0},
    }

    # ── 10. Upload store ──
    from app.services.upload_store import UploadStore

    app.state.upload_store = UploadStore()

    logger.info("=" * 60, extra={"stream": "app"})
    logger.info("  Startup complete — ready to serve requests", extra={"stream": "app"})
    logger.info("=" * 60, extra={"stream": "app"})

    # ── Yield control to the application ──
    yield

    # ── Shutdown ──
    logger.info("Shutting down...", extra={"stream": "app"})
    await llm_engine.close()
    metrics = app.state.metrics
    logger.info(
        f"Session stats: {metrics['total_queries']} queries served, "
        f"{app.state.session_manager.active_count} sessions active",
        extra={"stream": "app"},
    )
    logger.info("Shutdown complete.", extra={"stream": "app"})
