"""
Retriever Service
=================
Dual-backend retrieval engine supporting both FAISS HNSW and ChromaDB HNSW.

Refactored from ``src/retriever.py`` to:
  - Accept pre-loaded resources (no loading in __init__)
  - LRU cache on query embeddings to avoid recomputation
  - Configurable backend dispatch (chroma / faiss / both)
  - Structured latency logging per stage
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Simple LRU cache for query embeddings to avoid recomputation."""

    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class DualRetriever:
    """
    Retriever that searches FAISS HNSW and/or ChromaDB HNSW indices.

    Designed to receive pre-loaded resources at construction time
    (loaded once during app startup via the lifespan manager).
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        faiss_index: faiss.Index | None = None,
        chroma_collection: Any | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        overfetch_factor: int | None = None,
    ):
        self.model = embedding_model
        self.faiss_index = faiss_index
        self.chroma_collection = chroma_collection
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.ids = ids or []
        self.overfetch_factor = overfetch_factor or settings.overfetch_factor
        self._cache = EmbeddingCache(maxsize=256)

    def _embed_query(self, query: str) -> np.ndarray:
        """Encode query string into embedding vector, with LRU caching."""
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        embedding = self.model.encode([query])
        self._cache.put(query, embedding)
        return embedding

    # ──────────────────────────────────────────
    # FAISS search
    # ──────────────────────────────────────────

    @staticmethod
    def _check_metadata_match(metadata: dict, filters: dict) -> bool:
        """Check if metadata dict satisfies all filter criteria."""
        if not filters:
            return True
        
        if "$or" in filters:
            or_conditions = filters["$or"]
            # Check if at least one condition in the $or array matches
            or_match = any(
                all(metadata.get(k, "") == v for k, v in cond.items())
                for cond in or_conditions
            )
            if not or_match:
                return False
                
            # Check remaining AND filters
            remaining = {k: v for k, v in filters.items() if k != "$or"}
            return all(metadata.get(k, "") == v for k, v in remaining.items())
            
        return all(metadata.get(k, "") == v for k, v in filters.items())

    def search_faiss(
        self, query: str, k: int = 5, filters: dict | None = None
    ) -> dict:
        """
        Search FAISS HNSW index with optional post-retrieval metadata filtering.

        Over-fetches ``k * overfetch_factor`` results, then filters by metadata.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded")

        start = time.perf_counter()
        query_embedding = self._embed_query(query)
        t_embed = time.perf_counter()

        fetch_k = min(
            k * self.overfetch_factor if filters else k,
            self.faiss_index.ntotal,
        )
        D, I = self.faiss_index.search(query_embedding, k=fetch_k)
        t_search = time.perf_counter()

        result_docs, result_metas, result_ids, result_dists = [], [], [], []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            if filters and not self._check_metadata_match(self.metadatas[idx], filters):
                continue
            result_docs.append(self.documents[idx])
            result_metas.append(self.metadatas[idx])
            result_ids.append(self.ids[idx])
            result_dists.append(float(dist))
            if len(result_docs) >= k:
                break

        t_end = time.perf_counter()
        latency = {
            "embedding_ms": (t_embed - start) * 1000,
            "search_ms": (t_search - t_embed) * 1000,
            "filtering_ms": (t_end - t_search) * 1000,
            "total_ms": (t_end - start) * 1000,
        }

        logger.info(
            f"FAISS search: k={k}, returned={len(result_docs)}, "
            f"total={latency['total_ms']:.2f}ms",
            extra={"stream": "retrieval", "context": latency},
        )

        return {
            "documents": result_docs,
            "metadatas": result_metas,
            "ids": result_ids,
            "distances": result_dists,
            "latency": latency,
        }

    # ──────────────────────────────────────────
    # ChromaDB search
    # ──────────────────────────────────────────

    @staticmethod
    def _build_chroma_filter(filters: dict | None) -> dict | None:
        """Convert {field: value} dict to ChromaDB where clause."""
        if not filters:
            return None
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def search_chroma(
        self, query: str, k: int = 5, filters: dict | None = None
    ) -> dict:
        """Search ChromaDB collection with native metadata pre-filtering."""
        if self.chroma_collection is None:
            raise RuntimeError("ChromaDB collection not loaded")

        start = time.perf_counter()
        query_embedding = self._embed_query(query)
        t_embed = time.perf_counter()

        where_filter = self._build_chroma_filter(filters)
        query_params: dict[str, Any] = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.chroma_collection.query(**query_params)
        t_end = time.perf_counter()

        latency = {
            "embedding_ms": (t_embed - start) * 1000,
            "search_ms": (t_end - t_embed) * 1000,
            "total_ms": (t_end - start) * 1000,
        }

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        r_ids = results["ids"][0] if results["ids"] else []
        dists = results["distances"][0] if results["distances"] else []

        logger.info(
            f"ChromaDB search: k={k}, returned={len(docs)}, "
            f"total={latency['total_ms']:.2f}ms",
            extra={"stream": "retrieval", "context": latency},
        )

        return {
            "documents": docs,
            "metadatas": metas,
            "ids": r_ids,
            "distances": dists,
            "latency": latency,
        }

    # ──────────────────────────────────────────
    # Unified search dispatcher
    # ──────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
        backend: str | None = None,
    ) -> dict:
        """
        Dispatch search to the configured backend.

        Args:
            backend: Override the configured backend for this call.
                     One of 'chroma', 'faiss', 'both'.
        """
        backend = backend or settings.retrieval_backend

        if backend == "faiss":
            return self.search_faiss(query, k=k, filters=filters)
        elif backend == "both":
            return {
                "faiss": self.search_faiss(query, k=k, filters=filters),
                "chroma": self.search_chroma(query, k=k, filters=filters),
            }
        else:  # default: chroma
            return self.search_chroma(query, k=k, filters=filters)
