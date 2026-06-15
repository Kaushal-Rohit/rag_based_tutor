"""
Retriever Module
================
Dual-backend retrieval engine supporting both FAISS HNSW and ChromaDB HNSW.
Provides metadata pre-filtering, latency measurement, and result comparison.
"""

import json
import os
import time

import faiss
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


class DualRetriever:
    """
    Retriever that searches both FAISS HNSW and ChromaDB HNSW indices.

    FAISS:    Post-retrieval metadata filtering via over-fetch strategy.
    ChromaDB: Native pre-retrieval metadata filtering via 'where' clause.
    """

    def __init__(self, faiss_index_path, chroma_db_path, dataset_dir,
                 model_name="all-MiniLM-L6-v2", collection_name="rag_collection"):
        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_index_path)

        # Load ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = chroma_client.get_collection(name=collection_name)

        # Load document metadata for FAISS lookups
        self.documents, self.metadatas, self.ids = self._load_metadata(dataset_dir)

        assert self.faiss_index.ntotal == len(self.documents), (
            f"FAISS vectors ({self.faiss_index.ntotal}) != documents ({len(self.documents)})"
        )

    def _load_metadata(self, dataset_dir):
        """Rebuild document/metadata/id lists in the same order as indexing."""
        from src.indexer import load_documents
        return load_documents(dataset_dir)

    def _embed_query(self, query):
        """Encode query string into embedding vector."""
        return self.model.encode([query])

    @staticmethod
    def _build_chroma_filter(filters):
        """Convert {field: value} dict to ChromaDB where clause."""
        if not filters:
            return None
        conditions = [{k: {"$eq": v}} for k, v in filters.items()]
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    @staticmethod
    def _check_metadata_match(metadata, filters):
        """Check if metadata dict satisfies all filter criteria."""
        return all(metadata.get(k, "") == v for k, v in filters.items())

    def search_faiss(self, query, k=5, filters=None, overfetch_factor=10):
        """
        Search FAISS HNSW index with optional post-retrieval metadata filtering.

        Over-fetches k * overfetch_factor results, then filters by metadata.
        """
        start = time.perf_counter()
        query_embedding = self._embed_query(query)
        t_embed = time.perf_counter()

        fetch_k = min(k * overfetch_factor if filters else k, self.faiss_index.ntotal)
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
        return {
            "documents": result_docs,
            "metadatas": result_metas,
            "ids": result_ids,
            "distances": result_dists,
            "latency_ms": {
                "embedding": (t_embed - start) * 1000,
                "search": (t_search - t_embed) * 1000,
                "filtering": (t_end - t_search) * 1000,
                "total": (t_end - start) * 1000,
            },
        }

    def search_chroma(self, query, k=5, filters=None):
        """
        Search ChromaDB collection with native metadata pre-filtering.
        """
        start = time.perf_counter()
        query_embedding = self._embed_query(query)
        t_embed = time.perf_counter()

        where_filter = self._build_chroma_filter(filters)
        query_params = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)
        t_end = time.perf_counter()

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "latency_ms": {
                "embedding": (t_embed - start) * 1000,
                "search": (t_end - t_embed) * 1000,
                "total": (t_end - start) * 1000,
            },
        }

    def search_both(self, query, k=5, filters=None):
        """Run retrieval on both backends and return combined results."""
        return {
            "faiss": self.search_faiss(query, k=k, filters=filters),
            "chroma": self.search_chroma(query, k=k, filters=filters),
        }
