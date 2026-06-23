"""
Indexer Service
===============
Loads documents from the dataset, generates embeddings, and builds
both FAISS HNSW and ChromaDB HNSW vector indices.

Refactored from ``src/indexer.py`` — same loading logic, but designed
to accept pre-loaded models and work as a service rather than a script.
"""

import json
import os

import faiss
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def load_documents(dataset_dir: str | None = None) -> tuple[list[str], list[dict], list[str]]:
    """
    Load all text chunks and metadata from the dataset directory.

    Reads from:
      - ``dataset_dir/rag_chunks/all_chunks.jsonl``
      - ``dataset_dir/structured/**/*.json``

    Returns:
        tuple: (documents, metadatas, ids)
    """
    dataset_dir = dataset_dir or settings.dataset_dir
    rag_chunks_file = os.path.join(dataset_dir, "rag_chunks", "all_chunks.jsonl")
    structured_dir = os.path.join(dataset_dir, "structured")

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # ── Load pre-chunked RAG data ──
    if os.path.exists(rag_chunks_file):
        logger.info(f"Loading data from {rag_chunks_file}", extra={"stream": "app"})
        with open(rag_chunks_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get("text", "")
                chunk_id = data.get("chunk_id", f"chunk_{i}")
                if text:
                    documents.append(text)
                    metadata = {
                        k: str(v) if v is not None else ""
                        for k, v in data.items()
                        if k != "text"
                    }
                    metadatas.append(metadata)
                    ids.append(chunk_id)
        logger.info(
            f"Loaded {len(documents)} documents from all_chunks.jsonl",
            extra={"stream": "app"},
        )

    # ── Load structured JSONs ──
    if os.path.exists(structured_dir):
        logger.info(
            f"Scanning for structured JSON files in {structured_dir}",
            extra={"stream": "app"},
        )
        for root, _dirs, files in os.walk(structured_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for idx, item in enumerate(data):
                                    text = item.get("text", "")
                                    if text:
                                        documents.append(text)
                                        metadata = {
                                            k: str(v) if v is not None else ""
                                            for k, v in item.items()
                                            if k != "text"
                                        }
                                        metadata["source_file"] = file_path
                                        metadatas.append(metadata)
                                        ids.append(f"{file}_{idx}")
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Error decoding JSON from {file_path}",
                                extra={"stream": "error"},
                            )

    logger.info(
        f"Total documents loaded: {len(documents)}",
        extra={"stream": "app"},
    )
    return documents, metadatas, ids


def build_embeddings(
    documents: list[str],
    model: SentenceTransformer | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode documents into dense vectors using a sentence-transformer model.

    If ``model`` is provided, it is reused (no reload). Otherwise, a new model
    is loaded from ``model_name``.
    """
    if model is None:
        model_name = model_name or settings.embedding_model_name
        logger.info(f"Loading embedding model: {model_name}", extra={"stream": "app"})
        model = SentenceTransformer(model_name)

    logger.info(
        f"Encoding {len(documents)} documents (batch_size={batch_size})",
        extra={"stream": "app"},
    )
    embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape}", extra={"stream": "app"})
    return embeddings


def build_faiss_index(embeddings: np.ndarray, output_path: str, m: int = 32) -> faiss.Index:
    """Build a FAISS HNSW index and save to disk."""
    logger.info("Building FAISS HNSW index...", extra={"stream": "app"})
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, m)
    index.add(embeddings)
    faiss.write_index(index, output_path)
    logger.info(
        f"FAISS index built: {index.ntotal} vectors, dim={dimension}",
        extra={"stream": "app"},
    )
    return index


def build_chroma_index(
    documents: list[str],
    embeddings: np.ndarray,
    metadatas: list[dict],
    ids: list[str],
    chroma_path: str,
    collection_name: str = "rag_collection",
    batch_size: int = 5000,
) -> chromadb.Collection:
    """Build a ChromaDB persistent collection with HNSW cosine space."""
    logger.info("Building ChromaDB index...", extra={"stream": "app"})
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for i in tqdm(range(0, len(documents), batch_size)):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    logger.info(
        f"ChromaDB collection created: {collection.count()} vectors",
        extra={"stream": "app"},
    )
    return collection
