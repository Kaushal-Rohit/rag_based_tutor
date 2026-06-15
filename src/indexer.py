"""
Indexer Module
==============
Loads documents from the dataset, generates embeddings using sentence-transformers,
and builds both FAISS HNSW and ChromaDB HNSW vector indices.
"""

import json
import os

import faiss
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def load_documents(dataset_dir: str):
    """
    Load all text chunks and metadata from the dataset directory.

    Reads from:
      - dataset_dir/rag_chunks/all_chunks.jsonl
      - dataset_dir/structured/**/*.json

    Returns:
        tuple: (documents, metadatas, ids)
    """
    rag_chunks_file = os.path.join(dataset_dir, "rag_chunks", "all_chunks.jsonl")
    structured_dir = os.path.join(dataset_dir, "structured")

    documents = []
    metadatas = []
    ids = []

    # Load pre-chunked RAG data
    if os.path.exists(rag_chunks_file):
        print(f"Loading data from {rag_chunks_file}...")
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
        print(f"Loaded {len(documents)} documents from all_chunks.jsonl")

    # Load structured JSONs
    if os.path.exists(structured_dir):
        print(f"Scanning for structured JSON files in {structured_dir}...")
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
                            print(f"Error decoding JSON from {file_path}")

    print(f"Total documents loaded: {len(documents)}")
    return documents, metadatas, ids


def build_embeddings(documents, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Encode documents into dense vectors using a sentence-transformer model."""
    print(f"Initializing {model_name}...")
    model = SentenceTransformer(model_name)
    print("Encoding documents...")
    embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings, output_path, m=32):
    """Build a FAISS HNSW index and save to disk."""
    print("Initializing FAISS HNSW index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, m)
    print("Adding embeddings to FAISS...")
    index.add(embeddings)
    faiss.write_index(index, output_path)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def build_chroma_index(documents, embeddings, metadatas, ids, chroma_path,
                       collection_name="rag_collection", batch_size=5000):
    """Build a ChromaDB persistent collection with HNSW cosine space."""
    print("Initializing Chroma DB...")
    client = chromadb.PersistentClient(path=chroma_path)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print("Adding documents to Chroma DB...")
    for i in tqdm(range(0, len(documents), batch_size)):
        collection.add(
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    print(f"Chroma DB collection created with {collection.count()} vectors.")
    return collection


if __name__ == "__main__":
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
    FAISS_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "faiss_hnsw_index.bin")
    CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "chroma_db")

    documents, metadatas, ids = load_documents(DATASET_DIR)
    embeddings = build_embeddings(documents)
    build_faiss_index(embeddings, FAISS_OUTPUT)
    build_chroma_index(documents, embeddings, metadatas, ids, CHROMA_PATH)
    print("\nAll indices built successfully.")
