"""
Main Entry Point
=================
Runs the full Adaptive RAG pipeline as an interactive CLI chat session.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from src.llm_engine import LocalLLMEngine, DynamicRAGPipeline


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DB_PATH = os.path.join(BASE_DIR, "..", "notebooks", "chroma_db")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name="rag_collection")
    print(f"Database loaded with {collection.count()} vectors.\n")

    llm = LocalLLMEngine(model_name="llama3:latest")
    if not llm.check_connection():
        print("[ERROR] Cannot connect to Ollama. Run 'ollama serve' first.")
        return

    rag = DynamicRAGPipeline(collection, model, llm, base_k=5)

    print("=" * 60)
    print("  Adaptive RAG System - Interactive Chat")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break
        if not query:
            continue

        response = rag.chat(query)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    main()
