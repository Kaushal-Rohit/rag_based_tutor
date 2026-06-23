"""
Application Configuration
=========================
All configuration via environment variables, loaded with pydantic-settings.
No hardcoded secrets or absolute paths.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Central configuration for the Adaptive RAG Pipeline.

    All values can be overridden via environment variables prefixed with ``RAG_``.
    A ``.env`` file in the project root is also read automatically.
    """

    # ── Paths ──
    dataset_dir: str = "./dataset"
    faiss_index_path: str = "./notebooks/faiss_hnsw_index.bin"
    chroma_db_path: str = "./notebooks/chroma_db"
    log_dir: str = "./logs"

    # ── Embedding model ──
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ── Ollama LLM ──
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:latest"

    # ── Retrieval ──
    retrieval_backend: str = "chroma"  # chroma | faiss | both
    base_k: int = 5
    overfetch_factor: int = 10

    # ── Sentiment ──
    max_history: int = 5

    # ── CRAG (Corrective RAG) ──
    crag_enabled: bool = True

    # ── Logging ──
    log_level: str = "INFO"
    log_full_content: bool = False  # full prompt/response at DEBUG only

    # ── Upload pipeline ──
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 25
    max_upload_pages: int = 500
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Server ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance — import this wherever config is needed.
settings = AppConfig()
