# ──────────────────────────────────────────────────────────────
# Adaptive RAG Pipeline — Multi-stage Docker build
# ──────────────────────────────────────────────────────────────
# Build:  docker build -t rag-api .
# Run:    docker-compose up
#
# GPU support: To use faiss-gpu and CUDA, replace the base image
# with nvidia/cuda:12.1-runtime-ubuntu22.04, install Python, and
# swap faiss-cpu for faiss-gpu in requirements.txt.
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download TextBlob corpora (needed for sentiment analysis)
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/nltk_data')" 2>/dev/null || true
RUN pip install --no-cache-dir textblob && python -m textblob.download_corpora


# ── Stage 2: Runtime ──
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -r -s /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/nltk_data /home/appuser/nltk_data

# Set NLTK data path
ENV NLTK_DATA=/home/appuser/nltk_data

# Install curl for healthcheck + libmagic for file type detection
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libmagic1 && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Create log, data, and upload directories
RUN mkdir -p /app/logs /app/data /app/uploads && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
