"""
Report Service
===============
Generates a data-driven report by reading from structured logs,
benchmark outputs, and runtime metrics.

Every number in the report comes from a measured source — never hand-typed.
"""

import json
import os
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def generate_report_data(app_state) -> dict:
    """
    Collect report data from all measured sources.

    Sources:
      1. benchmarks/results.json — retrieval benchmark numbers
      2. app.state.metrics — runtime query stats
      3. app.state.upload_store — upload pipeline stats
      4. System info from benchmark results

    Returns:
        dict with all report sections populated from real data.
    """
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": {},
        "benchmarks": {},
        "runtime": {},
        "upload_stats": {},
    }

    # ── 1. Benchmark data (from results.json) ──
    benchmark_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "benchmarks", "results.json"
    )

    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)

            report["system"] = bench_data.get("system", {})
            report["benchmarks"] = bench_data.get("key_metrics", {})
            report["benchmarks"]["per_query"] = bench_data.get("per_query_results", [])
        except Exception as e:
            logger.warning(
                f"Failed to read benchmark data: {e}",
                extra={"stream": "app"},
            )
    else:
        report["benchmarks"]["note"] = (
            "No benchmark data found. Run `python benchmarks/bench_retrieval.py` "
            "to generate performance metrics."
        )

    # ── 2. Runtime metrics ──
    metrics = getattr(app_state, "metrics", {})
    latencies = list(metrics.get("retrieval_latencies", []))

    report["runtime"] = {
        "total_queries": metrics.get("total_queries", 0),
        "active_sessions": getattr(
            getattr(app_state, "session_manager", None), "active_count", 0
        ),
        "sentiment_distribution": metrics.get("sentiment_distribution", {}),
        "avg_retrieval_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "recent_latencies_ms": latencies[-10:],
    }

    # ── 3. Upload stats ──
    upload_store = getattr(app_state, "upload_store", None)
    if upload_store:
        all_jobs = list(upload_store._jobs.values())
        completed = [j for j in all_jobs if j.status == "ready"]
        failed = [j for j in all_jobs if j.status == "failed"]

        processing_times = []
        for j in completed:
            total = sum(j.timings.values())
            if total > 0:
                processing_times.append(total)

        report["upload_stats"] = {
            "total_uploads": len(all_jobs),
            "completed": len(completed),
            "failed": len(failed),
            "total_chunks_created": sum(j.chunks_created for j in completed),
            "avg_processing_time_ms": (
                round(sum(processing_times) / len(processing_times), 1)
                if processing_times else None
            ),
            "error_rate_pct": (
                round(len(failed) / len(all_jobs) * 100, 1)
                if all_jobs else 0.0
            ),
        }

    # ── 4. Index stats ──
    faiss_index = getattr(app_state, "faiss_index", None)
    chroma_collection = getattr(app_state, "chroma_collection", None)

    report["index"] = {
        "total_documents": len(getattr(app_state, "documents", [])),
        "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
        "chroma_vectors": chroma_collection.count() if chroma_collection else 0,
        "embedding_model": settings.embedding_model_name,
        "retrieval_backend": settings.retrieval_backend,
    }

    return report
