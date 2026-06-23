"""
Metrics Endpoint
================
``GET /api/v1/metrics`` — exposes runtime metrics.

Reports:
  - Total queries since startup
  - Active session count
  - Recent retrieval latencies
  - Sentiment distribution across queries
"""

from fastapi import APIRouter, Request

from app.models.schemas import MetricsResponse

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Runtime metrics",
    description="Returns current session metrics, retrieval latencies, and sentiment distribution.",
)
async def get_metrics(request: Request) -> MetricsResponse:
    """Return current runtime metrics."""
    metrics = request.app.state.metrics
    latencies = list(metrics["retrieval_latencies"])

    avg_latency = None
    if latencies:
        avg_latency = round(sum(latencies) / len(latencies), 2)

    return MetricsResponse(
        total_queries=metrics["total_queries"],
        active_sessions=request.app.state.session_manager.active_count,
        recent_retrieval_latencies_ms=[round(l, 2) for l in latencies[-20:]],
        average_retrieval_latency_ms=avg_latency,
        sentiment_distribution=dict(metrics["sentiment_distribution"]),
    )
