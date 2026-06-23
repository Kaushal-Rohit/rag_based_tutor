"""
API Router Aggregator
=====================
Collects all endpoint routers under the ``/api/v1`` prefix.
"""

from fastapi import APIRouter

from app.api.endpoints import health, ingest, metrics, query, upload

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(metrics.router, tags=["Metrics"])
api_router.include_router(query.router, tags=["Query"])
api_router.include_router(ingest.router, tags=["Ingestion"])
api_router.include_router(upload.router, tags=["Upload"])

