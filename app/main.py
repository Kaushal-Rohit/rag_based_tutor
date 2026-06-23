"""
FastAPI Application Entry Point
================================
Creates the FastAPI app with lifespan management, middleware, and routes.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.endpoints.notebook import router as notebook_router
from app.api.endpoints.report import router as report_router
from app.api.router import api_router
from app.core.lifespan import lifespan
from app.core.logging_config import get_logger, request_id_ctx

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Create the FastAPI application
# ──────────────────────────────────────────────
app = FastAPI(
    title="Adaptive RAG Pipeline API",
    description=(
        "Retrieval-Augmented Generation with HNSW indexing, "
        "sentiment-driven dynamic retrieval, and Corrective RAG. "
        "Built for NCERT textbook question-answering."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request ID middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a unique request_id to every incoming request."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_ctx.set(req_id)

    t_start = time.perf_counter()
    response = await call_next(request)
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    response.headers["X-Request-ID"] = req_id

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({latency_ms:.0f}ms)",
        extra={
            "stream": "access",
            "context": {
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "request_id": req_id,
            },
        },
    )

    return response


# ──────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a structured error response."""
    logger.error(
        f"Unhandled exception: {exc}",
        extra={"stream": "error"},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "internal_server_error",
            "message": str(exc),
            "request_id": request_id_ctx.get("-"),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured 422 errors with field-level details."""
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "validation_error",
            "message": "Request validation failed",
            "request_id": request_id_ctx.get("-"),
            "details": exc.errors(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Wrap HTTP exceptions in the consistent ErrorResponse schema."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"http_{exc.status_code}",
            "message": str(exc.detail),
            "request_id": request_id_ctx.get("-"),
        },
    )


# ── Mount static files ──
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Mount routes ──
app.include_router(api_router)
app.include_router(notebook_router)
app.include_router(report_router)


# ── Root redirect ──
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the notebook UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/notebook")
