"""
Upload Endpoints
==================
``POST /api/v1/upload`` — PDF upload with background processing.
``GET  /api/v1/upload/{job_id}/status`` — poll processing status.
``GET  /api/v1/upload/session/{session_id}`` — list session uploads.
``DELETE /api/v1/upload/{job_id}`` — remove uploaded source.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.logging_config import get_logger, request_id_ctx
from app.models.schemas import ErrorResponse, UploadResponse, UploadStatusResponse
from app.services.upload import UploadValidationError, process_upload, validate_pdf

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a PDF for RAG ingestion",
    description="Validates the PDF synchronously, then processes it in the background. "
                "Poll the status endpoint with the returned job_id.",
    responses={
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Invalid file type"},
        422: {"model": ErrorResponse, "description": "Encrypted or unreadable PDF"},
    },
)
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    session_id: str = Form(..., description="Session ID for scoping"),
):
    """Upload a PDF for processing and incremental indexing."""
    req_id = request_id_ctx.get("-")
    upload_store = request.app.state.upload_store

    # Read file bytes
    file_bytes = await file.read()
    filename = file.filename or "unknown.pdf"

    # Synchronous validation — fail fast before queueing
    try:
        doc = validate_pdf(file_bytes, filename)
        doc.close()  # We'll re-open in the background task
    except UploadValidationError as e:
        status_map = {
            "invalid_file_type": 415,
            "file_too_large": 413,
            "encrypted_pdf": 422,
            "unreadable_pdf": 422,
            "too_many_pages": 422,
        }
        status_code = status_map.get(e.error_code, 400)
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error_code=e.error_code,
                message=str(e),
                request_id=req_id,
            ).model_dump(),
        )

    # Create job
    job_id = str(uuid.uuid4())[:12]
    job = upload_store.create_job(job_id, session_id, filename)

    logger.info(
        f"Upload accepted: {filename} ({len(file_bytes)} bytes) -> job {job_id}",
        extra={
            "stream": "ingestion",
            "context": {
                "job_id": job_id,
                "session_id": session_id,
                "filename": filename,
                "size_bytes": len(file_bytes),
            },
        },
    )

    # Queue background processing
    background_tasks.add_task(
        process_upload, job, file_bytes, request.app.state,
    )

    return UploadResponse(job_id=job_id, status="queued", filename=filename)


@router.get(
    "/upload/{job_id}/status",
    response_model=UploadStatusResponse,
    summary="Check upload processing status",
    responses={404: {"model": ErrorResponse}},
)
async def upload_status(request: Request, job_id: str):
    """Get the current processing status of an upload job."""
    upload_store = request.app.state.upload_store
    job = upload_store.get_job(job_id)

    if job is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error_code="job_not_found",
                message=f"No upload job found with ID '{job_id}'",
                request_id=request_id_ctx.get("-"),
            ).model_dump(),
        )

    return UploadStatusResponse(**job.to_dict())


@router.get(
    "/upload/session/{session_id}",
    response_model=list[UploadStatusResponse],
    summary="List all uploads for a session",
)
async def session_uploads(request: Request, session_id: str):
    """List all upload jobs for a given session (newest first)."""
    upload_store = request.app.state.upload_store
    jobs = upload_store.get_session_jobs(session_id)
    return [UploadStatusResponse(**j.to_dict()) for j in jobs]


@router.delete(
    "/upload/{job_id}",
    summary="Delete an uploaded source",
    responses={404: {"model": ErrorResponse}},
)
async def delete_upload(request: Request, job_id: str):
    """Remove an uploaded source and its chunks from the indices."""
    upload_store = request.app.state.upload_store
    job = upload_store.get_job(job_id)

    if job is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error_code="job_not_found",
                message=f"No upload job found with ID '{job_id}'",
                request_id=request_id_ctx.get("-"),
            ).model_dump(),
        )

    # Remove chunks from in-memory lists (by upload_id)
    app_state = request.app.state
    indices_to_remove = [
        i for i, m in enumerate(app_state.metadatas)
        if m.get("upload_id") == job_id
    ]

    # Remove in reverse order to preserve indices
    for i in sorted(indices_to_remove, reverse=True):
        app_state.documents.pop(i)
        app_state.metadatas.pop(i)
        app_state.doc_ids.pop(i)

    # Note: FAISS doesn't support deletion from HNSW indices.
    # ChromaDB supports deletion by ID.
    if app_state.chroma_collection is not None:
        chunk_ids = [
            m.get("chunk_id") for m in app_state.metadatas
            if m.get("upload_id") == job_id
        ]
        if chunk_ids:
            try:
                app_state.chroma_collection.delete(ids=chunk_ids)
            except Exception:
                pass  # Best-effort; FAISS can't delete anyway

    upload_store.remove_job(job_id)

    logger.info(
        f"Deleted upload {job_id}: removed {len(indices_to_remove)} chunks",
        extra={"stream": "ingestion"},
    )

    return {"status": "deleted", "job_id": job_id, "chunks_removed": len(indices_to_remove)}
