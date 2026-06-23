"""
Upload Job Store
=================
In-memory store for tracking PDF upload processing jobs.

Each upload gets a ``UploadJob`` that transitions through:
    queued → extracting → chunking → embedding → indexing → ready
    (or → failed at any stage)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class UploadJob:
    """Tracks the state and timing of a single PDF upload."""

    job_id: str
    session_id: str
    filename: str
    status: str = "queued"  # queued|extracting|chunking|embedding|indexing|ready|failed
    error: str | None = None
    error_code: str | None = None
    chunks_created: int = 0
    pages_processed: int = 0
    total_pages: int = 0
    created_at: str = ""
    completed_at: str | None = None
    timings: dict = field(default_factory=dict)  # stage → ms
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str, error_code: str = "processing_error") -> None:
        """Mark the job as failed with an error message."""
        self.status = "failed"
        self.error = error
        self.error_code = error_code
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def complete(self, chunks: int) -> None:
        """Mark the job as successfully completed."""
        self.status = "ready"
        self.chunks_created = chunks
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "error_code": self.error_code,
            "chunks_created": self.chunks_created,
            "pages_processed": self.pages_processed,
            "total_pages": self.total_pages,
            "timings": self.timings,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class UploadStore:
    """
    Thread-safe in-memory store for upload jobs.

    Sufficient for single-process BackgroundTasks; would need Redis
    or a database for multi-worker deployments.
    """

    def __init__(self):
        self._jobs: dict[str, UploadJob] = {}

    def create_job(self, job_id: str, session_id: str, filename: str) -> UploadJob:
        """Create and register a new upload job."""
        job = UploadJob(job_id=job_id, session_id=session_id, filename=filename)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> UploadJob | None:
        """Get a job by ID, or None if not found."""
        return self._jobs.get(job_id)

    def get_session_jobs(self, session_id: str) -> list[UploadJob]:
        """Get all jobs for a given session, newest first."""
        jobs = [j for j in self._jobs.values() if j.session_id == session_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def remove_job(self, job_id: str) -> UploadJob | None:
        """Remove a job from the store. Returns the removed job or None."""
        return self._jobs.pop(job_id, None)

    @property
    def total_jobs(self) -> int:
        return len(self._jobs)

    @property
    def active_jobs(self) -> int:
        return sum(
            1 for j in self._jobs.values()
            if j.status not in ("ready", "failed")
        )
