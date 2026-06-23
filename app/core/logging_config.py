"""
Structured Logging Configuration
=================================
Sets up dual-output logging:
  - Console: human-readable format (for ``docker logs``)
  - File: JSON-structured (machine-parseable, one object per line)

Each log entry includes a ``request_id`` (threaded via contextvars) and a
``stream`` tag (access / retrieval / generation / error) for filtering.
"""

import json
import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from app.core.config import settings

# ──────────────────────────────────────────────
# Context variable for per-request tracking
# ──────────────────────────────────────────────
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ──────────────────────────────────────────────
# JSON formatter for file output
# ──────────────────────────────────────────────
class JSONLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "request_id": request_id_ctx.get("-"),
            "logger": record.name,
            "stream": getattr(record, "stream", "app"),
            "message": record.getMessage(),
        }
        # Attach optional structured context
        context = getattr(record, "context", None)
        if context:
            log_entry["context"] = context
        # Attach exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
            log_entry["stream"] = "error"
        return json.dumps(log_entry, default=str)


# ──────────────────────────────────────────────
# Human-readable formatter for console output
# ──────────────────────────────────────────────
class ConsoleLogFormatter(logging.Formatter):
    """Compact, colored format for console / ``docker logs``."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        rid = request_id_ctx.get("-")
        short_rid = rid[:8] if rid != "-" else "-"
        stream = getattr(record, "stream", "app")
        ts = datetime.now().strftime("%H:%M:%S")
        return (
            f"{color}{ts} [{record.levelname:<7}]{self.RESET} "
            f"[{short_rid}] [{stream}] {record.getMessage()}"
        )


# ──────────────────────────────────────────────
# Setup function — call once at startup
# ──────────────────────────────────────────────
def setup_logging() -> None:
    """Configure the root logger with console + rotating file handlers."""
    log_dir = settings.log_dir
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove any pre-existing handlers (e.g. from uvicorn defaults)
    root_logger.handlers.clear()

    # ── Console handler (force UTF-8 on Windows to avoid cp1252 errors) ──
    console_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(ConsoleLogFormatter())
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    # ── File handler (JSON, rotating) ──
    log_file = os.path.join(log_dir, "app.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONLogFormatter())
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers",
                   "uvicorn.access", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info("Logging initialized", extra={"stream": "app"})


# ──────────────────────────────────────────────
# Convenience helper for structured logging
# ──────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a named logger (uses the already-configured root handlers)."""
    return logging.getLogger(name)
