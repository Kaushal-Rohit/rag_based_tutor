"""
LLM Engine Service
==================
Async interface to a local Ollama instance with streaming support.

Refactored from ``src/llm_engine.py``:
  - Uses ``httpx.AsyncClient`` with connection pooling instead of ``requests``
  - Adds ``generate_stream()`` for SSE token-by-token delivery
  - Adds ``check_model_available()`` for startup health checks
  - Persistent client created once, reused across requests
"""

import hashlib
import json
import time
from typing import AsyncGenerator

import httpx

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class AsyncLLMEngine:
    """
    Async interface to a local Ollama instance.

    Sends structured prompts to the Ollama ``/api/generate`` endpoint
    with configurable system instructions. Supports both blocking and
    streaming generation.
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        self.model_name = model_name or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.api_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"
        timeout_config = httpx.Timeout(timeout, connect=10.0) if timeout else httpx.Timeout(None, connect=10.0)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_config,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def check_connection(self) -> bool:
        """Verify Ollama server is reachable."""
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def check_model_available(self) -> bool:
        """
        Check whether the configured model is available in Ollama.

        Calls ``/api/tags`` and looks for ``self.model_name`` in the list.
        """
        try:
            resp = await self._client.get("/api/tags")
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Ollama sometimes returns names with/without ':latest'
            target = self.model_name
            return any(
                target == m or target == m.split(":")[0]
                for m in models
            )
        except Exception as e:
            logger.error(
                f"Failed to check model availability: {e}",
                extra={"stream": "error"},
            )
            return False

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> str:
        """Send prompt to Ollama and return the full generated text."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_instruction,
            "stream": False,
        }

        t_start = time.perf_counter()
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            result = resp.json()
            response_text = result.get("response", "")
            t_end = time.perf_counter()

            # Log generation metrics
            latency_ms = (t_end - t_start) * 1000
            token_count = result.get("eval_count", None)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]

            logger.info(
                f"LLM generation: {latency_ms:.0f}ms, "
                f"tokens={token_count or 'N/A'}, "
                f"prompt_hash={prompt_hash}, "
                f"response_len={len(response_text)}",
                extra={
                    "stream": "generation",
                    "context": {
                        "latency_ms": latency_ms,
                        "token_count": token_count,
                        "model": self.model_name,
                        "prompt_length": len(prompt),
                        "response_length": len(response_text),
                    },
                },
            )

            if settings.log_full_content:
                logger.debug(
                    f"Full prompt:\n{prompt}\n\nFull response:\n{response_text}",
                    extra={"stream": "generation"},
                )

            return response_text

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ollama HTTP error: {e.response.status_code} — {e.response.text}",
                extra={"stream": "error"},
            )
            return f"[ERROR] LLM generation failed: HTTP {e.response.status_code}"
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama. Is it running?",
                extra={"stream": "error"},
            )
            return "[ERROR] Cannot connect to Ollama. Run 'ollama serve' first."
        except Exception as e:
            logger.error(
                f"LLM generation failed: {e}",
                extra={"stream": "error"},
                exc_info=True,
            )
            return f"[ERROR] LLM generation failed: {e}"

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama one at a time.

        Yields individual response text chunks as they arrive.
        Used for SSE (Server-Sent Events) delivery to the client.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_instruction,
            "stream": True,
        }

        t_start = time.perf_counter()
        full_response = ""

        try:
            async with self._client.stream(
                "POST", "/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            full_response += token
                            yield token
                        # Ollama sends done=true on the last chunk
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPStatusError as e:
            error_msg = f"[ERROR] Ollama stream error: HTTP {e.response.status_code}"
            logger.error(error_msg, extra={"stream": "error"})
            yield error_msg
            return
        except Exception as e:
            error_msg = f"[ERROR] LLM stream failed: {e}"
            logger.error(error_msg, extra={"stream": "error"}, exc_info=True)
            yield error_msg
            return

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        logger.info(
            f"LLM stream complete: {latency_ms:.0f}ms, response_len={len(full_response)}",
            extra={
                "stream": "generation",
                "context": {
                    "latency_ms": latency_ms,
                    "model": self.model_name,
                    "response_length": len(full_response),
                    "streamed": True,
                },
            },
        )

        if settings.log_full_content:
            logger.debug(
                f"Full streamed response:\n{full_response}",
                extra={"stream": "generation"},
            )
