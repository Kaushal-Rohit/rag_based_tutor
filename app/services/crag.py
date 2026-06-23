"""
Corrective RAG (CRAG) Pipeline
================================
Three-stage agentic pipeline inserted between retrieval and generation:

  1. **Query Rewriter** — rewrites the user query into a retrieval-optimized form
  2. **Relevance Grader** — grades each retrieved chunk's relevance, drops failures
  3. **Hallucination Checker** — verifies the generated answer is grounded in chunks

All stages use the same Ollama LLM (via AsyncLLMEngine), keeping dependencies light.
The entire pipeline is toggleable via ``RAG_CRAG_ENABLED`` env var.
"""

import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.llm_engine import AsyncLLMEngine

logger = get_logger(__name__)


@dataclass
class CRAGResult:
    """Container for CRAG pipeline results."""
    rewritten_query: str = ""
    original_query: str = ""
    graded_chunks: list[str] = field(default_factory=list)
    dropped_chunks: int = 0
    groundedness: str = ""  # SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
    rewrite_latency_ms: float = 0.0
    grading_latency_ms: float = 0.0
    groundedness_latency_ms: float = 0.0
    retried: bool = False


class CRAGPipeline:
    """
    Corrective RAG pipeline using Ollama for all LLM calls.

    Can be disabled entirely via ``settings.crag_enabled``.
    """

    def __init__(self, llm_engine: AsyncLLMEngine):
        self.llm = llm_engine

    @property
    def enabled(self) -> bool:
        return settings.crag_enabled

    # ──────────────────────────────────────────
    # Stage 1: Query Rewriting
    # ──────────────────────────────────────────

    async def rewrite_query(self, query: str, history: str = "") -> tuple[str, float]:
        """
        Rewrite the user query into a retrieval-optimized form.

        Returns:
            tuple of (rewritten_query, latency_ms)
        """
        system = (
            "You are a query rewriting assistant for a textbook search system. "
            "Your job is to rewrite the student's question into an optimal search query "
            "for a vector database containing NCERT textbook content or uploaded material. "
            "Output ONLY the rewritten query, nothing else. "
            "Keep it concise and focused on key concepts. "
            "If the question clearly shifts to a completely new topic, IGNORE the history and just use the new topic. "
            "If the question uses pronouns or refers to the history, resolve them using the conversation history."
        )
        prompt = f"Conversation History:\n{history}\n\nRewrite this student question for vector search:\n\n{query}"

        t_start = time.perf_counter()
        rewritten = await self.llm.generate(prompt, system_instruction=system)
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Clean up — remove quotes, extra whitespace
        rewritten = rewritten.strip().strip('"').strip("'").strip()

        # Fallback: if rewrite fails or is empty, use original
        if not rewritten or len(rewritten) < 3:
            rewritten = query

        logger.info(
            f"CRAG rewrite: '{query[:50]}...' -> '{rewritten[:50]}...' "
            f"({latency_ms:.0f}ms)",
            extra={
                "stream": "retrieval",
                "context": {
                    "stage": "query_rewrite",
                    "original_query": query[:100],
                    "rewritten_query": rewritten[:100],
                    "latency_ms": latency_ms,
                },
            },
        )

        return rewritten, latency_ms

    # ──────────────────────────────────────────
    # Stage 2: Relevance Grading
    # ──────────────────────────────────────────

    async def grade_chunk(self, query: str, chunk: str) -> bool:
        """
        Grade whether a single chunk is relevant to the query.

        Returns True if relevant, False if not.
        """
        system = (
            "You are a relevance grading assistant. Given a student's question and a "
            "text chunk from a textbook, determine if the chunk is relevant to answering "
            "the question. Answer with ONLY 'YES' or 'NO'."
        )
        prompt = (
            f"Question: {query}\n\n"
            f"Text chunk: {chunk[:500]}\n\n"
            f"Is this chunk relevant? (YES/NO):"
        )

        response = await self.llm.generate(prompt, system_instruction=system)
        answer = response.strip().upper()
        return answer.startswith("YES")

    async def grade_chunks(
        self, query: str, chunks: list[str]
    ) -> tuple[list[str], int, float]:
        """
        Grade all retrieved chunks for relevance.

        Returns:
            tuple of (relevant_chunks, dropped_count, latency_ms)
        """
        t_start = time.perf_counter()
        relevant = []
        dropped = 0

        for i, chunk in enumerate(chunks):
            is_relevant = await self.grade_chunk(query, chunk)
            if is_relevant:
                relevant.append(chunk)
            else:
                dropped += 1
                logger.debug(
                    f"CRAG dropped chunk {i}: not relevant",
                    extra={"stream": "retrieval"},
                )

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Safety: if all chunks were dropped, keep the originals
        if not relevant and chunks:
            logger.warning(
                "CRAG grading dropped ALL chunks — keeping originals as fallback",
                extra={"stream": "retrieval"},
            )
            relevant = chunks
            dropped = 0

        logger.info(
            f"CRAG grading: {len(relevant)}/{len(chunks)} chunks passed "
            f"({dropped} dropped, {latency_ms:.0f}ms)",
            extra={
                "stream": "retrieval",
                "context": {
                    "stage": "relevance_grading",
                    "total_chunks": len(chunks),
                    "passed": len(relevant),
                    "dropped": dropped,
                    "latency_ms": latency_ms,
                },
            },
        )

        return relevant, dropped, latency_ms

    # ──────────────────────────────────────────
    # Stage 3: Hallucination / Groundedness Check
    # ──────────────────────────────────────────

    async def check_groundedness(
        self, answer: str, chunks: list[str]
    ) -> tuple[str, float]:
        """
        Check whether the generated answer is grounded in the retrieved chunks.

        Returns:
            tuple of (groundedness_verdict, latency_ms)
            verdict is one of: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED
        """
        context = "\n\n---\n\n".join(chunks[:5])  # Limit context size

        system = (
            "You are a fact-checking assistant. Given an AI-generated answer and the "
            "source context chunks it was based on, determine if the answer is supported "
            "by the context. Answer with ONLY one of: SUPPORTED, PARTIALLY_SUPPORTED, "
            "or NOT_SUPPORTED."
        )
        prompt = (
            f"Context chunks:\n{context}\n\n"
            f"AI-generated answer:\n{answer}\n\n"
            f"Verdict (SUPPORTED/PARTIALLY_SUPPORTED/NOT_SUPPORTED):"
        )

        t_start = time.perf_counter()
        response = await self.llm.generate(prompt, system_instruction=system)
        latency_ms = (time.perf_counter() - t_start) * 1000

        verdict = response.strip().upper()
        # Normalize to one of the three expected values
        if "NOT_SUPPORTED" in verdict or "NOT SUPPORTED" in verdict:
            verdict = "NOT_SUPPORTED"
        elif "PARTIALLY" in verdict:
            verdict = "PARTIALLY_SUPPORTED"
        else:
            verdict = "SUPPORTED"

        logger.info(
            f"CRAG groundedness check: {verdict} ({latency_ms:.0f}ms)",
            extra={
                "stream": "retrieval",
                "context": {
                    "stage": "groundedness_check",
                    "verdict": verdict,
                    "answer_length": len(answer),
                    "context_chunks": len(chunks),
                    "latency_ms": latency_ms,
                },
            },
        )

        return verdict, latency_ms

    # ──────────────────────────────────────────
    # Full CRAG pipeline orchestration
    # ──────────────────────────────────────────

    async def run(
        self,
        original_query: str,
        chunks: list[str],
        answer: str,
    ) -> CRAGResult:
        """
        Run the full CRAG pipeline: rewrite → grade → groundedness check.

        Note: Query rewriting should be called BEFORE retrieval.
        Grading and groundedness are called AFTER retrieval and generation.
        This method handles grading + groundedness (post-retrieval stages).
        """
        result = CRAGResult(original_query=original_query)

        if not self.enabled:
            result.graded_chunks = chunks
            result.groundedness = "SKIPPED"
            return result

        # Grade chunks
        graded, dropped, grade_latency = await self.grade_chunks(
            original_query, chunks
        )
        result.graded_chunks = graded
        result.dropped_chunks = dropped
        result.grading_latency_ms = grade_latency

        # Check groundedness
        verdict, ground_latency = await self.check_groundedness(answer, graded)
        result.groundedness = verdict
        result.groundedness_latency_ms = ground_latency

        return result
