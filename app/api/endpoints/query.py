"""
Query Endpoint
==============
``POST /api/v1/query`` — main RAG query endpoint.

Supports both:
  - Standard JSON response (non-streaming)
  - Server-Sent Events (SSE) streaming for real-time token delivery

Pipeline:
  1. Generate request_id
  2. Resolve session for conversation continuity
  3. Compute rolling sentiment → dynamic k
  4. (CRAG) Rewrite query for optimal retrieval
  5. Run retrieval in thread pool (FAISS/Chroma are blocking)
  6. (CRAG) Grade chunk relevance
  7. Build prompt with history + context + query
  8. Generate response via Ollama (streaming or blocking)
  9. (CRAG) Check groundedness
  10. Update conversation history + metrics
"""

import asyncio
import hashlib
import json
import time
import uuid

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.logging_config import get_logger, request_id_ctx
from app.models.schemas import QueryRequest, QueryResponse, SourceCitation

logger = get_logger(__name__)
router = APIRouter()


def _build_prompt(history_str: str, context_str: str, query: str) -> str:
    """Build the full LLM prompt with conversation history and context."""
    return (
        f"**Conversation History (Last 5 turns):**\n{history_str}\n\n"
        f"**Retrieved Knowledge Chunks:**\n{context_str}\n\n"
        f"**Current User Query:**\n{query}\n\n"
        f"**Your Answer:**\n"
    )


@router.post(
    "/query",
    summary="RAG query",
    description="Main RAG endpoint. Processes a query through the full "
                "adaptive pipeline with sentiment-driven retrieval and optional CRAG.",
)
async def query(request: Request, body: QueryRequest):
    """
    Process a user query through the full adaptive RAG pipeline.

    If the ``Accept`` header includes ``text/event-stream``, returns an SSE stream.
    Otherwise returns a standard JSON response.
    """
    req_id = str(uuid.uuid4())
    request_id_ctx.set(req_id)
    t_start = time.perf_counter()

    # Resolve components from app state
    app = request.app
    retriever = app.state.retriever
    llm_engine = app.state.llm_engine
    crag = app.state.crag
    session_mgr = app.state.session_manager
    k_selector = app.state.k_selector
    metrics = app.state.metrics

    # Session management
    session_id = body.session_id or str(uuid.uuid4())
    convo = session_mgr.get_or_create(session_id)

    # ── 1. Sentiment → dynamic k ──
    avg_sentiment = convo.get_average_sentiment(body.query)
    k, user_state, system_instruction = k_selector.select(avg_sentiment)

    logger.info(
        f"Query: sentiment={avg_sentiment:.2f}, state={user_state.value}, k={k}",
        extra={
            "stream": "retrieval",
            "context": {
                "session_id": session_id,
                "sentiment": avg_sentiment,
                "user_state": user_state.value,
                "k": k,
                "query_hash": hashlib.sha256(body.query.encode()).hexdigest()[:12],
                "query_length": len(body.query),
            },
        },
    )

    # ── 2. (CRAG) Query rewriting ──
    search_query = body.query
    crag_applied = False
    if crag.enabled:
        crag_applied = True
        # Pass conversation history to rewriter so it can detect topic shifts and resolve pronouns
        history_str_for_rewrite = convo.get_formatted_history()
        search_query, rewrite_latency = await crag.rewrite_query(body.query, history=history_str_for_rewrite)

    # ── 3. Retrieval (in thread pool — blocking C++/SQLite calls) ──
    filters = body.filters.to_filter_dict() if body.filters else {}
    # Inject session scoping: search NCERT corpus OR the current session's uploads
    filters["$or"] = [
        {"source_type": "ncert"},
        {"session_id": session_id}
    ]
    
    t_retrieval_start = time.perf_counter()

    results = await asyncio.to_thread(
        retriever.search, search_query, k=k, filters=filters
    )
    t_retrieval_end = time.perf_counter()
    retrieval_latency_ms = (t_retrieval_end - t_retrieval_start) * 1000

    # Extract documents and metadata from results
    chunks = results.get("documents", [])
    chunk_metas = results.get("metadatas", [])
    chunk_ids = results.get("ids", [])
    result_latency = results.get("latency", {})

    logger.info(
        f"Retrieval complete: {len(chunks)} chunks in {retrieval_latency_ms:.2f}ms",
        extra={
            "stream": "retrieval",
            "context": {
                "sentiment": avg_sentiment,
                "k": k,
                "backend": settings.retrieval_backend,
                "chunks_returned": len(chunks),
                "embed_ms": result_latency.get("embedding_ms", 0),
                "search_ms": result_latency.get("search_ms", 0),
                "total_ms": retrieval_latency_ms,
                "filters": filters,
            },
        },
    )

    # ── 4. (CRAG) Relevance grading ──
    if crag_applied and chunks:
        graded_chunks, dropped, _ = await crag.grade_chunks(body.query, chunks)
        chunks = graded_chunks

    # ── 5. Build prompt ──
    context_str = "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."
    history_str = convo.get_formatted_history()
    prompt = _build_prompt(history_str, context_str, body.query)

    # ── 6. Check if SSE streaming is requested ──
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return _stream_response(
            request, prompt, system_instruction, body.query,
            session_id, convo, crag, chunks, crag_applied,
            avg_sentiment, user_state, k, retrieval_latency_ms,
            req_id, metrics, t_start,
        )

    # ── 7. Standard (non-streaming) generation ──
    response_text = await llm_engine.generate(prompt, system_instruction)

    # ── 8. (CRAG) Groundedness check ──
    disclaimer = ""
    if crag_applied and chunks and response_text:
        crag_result = await crag.run(body.query, chunks, response_text)
        if crag_result.groundedness == "NOT_SUPPORTED":
            # One retry with k+2
            logger.warning(
                "Groundedness check failed — retrying with k+2",
                extra={"stream": "retrieval"},
            )
            retry_results = await asyncio.to_thread(
                retriever.search, search_query, k=k + 2, filters=filters
            )
            retry_chunks = retry_results.get("documents", [])
            if retry_chunks:
                retry_context = "\n\n---\n\n".join(retry_chunks)
                retry_prompt = _build_prompt(history_str, retry_context, body.query)
                response_text = await llm_engine.generate(
                    retry_prompt, system_instruction
                )
                # Re-check
                retry_crag = await crag.run(body.query, retry_chunks, response_text)
                if retry_crag.groundedness == "NOT_SUPPORTED":
                    disclaimer = (
                        "\n\n⚠️ *This answer may not be fully grounded "
                        "in the textbook content.*"
                    )
        elif crag_result.groundedness == "PARTIALLY_SUPPORTED":
            disclaimer = (
                "\n\n⚠️ *Parts of this answer may extend beyond "
                "the available textbook content.*"
            )

    final_answer = response_text + disclaimer

    # ── 9. Build source citations ──
    sources = []
    for i, meta in enumerate(chunk_metas):
        cid = chunk_ids[i] if i < len(chunk_ids) else f"chunk_{i}"
        sources.append(SourceCitation(
            filename=meta.get("file_name") or meta.get("source_file", ""),
            page=meta.get("page_number"),
            chunk_id=cid,
            source_type=meta.get("source_type", "static"),
        ))

    # ── 10. Update conversation + metrics ──
    convo.add_turn(body.query, final_answer)
    metrics["total_queries"] += 1
    metrics["retrieval_latencies"].append(retrieval_latency_ms)
    metrics["sentiment_distribution"][user_state.value] += 1

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Request complete: {total_ms:.0f}ms total",
        extra={
            "stream": "access",
            "context": {
                "method": "POST",
                "path": "/api/v1/query",
                "status_code": 200,
                "total_latency_ms": total_ms,
            },
        },
    )

    return QueryResponse(
        answer=final_answer,
        session_id=session_id,
        sentiment_score=round(avg_sentiment, 4),
        user_state=user_state,
        k_used=k,
        chunks_retrieved=len(chunks),
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        crag_applied=crag_applied,
        request_id=req_id,
        sources=[s.model_dump() for s in sources],
    )


def _stream_response(
    request, prompt, system_instruction, query,
    session_id, convo, crag, chunks, crag_applied,
    avg_sentiment, user_state, k, retrieval_latency_ms,
    req_id, metrics, t_start,
):
    """Build an SSE EventSourceResponse that streams LLM tokens."""

    async def event_generator():
        llm_engine = request.app.state.llm_engine
        full_response = ""

        # Send metadata event first
        yield {
            "event": "metadata",
            "data": json.dumps({
                "request_id": req_id,
                "session_id": session_id,
                "sentiment_score": round(avg_sentiment, 4),
                "user_state": user_state.value,
                "k_used": k,
                "chunks_retrieved": len(chunks),
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "crag_applied": crag_applied,
            }),
        }

        # Stream tokens
        async for token in llm_engine.generate_stream(prompt, system_instruction):
            full_response += token
            yield {"event": "token", "data": token}

        # Post-stream: update conversation + metrics
        convo.add_turn(query, full_response)
        metrics["total_queries"] += 1
        metrics["retrieval_latencies"].append(retrieval_latency_ms)
        metrics["sentiment_distribution"][user_state.value] += 1

        total_ms = (time.perf_counter() - t_start) * 1000

        # Send completion event
        yield {
            "event": "done",
            "data": json.dumps({
                "total_latency_ms": round(total_ms, 2),
                "response_length": len(full_response),
            }),
        }

        logger.info(
            f"SSE stream complete: {total_ms:.0f}ms total",
            extra={"stream": "access"},
        )

    return EventSourceResponse(event_generator())
