# Adaptive RAG Pipeline: Live Demo Script

This script provides a step-by-step walkthrough for presenting the Adaptive RAG Pipeline during a technical interview or live demo.

## Pre-Flight Checklist
Before the demo begins, ensure the following steps have been completed:
1. `ollama serve` is running in the background.
2. API Server is up: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Run the automated smoke test to verify end-to-end functionality:
   ```bash
   python scripts/demo_smoke_test.py
   ```
   *Wait for the `[SUCCESS] SMOKE TEST PASSED!` message before starting the demo.*

---

## 1. Introduction & The Goal
**Goal:** Introduce the problem space—Standard RAG is static, often hallucinates when confused, and is slow to start up.
**Action:** Open the **Notebook** UI ([http://localhost:8000/notebook](http://localhost:8000/notebook)).
**Talking Points:**
- "This is an Adaptive RAG Pipeline designed for educational tutoring. Instead of just dumping context into an LLM, it actively grades relevance (Corrective RAG) and adjusts how much context it provides based on user sentiment."
- "The UI is intentionally minimalist to focus on the chat and the architecture."

## 2. Architecture & Rationale Review
**Goal:** Prove architectural intent and production readiness.
**Action:** Click the **"Architecture Rationale"** link at the top right of the notebook, which opens `http://localhost:8000/report`.
**Talking Points:**
- Walk through the 9 core decisions.
- **Example 1 (Backend):** "We use ChromaDB as the primary vector store for exact metadata pre-filtering, preventing 0-result post-filter drops. But we retain FAISS for raw speed (~0.3ms vs ~44ms) when needed."
- **Example 2 (CRAG):** "We implemented a lightweight 3-stage Corrective RAG pipeline (Rewrite -> Grade -> Generate) without heavy frameworks like LangChain, keeping dependencies minimal and reducing hallucinations."
- **Example 3 (Async & Lifespan):** "We transitioned from a sync CLI to an async FastAPI app using a lifespan context manager. Models load exactly once at startup, reducing per-request overhead from 3-4 seconds down to 0ms."

## 3. The Live Notebook (PDF Upload & RAG)
**Goal:** Demonstrate the working pipeline live.
**Action:** Return to the Notebook. Click **"Upload PDF"** and select a sample document.
**Talking Points:**
- "The system supports incremental indexing. When a document is uploaded, it synchronously validates it, accepts it, and pushes it to a background task for chunking and embedding."
- Watch the indexing progress indicator complete.

**Action:** Ask a question specifically answered by the uploaded PDF.
**Talking Points:**
- As the answer streams in via Server-Sent Events (SSE), explain: "The answer is streaming dynamically. Under the hood, it's grading the retrieved chunks and citing the specific page of the uploaded PDF."
- Show the citation chip appearing at the bottom of the response.

## 4. Closing & Metrics
**Goal:** End on a strong, data-driven note.
**Action:** Scroll down on the Rationale Report to the **Metrics** section, or refer to the logged benchmark data.
**Talking Points:**
- "By caching embeddings, we save ~21ms per cache hit."
- "Everything is containerized and logged with structured JSON and contextvars for request tracing, making it ready for AWS deployment."
- "Any questions on the implementation?"
