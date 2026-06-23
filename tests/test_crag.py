"""
Unit Tests: CRAG Pipeline
===========================
Tests for query rewriting, relevance grading, and groundedness checking
with mocked Ollama responses.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.crag import CRAGPipeline, CRAGResult


@pytest.fixture
def mock_llm():
    """Create a mock AsyncLLMEngine."""
    llm = AsyncMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def crag(mock_llm):
    """Create a CRAGPipeline with mock LLM."""
    return CRAGPipeline(mock_llm)


class TestQueryRewriter:
    """Tests for the query rewriting stage."""

    @pytest.mark.asyncio
    async def test_rewrite_returns_different_query(self, crag, mock_llm):
        mock_llm.generate.return_value = "electromagnetic induction Faraday's law physics"
        rewritten, latency = await crag.rewrite_query(
            "What is electromagnetic induction?"
        )
        assert rewritten != ""
        assert latency >= 0
        assert "electromagnetic" in rewritten.lower()

    @pytest.mark.asyncio
    async def test_rewrite_strips_quotes(self, crag, mock_llm):
        mock_llm.generate.return_value = '"optimized search query"'
        rewritten, _ = await crag.rewrite_query("test query")
        assert not rewritten.startswith('"')
        assert not rewritten.endswith('"')

    @pytest.mark.asyncio
    async def test_rewrite_fallback_on_empty_response(self, crag, mock_llm):
        mock_llm.generate.return_value = ""
        rewritten, _ = await crag.rewrite_query("original query")
        assert rewritten == "original query"

    @pytest.mark.asyncio
    async def test_rewrite_fallback_on_short_response(self, crag, mock_llm):
        mock_llm.generate.return_value = "ab"
        rewritten, _ = await crag.rewrite_query("original query")
        assert rewritten == "original query"


class TestRelevanceGrader:
    """Tests for the relevance grading stage."""

    @pytest.mark.asyncio
    async def test_grade_relevant_chunk(self, crag, mock_llm):
        mock_llm.generate.return_value = "YES"
        result = await crag.grade_chunk("What is gravity?", "Gravity is a force...")
        assert result is True

    @pytest.mark.asyncio
    async def test_grade_irrelevant_chunk(self, crag, mock_llm):
        mock_llm.generate.return_value = "NO"
        result = await crag.grade_chunk("What is gravity?", "Cooking recipes...")
        assert result is False

    @pytest.mark.asyncio
    async def test_grade_chunks_filters_irrelevant(self, crag, mock_llm):
        # First two calls return YES, third returns NO
        mock_llm.generate.side_effect = ["YES", "YES", "NO"]
        chunks = ["relevant 1", "relevant 2", "irrelevant"]
        graded, dropped, latency = await crag.grade_chunks("test query", chunks)
        assert len(graded) == 2
        assert dropped == 1

    @pytest.mark.asyncio
    async def test_grade_chunks_keeps_all_when_all_dropped(self, crag, mock_llm):
        """Safety fallback: if all chunks are dropped, keep originals."""
        mock_llm.generate.side_effect = ["NO", "NO"]
        chunks = ["chunk1", "chunk2"]
        graded, dropped, _ = await crag.grade_chunks("test", chunks)
        assert len(graded) == 2  # originals kept
        assert dropped == 0  # reset because of fallback


class TestGroundednessChecker:
    """Tests for the hallucination / groundedness check."""

    @pytest.mark.asyncio
    async def test_supported_verdict(self, crag, mock_llm):
        mock_llm.generate.return_value = "SUPPORTED"
        verdict, latency = await crag.check_groundedness(
            "The force is F=ma", ["Newton's second law: F=ma"]
        )
        assert verdict == "SUPPORTED"

    @pytest.mark.asyncio
    async def test_not_supported_verdict(self, crag, mock_llm):
        mock_llm.generate.return_value = "NOT_SUPPORTED"
        verdict, _ = await crag.check_groundedness(
            "Made up answer", ["Unrelated chunk"]
        )
        assert verdict == "NOT_SUPPORTED"

    @pytest.mark.asyncio
    async def test_partially_supported_verdict(self, crag, mock_llm):
        mock_llm.generate.return_value = "PARTIALLY_SUPPORTED"
        verdict, _ = await crag.check_groundedness(
            "Partial answer", ["Some context"]
        )
        assert verdict == "PARTIALLY_SUPPORTED"

    @pytest.mark.asyncio
    async def test_normalizes_variant_text(self, crag, mock_llm):
        """LLM might return 'NOT SUPPORTED' (with space) — should normalize."""
        mock_llm.generate.return_value = "NOT SUPPORTED - the answer is not in context"
        verdict, _ = await crag.check_groundedness("answer", ["chunk"])
        assert verdict == "NOT_SUPPORTED"


class TestCRAGPipelineFull:
    """Tests for the full CRAG pipeline orchestration."""

    @pytest.mark.asyncio
    async def test_run_returns_crag_result(self, crag, mock_llm):
        # grade_chunks: all YES, groundedness: SUPPORTED
        mock_llm.generate.side_effect = ["YES", "YES", "SUPPORTED"]
        result = await crag.run(
            original_query="What is F=ma?",
            chunks=["Force equals mass times acceleration", "Newton's second law"],
            answer="F=ma means force equals mass times acceleration.",
        )
        assert isinstance(result, CRAGResult)
        assert len(result.graded_chunks) == 2
        assert result.groundedness == "SUPPORTED"

    @pytest.mark.asyncio
    @patch("app.services.crag.settings")
    async def test_skips_when_disabled(self, mock_settings, crag, mock_llm):
        mock_settings.crag_enabled = False
        result = await crag.run("query", ["chunk"], "answer")
        assert result.groundedness == "SKIPPED"
        assert result.graded_chunks == ["chunk"]
        mock_llm.generate.assert_not_called()
