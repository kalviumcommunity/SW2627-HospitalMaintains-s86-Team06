"""Unit tests for RAG Candidate Retrieval and Re-ranking (src/reranking.py)."""

import pytest
from src.reranking import (
    build_reranking_report,
    call_reranker_model,
    explain_cost_latency_tradeoffs,
    rerank_candidates,
    rerank_score,
    retrieve_candidates,
    show,
)


class TestRerankingPipeline:
    """Test suite for candidate retrieval, re-ranking scoring, and ordering."""

    def test_retrieve_candidates_default(self):
        """Test candidate retrieval returns specified pool size k."""
        query = "What evidence is required for project submission?"
        candidates = retrieve_candidates(query, k=10)
        assert len(candidates) == 10
        # Check candidates are sorted by initial vector score descending
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_score_with_custom_model_fn(self):
        """Test rerank_score invokes model scoring with expected prompt formatting."""
        received_prompts = []

        def mock_call_model(prompt: str) -> float:
            received_prompts.append(prompt)
            return 9.5

        chunk = {"text": "Required evidence includes security audit and sign-off certificates."}
        score = rerank_score("What evidence is required?", chunk, call_model_fn=mock_call_model)

        assert score == 9.5
        assert len(received_prompts) == 1
        assert "Query: What evidence is required?" in received_prompts[0]
        assert "Chunk: Required evidence includes security audit" in received_prompts[0]
        assert "Score how relevant this chunk is to the query from 0 to 10." in received_prompts[0]

    def test_rerank_candidates_changes_top_ordering(self):
        """Test re-ranking moves the most relevant chunk to rank 1 over higher initial vector scores."""
        query = "What evidence is required for project submission?"
        candidates = retrieve_candidates(query, k=10)

        # Confirm initial vector rank 1 is a general overview chunk
        assert "Overview" in candidates[0]["metadata"]["section"]

        # Run re-ranking
        final_k = 3
        final_context = rerank_candidates(query, candidates, final_k=final_k)

        assert len(final_context) == final_k
        # After re-ranking, rank 1 should be the chunk detailing exact required evidence
        top_chunk = final_context[0]
        assert top_chunk["rerank_score"] > candidates[0].get("rerank_score", 0.0)
        assert "Submission Evidence" in top_chunk["metadata"]["section"]
        assert "Required evidence" in top_chunk["text"]

    def test_show_output_formatting(self, capsys):
        """Test show function prints expected formatted fields."""
        rows = [
            {
                "score": 0.85,
                "rerank_score": 9.2,
                "metadata": {"source": "test-doc.pdf"},
                "text": "This is sample chunk text for testing output formatting.",
            }
        ]

        show("before re-ranking", rows)
        captured = capsys.readouterr().out

        assert "before re-ranking" in captured
        assert "rank: 1" in captured
        assert "vector_score: 0.85" in captured
        assert "rerank_score: 9.2" in captured
        assert "source: test-doc.pdf" in captured
        assert "text: This is sample chunk text" in captured

    def test_explain_cost_latency_tradeoffs(self):
        """Test trade-off breakdown returns expected structure and metrics."""
        tradeoffs = explain_cost_latency_tradeoffs()
        assert "retrieval_stage" in tradeoffs
        assert "reranking_stage" in tradeoffs
        assert "best_practice_recommendation" in tradeoffs

        assert "Fast" in tradeoffs["retrieval_stage"]["latency"]
        assert "Higher" in tradeoffs["reranking_stage"]["latency"]
        assert "10 to 20" in tradeoffs["best_practice_recommendation"]["candidate_set_size_k"]
        assert "3 to 5" in tradeoffs["best_practice_recommendation"]["final_context_size_k"]

    def test_build_reranking_report(self):
        """Test markdown report generation includes tables and analysis sections."""
        query = "What evidence is required for project submission?"
        candidates = retrieve_candidates(query, k=5)
        final_context = rerank_candidates(query, candidates, final_k=3)

        report = build_reranking_report(
            query=query,
            candidates=candidates,
            final_context=final_context,
            final_k=3,
        )

        assert "# RAG Retrieval & Re-ranking Analysis Report" in report
        assert "Initial Vector Retrieval (Before Re-ranking)" in report
        assert "Re-ranked Results (After Re-ranking)" in report
        assert "Cost & Latency Trade-off Analysis" in report
        assert "| Rank | Vector Score | Source Document | Section | Text Preview |" in report
