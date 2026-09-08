"""Unit tests for Quantitative Retrieval Quality Evaluation (src/retrieval_eval.py)."""

import pytest
from src.retrieval_eval import (
    LABELLED_QUERIES,
    build_evaluation_report,
    default_retrieve,
    evaluate_dataset,
    evaluate_query,
    explain_improvement_strategies,
    inspect_failures,
)


class TestRetrievalEvaluation:
    """Test suite for recall@k, precision@k evaluation and failure inspection."""

    def test_evaluate_query_perfect_recall(self):
        """Test evaluate_query returns 1.0 recall when all relevant chunk IDs are retrieved."""
        item = {
            "query": "How can a learner reset their password?",
            "relevant_chunk_ids": {"chunk-1", "chunk-2"},
        }

        def mock_retrieve(query: str, k: int = 5):
            return [
                {"id": "chunk-1", "text": "pass reset"},
                {"id": "chunk-2", "text": "learner reset"},
                {"id": "chunk-3", "text": "unrelated"},
            ]

        res = evaluate_query(item, k=5, retrieve_fn=mock_retrieve)

        assert res["recall"] == 1.0
        assert res["precision"] == round(2 / 3, 4)
        assert res["hits"] == ["chunk-1", "chunk-2"]

    def test_evaluate_query_partial_recall(self):
        """Test evaluate_query calculates partial recall correctly."""
        item = {
            "query": "What evidence is required?",
            "relevant_chunk_ids": {"req-1", "req-2"},
        }

        def mock_retrieve(query: str, k: int = 5):
            return [{"id": "req-1", "text": "evidence 1"}]

        res = evaluate_query(item, k=5, retrieve_fn=mock_retrieve)

        assert res["recall"] == 0.5
        assert res["precision"] == 1.0
        assert res["hits"] == ["req-1"]

    def test_evaluate_query_zero_retrieved(self):
        """Test evaluate_query handles empty retrieval without division by zero."""
        item = {
            "query": "Unknown query",
            "relevant_chunk_ids": {"chunk-99"},
        }

        def mock_retrieve(query: str, k: int = 5):
            return []

        res = evaluate_query(item, k=5, retrieve_fn=mock_retrieve)

        assert res["recall"] == 0.0
        assert res["precision"] == 0.0
        assert res["hits"] == []

    def test_evaluate_dataset_aggregation(self):
        """Test dataset evaluation aggregates average recall and precision across queries."""
        dataset = [
            {"query": "q1", "relevant_chunk_ids": {"c1"}},
            {"query": "q2", "relevant_chunk_ids": {"c2"}},
        ]

        def mock_retrieve(query: str, k: int = 5):
            if query == "q1":
                return [{"id": "c1"}]
            return [{"id": "other"}]

        summary = evaluate_dataset(dataset, k=5, retrieve_fn=mock_retrieve)

        assert summary["total_queries"] == 2
        assert summary["avg_recall"] == 0.5
        assert summary["avg_precision"] == 0.5
        assert len(summary["failures"]) == 1
        assert summary["failures"][0]["query"] == "q2"

    def test_inspect_failures_diagnostics(self):
        """Test failure inspection correctly identifies missing chunk IDs and root causes."""
        failures = [
            {
                "query": "q2",
                "retrieved_ids": ["other1", "other2"],
                "relevant_chunk_ids": ["c2", "c3"],
                "recall": 0.0,
                "precision": 0.0,
            }
        ]

        diagnostics = inspect_failures(failures)

        assert len(diagnostics) == 1
        assert diagnostics[0]["query"] == "q2"
        assert diagnostics[0]["missing_ids"] == ["c2", "c3"]
        assert "Complete retrieval miss" in diagnostics[0]["root_cause_diagnosis"]

    def test_build_evaluation_report(self):
        """Test markdown report generation contains metric tables and strategy section."""
        summary = evaluate_dataset(LABELLED_QUERIES, k=5)
        report = build_evaluation_report(summary)

        assert "# Quantitative Retrieval Quality Evaluation Report" in report
        assert "Recall@5" in report
        assert "Precision@5" in report
        assert "Query-Level Evaluation Results" in report
        assert "Actionable Strategies to Improve Recall" in report
