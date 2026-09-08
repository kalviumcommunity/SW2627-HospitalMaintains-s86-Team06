"""Unit tests for Metadata Filtering and Hybrid Search (src/metadata_hybrid_search.py)."""

import pytest
from src.metadata_hybrid_search import (
    DEFAULT_METADATA_CORPUS,
    DEFAULT_METADATA_VECTORS,
    build_metadata_hybrid_report,
    hybrid_rank,
    keyword_score,
    retrieve,
    show_results,
)
from src.vector_db import VectorDBStore


class TestRetrieveMetadataFiltering:
    """Test suite for vector retrieval with metadata filtering."""

    def test_unfiltered_retrieval(self):
        """Test retrieving top-k records without metadata filter."""
        results = retrieve("What are the password reset steps?", k=3)
        assert len(results) == 3
        assert "score" in results[0]
        assert "text" in results[0]
        assert "metadata" in results[0]

    def test_metadata_filtered_retrieval(self):
        """Test metadata filter scopes results to matching section."""
        filter_dict = {"section": "Account access"}
        results = retrieve("What are the password reset steps?", k=3, metadata_filter=filter_dict)

        assert len(results) >= 1
        for item in results:
            assert item["metadata"]["section"] == "Account access"

    def test_metadata_filter_improves_precision(self):
        """Test metadata filter excludes unrelated sections that would otherwise rank in unfiltered results."""
        unfiltered = retrieve("What are the password reset steps?", k=5)
        filtered = retrieve("What are the password reset steps?", k=5, metadata_filter={"section": "Account access"})

        unfiltered_sections = {item["metadata"].get("section") for item in unfiltered}
        filtered_sections = {item["metadata"].get("section") for item in filtered}

        assert len(unfiltered_sections) > 1
        assert filtered_sections == {"Account access"}

    def test_retrieve_with_chromadb_collection(self):
        """Test retrieve function integrated with ChromaDB VectorCollection and where filter."""
        store = VectorDBStore(in_memory=True)
        col = store.create_collection(name="test_hybrid_col", dimension=8, metric="cosine")

        # Upsert test records
        for i, chunk in enumerate(DEFAULT_METADATA_CORPUS):
            col.upsert(
                {
                    "id": f"chunk-{i}",
                    "vector": DEFAULT_METADATA_VECTORS[i],
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

        # Query with collection and metadata filter
        results = retrieve(
            query="password reset",
            k=2,
            metadata_filter={"department": "IT Support"},
            collection=col,
            query_vector=[0.05, 0.02, 0.92, 0.04, 0.08, 0.02, 0.01, 0.05],
        )

        assert len(results) == 1
        assert results[0]["metadata"]["department"] == "IT Support"
        assert results[0]["metadata"]["section"] == "Account access"


class TestKeywordScoreAndHybridRank:
    """Test suite for lexical keyword scoring and hybrid ranking."""

    def test_keyword_score_basic(self):
        """Test exact keyword matching in text."""
        text = "To reset your password, visit IT support."
        keywords = ["password", "reset", "unrelated"]
        score = keyword_score(text, keywords)
        assert score == 2

    def test_keyword_score_case_insensitive(self):
        """Test keyword scoring is case-insensitive."""
        text = "Password Reset Instructions"
        keywords = ["PASSWORD", "reset"]
        score = keyword_score(text, keywords)
        assert score == 2

    def test_keyword_score_no_matches(self):
        """Test keyword score returns 0 when no terms match."""
        text = "Medication dosage instructions"
        keywords = ["password", "network", "router"]
        assert keyword_score(text, keywords) == 0

    def test_hybrid_rank_combination_and_sorting(self):
        """Test hybrid_rank calculates weighted score and sorts in descending order."""
        vector_results = [
            {
                "id": "rec-1",
                "score": 0.85,
                "text": "Generic security policy overview",
                "metadata": {"section": "Security policy"},
            },
            {
                "id": "rec-2",
                "score": 0.80,
                "text": "Self-service password reset guide for account access",
                "metadata": {"section": "Account access"},
            },
        ]
        keywords = ["password", "reset", "account"]

        # Rec-1 keyword score = 0, combined = 0.8 * 0.85 + 0.2 * 0 = 0.68
        # Rec-2 keyword score = 3, combined = 0.8 * 0.80 + 0.2 * 3 = 0.64 + 0.6 = 1.24
        hybrid = hybrid_rank(vector_results, keywords=keywords, vector_weight=0.8, keyword_weight=0.2)

        assert len(hybrid) == 2
        assert hybrid[0]["id"] == "rec-2"
        assert hybrid[0]["keyword_score"] == 3
        assert hybrid[0]["hybrid_score"] > hybrid[1]["hybrid_score"]

    def test_show_results_output(self, capsys):
        """Test show_results prints output without raising exceptions."""
        results = [
            {
                "score": 0.95,
                "text": "Sample password reset text",
                "metadata": {"source": "handbook.pdf", "section": "Account access"},
            }
        ]
        show_results("TEST LABEL", results)
        captured = capsys.readouterr()
        assert "TEST LABEL" in captured.out
        assert "Account access" in captured.out
        assert "0.95" in captured.out


class TestReportGeneration:
    """Test suite for Markdown report generation."""

    def test_build_metadata_hybrid_report(self):
        """Test that the generated report contains expected headings and sections."""
        unfiltered = retrieve("password reset", k=2)
        filtered = retrieve("password reset", k=2, metadata_filter={"section": "Account access"})
        hybrid = hybrid_rank(filtered, keywords=["password", "reset"])

        report = build_metadata_hybrid_report(
            query="password reset",
            unfiltered_results=unfiltered,
            filtered_results=filtered,
            hybrid_results=hybrid,
            metadata_filter={"section": "Account access"},
            keywords=["password", "reset"],
        )

        assert "# Metadata Filtering & Hybrid Search Report" in report
        assert "## Unfiltered Vector Retrieval" in report
        assert "## Metadata-Filtered Retrieval" in report
        assert "## Hybrid Lexical-Semantic Search" in report
        assert "## Trade-offs & Strategic Guidance" in report
