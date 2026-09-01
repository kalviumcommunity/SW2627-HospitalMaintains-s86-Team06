"""Unit tests for Corpus Record Indexing & Verification (src/corpus_indexer.py)."""

import pytest
from src.corpus_indexer import CorpusIndexer, batches, to_vector_record
from src.vector_db import COLLECTION_NAME, VECTOR_DIMENSION, VectorDBStore


@pytest.fixture
def test_collection():
    """Fixture providing an ephemeral in-memory vector collection."""
    store = VectorDBStore(in_memory=True)
    return store.create_collection("indexer_test_col", dimension=8, metric="cosine")


class TestCorpusIndexerUtils:
    """Test suite for record transformation and batching helpers."""

    def test_batches_generator(self):
        """Test batches generator divides items correctly."""
        items = list(range(25))
        batch_list = list(batches(items, size=10))

        assert len(batch_list) == 3  # 10, 10, 5
        assert len(batch_list[0]) == 10
        assert len(batch_list[1]) == 10
        assert len(batch_list[2]) == 5

    def test_to_vector_record_formatting(self):
        """Test transforming embedded chunk into vector record dict."""
        chunk = {
            "id": "doc1:0",
            "embedding": [0.1] * 8,
            "text": "Cardiology protocol content.",
            "metadata": {
                "source": "cardiology.md",
                "chunk_index": 0,
                "section": "Triage",
                "department": "Cardiology",
            },
        }

        record = to_vector_record(chunk)

        assert record["id"] == "doc1:0"
        assert len(record["vector"]) == 8
        assert record["text"] == "Cardiology protocol content."
        assert record["metadata"]["source"] == "cardiology.md"
        assert record["metadata"]["chunk_index"] == 0
        assert record["metadata"]["section"] == "Triage"
        assert record["metadata"]["department"] == "Cardiology"

    def test_to_vector_record_missing_id_raises(self):
        """Test ValueError is raised if chunk is missing id."""
        with pytest.raises(ValueError):
            to_vector_record({"embedding": [0.1] * 8, "text": "No ID"})

    def test_to_vector_record_missing_embedding_raises(self):
        """Test ValueError is raised if chunk is missing embedding vector."""
        with pytest.raises(ValueError):
            to_vector_record({"id": "doc:0", "text": "No embedding"})


class TestCorpusIndexer:
    """Test suite for CorpusIndexer batch insertion and spot checks."""

    def test_index_chunks_and_count_validation(self, test_collection):
        """Test bulk indexing and verifying indexed count matches expected chunk count."""
        chunks = [
            {
                "id": f"chunk:{i}",
                "embedding": [0.01 * i] * 8,
                "text": f"Clinical text content for chunk {i}",
                "metadata": {"source": "policy.txt", "chunk_index": i},
            }
            for i in range(15)
        ]

        indexer = CorpusIndexer(collection=test_collection)
        summary = indexer.index_chunks(chunks, batch_size=5)

        assert summary.expected_count == 15
        assert summary.inserted_this_run == 15
        assert summary.indexed_count == 15
        assert summary.total_batches == 3
        assert summary.count_matches
        assert len(summary.failures) == 0

    def test_spot_check_integrity(self, test_collection):
        """Test spot-checking stored record against original source chunk."""
        sample_chunk = {
            "id": "account-guide.md:0",
            "embedding": [0.2] * 8,
            "text": "Password reset instructions for learner accounts.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 0,
                "section": "Account access",
            },
        }

        indexer = CorpusIndexer(collection=test_collection)
        indexer.index_chunks([sample_chunk], batch_size=10)

        spot_result = indexer.spot_check(sample_chunk)

        assert spot_result["spot_check_passed"]
        assert spot_result["id"] == "account-guide.md:0"
        assert spot_result["source"] == "account-guide.md"
        assert spot_result["vector_length"] == 8
        assert "Password reset" in spot_result["text_preview"]

    def test_spot_check_nonexistent_raises(self, test_collection):
        """Test spot check raises ValueError if chunk ID is not found."""
        indexer = CorpusIndexer(collection=test_collection)
        with pytest.raises(ValueError):
            indexer.spot_check({"id": "missing:0", "embedding": [0.1] * 8, "text": "Missing"})
