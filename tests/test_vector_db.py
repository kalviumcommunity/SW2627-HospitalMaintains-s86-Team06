"""Unit tests for Vector Database Setup and Collection Design (src/vector_db.py)."""

import tempfile
import pytest
from src.vector_db import (
    COLLECTION_NAME,
    VECTOR_DIMENSION,
    VectorCollection,
    VectorDBStore,
    VectorRecord,
)


@pytest.fixture
def vector_store():
    """Fixture providing an ephemeral in-memory VectorDBStore instance."""
    return VectorDBStore(in_memory=True)


@pytest.fixture
def default_collection(vector_store):
    """Fixture providing a standard vector collection with dimension 1536."""
    return vector_store.create_collection(
        name=COLLECTION_NAME,
        dimension=VECTOR_DIMENSION,
        metric="cosine",
    )


class TestVectorDBStore:
    """Test suite for VectorDBStore initialization and collection management."""

    def test_in_memory_initialization(self, vector_store):
        """Test creating an in-memory client."""
        assert vector_store.client is not None

    def test_persistent_initialization(self):
        """Test persistent storage initialization."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            store = VectorDBStore(persist_directory=tmp_dir, in_memory=False)
            collection = store.create_collection("test_persisted", dimension=8)
            collection.upsert(
                {
                    "id": "rec-1",
                    "vector": [0.1] * 8,
                    "text": "Persisted chunk",
                    "metadata": {"doc": "test.txt"},
                }
            )
            assert collection.count() == 1

    def test_create_and_get_collection(self, vector_store):
        """Test creating and getting collection."""
        col = vector_store.create_collection("custom_col", dimension=128)
        assert col.name == "custom_col"
        assert col.dimension == 128

        retrieved = vector_store.get_collection("custom_col", dimension=128)
        assert retrieved.name == "custom_col"

    def test_delete_collection(self, vector_store):
        """Test deleting a collection."""
        vector_store.create_collection("temp_col", dimension=8)
        vector_store.delete_collection("temp_col")
        with pytest.raises(Exception):
            vector_store.get_collection("temp_col")


class TestVectorCollection:
    """Test suite for VectorCollection operations."""

    def test_upsert_and_readback_record(self, default_collection):
        """Test upserting and reading back a 1536-dimensional record."""
        vector = [0.05] * VECTOR_DIMENSION
        record = {
            "id": "account-guide.md:0",
            "vector": vector,
            "text": "Password reset instructions for learner accounts.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 0,
                "section": "Account access",
            },
        }

        default_collection.upsert([record])
        stored = default_collection.get("account-guide.md:0")

        assert stored is not None
        assert stored["id"] == "account-guide.md:0"
        assert len(stored["vector"]) == VECTOR_DIMENSION
        assert stored["text"] == "Password reset instructions for learner accounts."
        assert stored["metadata"]["source"] == "account-guide.md"
        assert stored["metadata"]["chunk_index"] == 0
        assert stored["metadata"]["section"] == "Account access"

    def test_upsert_vector_record_dataclass(self, default_collection):
        """Test upserting using VectorRecord dataclass."""
        rec = VectorRecord(
            id="doc1:0",
            vector=[0.01] * VECTOR_DIMENSION,
            text="Clinical protocol for triage.",
            metadata={"source": "triage.pdf"},
        )
        default_collection.upsert(rec)
        stored = default_collection.get("doc1:0")

        assert stored is not None
        assert stored["id"] == "doc1:0"
        assert stored["text"] == "Clinical protocol for triage."

    def test_dimension_validation_failure(self, default_collection):
        """Test that invalid vector dimensions fail early with ValueError."""
        invalid_record = {
            "id": "bad-chunk:0",
            "vector": [0.1, 0.2, 0.3],  # 3 elements instead of 1536
            "text": "Invalid vector dimension text.",
            "metadata": {"source": "bad.txt"},
        }

        with pytest.raises(ValueError) as exc_info:
            default_collection.upsert([invalid_record])

        assert "Vector dimension mismatch" in str(exc_info.value)
        assert f"expected {VECTOR_DIMENSION}, got 3" in str(exc_info.value)

    def test_query_nearest_neighbors(self, vector_store):
        """Test querying nearest neighbors by vector similarity."""
        col = vector_store.create_collection("query_test", dimension=4, metric="cosine")

        col.upsert(
            [
                {
                    "id": "vec-1",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "text": "Medication guidance text",
                    "metadata": {"category": "medication"},
                },
                {
                    "id": "vec-2",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "text": "Password reset IT support text",
                    "metadata": {"category": "it_support"},
                },
            ]
        )

        # Query vector close to vec-1
        results = col.query(query_vector=[0.9, 0.1, 0.0, 0.0], n_results=1)

        assert len(results) == 1
        assert results[0]["id"] == "vec-1"
        assert results[0]["text"] == "Medication guidance text"

    def test_get_nonexistent_record(self, default_collection):
        """Test reading back a non-existent record returns None."""
        assert default_collection.get("non-existent-id") is None
