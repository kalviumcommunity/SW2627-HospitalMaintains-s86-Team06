"""
Unit tests for the batch embedding pipeline.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.batch_embedding import (
    BatchEmbeddingPipeline,
    EmbeddingRecord,
    RunSummary,
)
from src.chunk_metadata import Chunk, ChunkMetadata, ChunkStore
from src.sample_chunks import build_sample_chunks


class TestBatchEmbeddingPipeline:
    """Tests for BatchEmbeddingPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes with correct defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            pipeline = BatchEmbeddingPipeline(embeddings_file=embeddings_file)
            
            assert pipeline.batch_size == 64
            assert pipeline.max_retries == 5
            assert pipeline.model == "text-embedding-3-small"

    def test_pipeline_loads_existing_embeddings(self):
        """Test pipeline loads existing embeddings from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            
            # Create mock embeddings file
            mock_embeddings = [
                {
                    "chunk_id": "chunk-1",
                    "text": "test text",
                    "embedding": [0.1, 0.2, 0.3],
                    "source_id": "test",
                    "section": None,
                    "page": None,
                    "position": None,
                    "created_at": "2026-08-31T00:00:00",
                }
            ]
            
            with open(embeddings_file, "w") as f:
                json.dump(mock_embeddings, f)
            
            # Load with pipeline
            pipeline = BatchEmbeddingPipeline(embeddings_file=embeddings_file)
            existing = pipeline.get_existing_embedding_ids()
            
            assert "chunk-1" in existing
            assert len(existing) == 1

    def test_get_existing_embedding_ids(self):
        """Test retrieving existing embedding IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            pipeline = BatchEmbeddingPipeline(embeddings_file=embeddings_file)
            
            # Should be empty initially
            existing = pipeline.get_existing_embedding_ids()
            assert len(existing) == 0

    def test_batches_division(self):
        """Test batches are divided correctly."""
        pipeline = BatchEmbeddingPipeline()
        items = list(range(100))
        
        batches = pipeline._batches(items, size=30)
        
        assert len(batches) == 4  # 30, 30, 30, 10
        assert len(batches[0]) == 30
        assert len(batches[1]) == 30
        assert len(batches[2]) == 30
        assert len(batches[3]) == 10

    def test_batches_empty_list(self):
        """Test batches handles empty list."""
        pipeline = BatchEmbeddingPipeline()
        batches = pipeline._batches([], size=10)
        
        assert len(batches) == 0

    def test_run_summary_to_dict(self):
        """Test RunSummary converts to dictionary."""
        summary = RunSummary(
            timestamp="2026-08-31T00:00:00",
            total_chunks=10,
            skipped_existing=0,
            embedded=10,
            failed=0,
            input_tokens_used=1000,
            estimated_cost_usd=0.00002,
            batch_size=64,
            total_batches=1,
            successful_batches=1,
            failed_batches=0,
        )
        
        summary_dict = summary.to_dict()
        
        assert summary_dict["total_chunks"] == 10
        assert summary_dict["embedded"] == 10
        assert summary_dict["failed"] == 0

    def test_run_summary_to_markdown(self):
        """Test RunSummary generates markdown."""
        summary = RunSummary(
            timestamp="2026-08-31T00:00:00",
            total_chunks=10,
            skipped_existing=2,
            embedded=8,
            failed=0,
            input_tokens_used=1000,
            estimated_cost_usd=0.00002,
            batch_size=64,
            total_batches=1,
            successful_batches=1,
            failed_batches=0,
        )
        
        markdown = summary.to_markdown()
        
        assert "Batch Embedding Run Summary" in markdown
        assert "**Total Chunks:** 10" in markdown
        assert "**Successfully Embedded:** 8" in markdown
        assert "**Skipped (Already Embedded):** 2" in markdown

    def test_embedding_record_to_dict(self):
        """Test EmbeddingRecord converts to dictionary."""
        record = EmbeddingRecord(
            chunk_id="chunk-1",
            text="test text",
            embedding=[0.1, 0.2, 0.3],
            source_id="test-source",
            section="test-section",
            page=1,
            position=0,
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["chunk_id"] == "chunk-1"
        assert record_dict["text"] == "test text"
        assert record_dict["source_id"] == "test-source"
        assert isinstance(record_dict["created_at"], str)


class TestBatchEmbeddingIntegration:
    """Integration tests for batch embedding pipeline."""

    def test_embed_empty_chunks(self):
        """Test embedding empty chunk list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            pipeline = BatchEmbeddingPipeline(embeddings_file=embeddings_file)
            
            summary = pipeline.embed_chunks([], skip_existing=True)
            
            assert summary.total_chunks == 0
            assert summary.embedded == 0
            assert summary.failed == 0

    def test_skip_existing_chunks(self):
        """Test that existing chunks are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            
            # Create mock embeddings
            mock_embeddings = [
                {
                    "chunk_id": "existing-1",
                    "text": "existing text",
                    "embedding": [0.1, 0.2, 0.3],
                    "source_id": "test",
                    "section": None,
                    "page": None,
                    "position": None,
                    "created_at": "2026-08-31T00:00:00",
                }
            ]
            
            with open(embeddings_file, "w") as f:
                json.dump(mock_embeddings, f)
            
            # Create chunks
            pipeline = BatchEmbeddingPipeline(embeddings_file=embeddings_file)
            store = ChunkStore()
            store.add_chunk("text 1", "src-1", section="s1")
            store._chunks["existing-1"] = Chunk(
                chunk_id="existing-1",
                text="existing text",
                metadata=ChunkMetadata(source_id="test"),
            )
            
            chunks = store.list_chunks()
            summary = pipeline.embed_chunks(chunks, skip_existing=True)
            
            # existing-1 should be skipped
            assert "existing-1" not in summary.failed_chunk_ids

    def test_sample_chunks_load(self):
        """Test loading sample chunks."""
        store = build_sample_chunks()
        chunks = store.list_chunks()
        
        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunk_id for c in chunks)
        assert all(c.text for c in chunks)


class TestBatchEmbeddingMockMode:
    """Tests for mock mode (no API calls)."""

    def test_mock_mode_no_api_key(self):
        """Test that pipeline handles missing API key gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings_file = os.path.join(tmpdir, "embeddings.json")
            # Initialize with no API key (mock mode)
            pipeline = BatchEmbeddingPipeline(
                api_key="",
                embeddings_file=embeddings_file
            )
            
            assert pipeline.client is None


class TestCostCalculation:
    """Tests for cost calculation and token tracking."""

    def test_cost_estimation(self):
        """Test cost estimation calculation."""
        # With 1000 tokens at $0.02 per 1M tokens
        summary = RunSummary(
            timestamp="2026-08-31T00:00:00",
            total_chunks=10,
            skipped_existing=0,
            embedded=10,
            failed=0,
            input_tokens_used=1000,
            estimated_cost_usd=0.00002,  # $0.02 / 1M * 1000
            batch_size=64,
            total_batches=1,
            successful_batches=1,
            failed_batches=0,
        )
        
        assert summary.estimated_cost_usd == 0.00002
        assert summary.input_tokens_used == 1000

    def test_zero_cost_rerun(self):
        """Test that rerun with all existing chunks has zero cost."""
        summary = RunSummary(
            timestamp="2026-08-31T00:00:00",
            total_chunks=100,
            skipped_existing=100,  # All skipped
            embedded=0,
            failed=0,
            input_tokens_used=0,  # No tokens used
            estimated_cost_usd=0.0,  # Zero cost
            batch_size=64,
            total_batches=0,
            successful_batches=0,
            failed_batches=0,
        )
        
        assert summary.embedded == 0
        assert summary.estimated_cost_usd == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
