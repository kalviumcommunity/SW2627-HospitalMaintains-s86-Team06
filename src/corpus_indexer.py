"""Corpus Record Indexing and Verification Module.

Handles batch indexing of embedded corpus chunks into vector database collections,
verifies indexed counts against expected counts, and performs record spot-checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

try:
    from src.vector_db import VectorCollection
except (ModuleNotFoundError, ImportError):
    from vector_db import VectorCollection

logger = logging.getLogger(__name__)


@dataclass
class IndexingSummary:
    """Summary of batch corpus indexing operation."""

    expected_count: int
    inserted_this_run: int
    indexed_count: int
    batch_size: int
    total_batches: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    spot_check_passed: bool = False

    @property
    def count_matches(self) -> bool:
        return self.indexed_count == self.expected_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_count": self.expected_count,
            "inserted_this_run": self.inserted_this_run,
            "indexed_count": self.indexed_count,
            "batch_size": self.batch_size,
            "total_batches": self.total_batches,
            "count_matches": self.count_matches,
            "failures": self.failures,
            "spot_check_passed": self.spot_check_passed,
        }


def batches(items: Sequence[Any], size: int = 100) -> Generator[Sequence[Any], None, None]:
    """Yield successive batches of given size from items sequence."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def to_vector_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Transform an embedded chunk dict into a standardized vector database record.

    Args:
        chunk: Dict containing id (or chunk_id), vector (or embedding), text, and metadata.

    Returns:
        Standardized dict with keys: id, vector, text, metadata.
    """
    rec_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    if not rec_id:
        raise ValueError("Chunk must contain a valid 'id' or 'chunk_id'")

    vector = chunk.get("vector") or chunk.get("embedding")
    if vector is None:
        raise ValueError(f"Chunk '{rec_id}' missing 'vector' or 'embedding'")

    text = chunk.get("text", "")
    metadata = dict(chunk.get("metadata", {}))

    # Standardize metadata schema fields
    source = metadata.get("source") or metadata.get("source_document") or metadata.get("source_id") or "unknown"
    chunk_index = metadata.get("chunk_index", 0)
    section = metadata.get("section")

    formatted_meta: Dict[str, Any] = {
        "source": str(source),
        "chunk_index": int(chunk_index),
    }
    if section is not None:
        formatted_meta["section"] = str(section)

    # Preserve any additional custom metadata key-values
    for k, v in metadata.items():
        if k not in formatted_meta and isinstance(v, (str, int, float, bool)):
            formatted_meta[k] = v

    return {
        "id": rec_id,
        "vector": [float(x) for x in vector],
        "text": text,
        "metadata": formatted_meta,
    }


class CorpusIndexer:
    """Manages batch indexing, verification, and spot-checking of embedded chunks."""

    def __init__(self, collection: VectorCollection) -> None:
        """Initialize with target vector collection.

        Args:
            collection: VectorCollection instance to index chunks into.
        """
        self.collection = collection

    def index_chunks(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> IndexingSummary:
        """Index embedded corpus chunks in batches and verify counts.

        Args:
            embedded_chunks: List of embedded chunk dictionaries.
            batch_size: Batch size for database upserts (default: 100).

        Returns:
            IndexingSummary with detailed execution statistics.
        """
        records = [to_vector_record(chunk) for chunk in embedded_chunks]
        expected_count = len(records)
        inserted = 0
        failures: List[Dict[str, Any]] = []
        batch_list = list(batches(records, size=batch_size))

        for batch in batch_list:
            try:
                self.collection.upsert(list(batch))
                inserted += len(batch)
            except Exception as error:
                logger.error(f"Failed to upsert batch starting with ID '{batch[0]['id']}': {error}")
                failures.append(
                    {
                        "batch_start_id": batch[0]["id"],
                        "batch_len": len(batch),
                        "error": str(error),
                    }
                )

        indexed_count = self.collection.count()
        summary = IndexingSummary(
            expected_count=expected_count,
            inserted_this_run=inserted,
            indexed_count=indexed_count,
            batch_size=batch_size,
            total_batches=len(batch_list),
            failures=failures,
        )

        if indexed_count != expected_count:
            logger.warning(
                f"Indexed count ({indexed_count}) does not match expected chunk count ({expected_count})!"
            )

        return summary

    def spot_check(self, sample_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Spot-check a stored record against its original source chunk.

        Args:
            sample_chunk: Original embedded chunk dict.

        Returns:
            Dict containing spot-check verification details.
        """
        rec_id = str(sample_chunk.get("id") or sample_chunk.get("chunk_id"))
        stored = self.collection.get(rec_id)

        if not stored:
            raise ValueError(f"Spot check failed: Record '{rec_id}' not found in collection")

        original_vector = sample_chunk.get("vector") or sample_chunk.get("embedding") or []
        original_meta = sample_chunk.get("metadata", {})
        original_source = original_meta.get("source") or original_meta.get("source_document") or original_meta.get("source_id")

        text_matches = stored["text"] == sample_chunk["text"]
        source_matches = stored["metadata"]["source"] == str(original_source)
        vector_len_matches = len(stored["vector"]) == len(original_vector)

        if not (text_matches and source_matches and vector_len_matches):
            raise AssertionError(
                f"Spot check mismatch for ID '{rec_id}': "
                f"text_matches={text_matches}, source_matches={source_matches}, vector_len_matches={vector_len_matches}"
            )

        return {
            "spot_check_passed": True,
            "id": stored["id"],
            "source": stored["metadata"]["source"],
            "chunk_index": stored["metadata"].get("chunk_index"),
            "vector_length": len(stored["vector"]),
            "text_preview": stored["text"][:120],
        }
