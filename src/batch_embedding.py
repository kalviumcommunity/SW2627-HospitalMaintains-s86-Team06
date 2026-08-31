"""
Batch Embedding Pipeline with Retry Logic, Cost Tracking, and Deduplication.

This module provides efficient embedding of large corpora by:
- Batching multiple chunks per API request
- Implementing exponential backoff for rate limits and transient errors
- Tracking cost and token usage
- Skipping already-embedded chunks on re-runs
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

# Try importing tiktoken for accurate token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from src.chunk_metadata import Chunk, ChunkStore
    from src.token_counter import count_tokens
except (ModuleNotFoundError, ImportError):  # pragma: no cover - direct script execution fallback
    from chunk_metadata import Chunk, ChunkStore
    from token_counter import count_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Embedding model and pricing (adjust as needed)
EMBEDDING_MODEL = "text-embedding-3-small"
# Pricing: text-embedding-3-small is $0.02 per 1M input tokens
PRICE_PER_1M_INPUT_TOKENS = 0.02
PRICE_PER_1K_INPUT_TOKENS = PRICE_PER_1M_INPUT_TOKENS / 1000


@dataclass
class EmbeddingRecord:
    """Represents a chunk with its embedding."""
    chunk_id: str
    text: str
    embedding: List[float]
    source_id: str
    section: Optional[str] = None
    page: Optional[int] = None
    position: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RunSummary:
    """Summary of an embedding run."""
    timestamp: str
    total_chunks: int
    skipped_existing: int
    embedded: int
    failed: int
    input_tokens_used: int
    estimated_cost_usd: float
    batch_size: int
    total_batches: int
    successful_batches: int
    failed_batches: int
    failed_chunk_ids: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    model: str = EMBEDDING_MODEL
    run_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate a markdown summary."""
        lines = [
            "# Batch Embedding Run Summary",
            "",
            f"**Timestamp:** {self.timestamp}",
            f"**Model:** {self.model}",
            f"**Duration:** {self.duration_seconds:.2f} seconds",
            "",
            "## Chunk Statistics",
            f"- **Total Chunks:** {self.total_chunks}",
            f"- **Skipped (Already Embedded):** {self.skipped_existing}",
            f"- **Successfully Embedded:** {self.embedded}",
            f"- **Failed:** {self.failed}",
            "",
            "## Batch Processing",
            f"- **Batch Size:** {self.batch_size}",
            f"- **Total Batches:** {self.total_batches}",
            f"- **Successful Batches:** {self.successful_batches}",
            f"- **Failed Batches:** {self.failed_batches}",
            "",
            "## Token & Cost Analysis",
            f"- **Input Tokens Used:** {self.input_tokens_used:,}",
            f"- **Estimated Cost:** ${self.estimated_cost_usd:.6f} USD",
            "",
        ]

        if self.failed_chunk_ids:
            lines.extend([
                "## Failed Chunks",
                "The following chunks failed to embed and should be retried:",
                "",
            ])
            for chunk_id in self.failed_chunk_ids[:20]:  # Show first 20
                lines.append(f"- `{chunk_id}`")
            if len(self.failed_chunk_ids) > 20:
                lines.append(f"- ... and {len(self.failed_chunk_ids) - 20} more")
            lines.append("")

        if self.run_errors:
            lines.extend([
                "## Errors Encountered",
                "These errors occurred during the run:",
                "",
            ])
            for error in self.run_errors[:5]:  # Show first 5 errors
                lines.append(f"- {error}")
            if len(self.run_errors) > 5:
                lines.append(f"- ... and {len(self.run_errors) - 5} more errors")
            lines.append("")

        return "\n".join(lines)


class BatchEmbeddingPipeline:
    """
    Embeds chunks in batches with retry logic, cost tracking, and deduplication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 64,
        max_retries: int = 5,
        embeddings_file: Optional[str] = None,
    ):
        """
        Initialize the batch embedding pipeline.

        Args:
            api_key: OpenAI API key (or from .env)
            base_url: Base URL for API (or from .env)
            model: Embedding model name
            batch_size: Number of texts per batch (max 2048 for text-embedding-3-small)
            max_retries: Maximum retry attempts per batch
            embeddings_file: Path to save embeddings JSON (defaults to outputs/embeddings.json)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.embeddings_file = embeddings_file or self._default_embeddings_path()

        if not self.api_key:
            logger.warning("No API key found. Using mock mode.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        # In-memory storage for embeddings
        self._embeddings: Dict[str, EmbeddingRecord] = {}
        self._load_existing_embeddings()

    @staticmethod
    def _default_embeddings_path() -> str:
        """Get default embeddings file path."""
        outputs_dir = Path(__file__).parent.parent / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return str(outputs_dir / "embeddings.json")

    def _load_existing_embeddings(self) -> None:
        """Load existing embeddings from file to enable deduplication."""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, "r") as f:
                    data = json.load(f)
                    for record_dict in data:
                        record = EmbeddingRecord(**record_dict)
                        self._embeddings[record.chunk_id] = record
                logger.info(f"Loaded {len(self._embeddings)} existing embeddings from {self.embeddings_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing embeddings: {e}")

    def get_existing_embedding_ids(self) -> set[str]:
        """Get IDs of all chunks that already have embeddings."""
        return set(self._embeddings.keys())

    def _save_embeddings(self) -> None:
        """Save all embeddings to file."""
        try:
            with open(self.embeddings_file, "w") as f:
                records = [record.to_dict() for record in self._embeddings.values()]
                json.dump(records, f, indent=2)
            logger.info(f"Saved {len(self._embeddings)} embeddings to {self.embeddings_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def _batches(self, items: List[Any], size: int) -> List[List[Any]]:
        """Yield successive batches from items."""
        batches = []
        for start in range(0, len(items), size):
            batches.append(items[start : start + size])
        return batches

    def _embed_batch_with_retry(self, texts: List[str], batch_num: int) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Embed a batch of texts with exponential backoff retry logic.

        Returns:
            Tuple of (embeddings_list, error_message)
            - embeddings_list: List of embedding dicts with 'index' and 'embedding' keys, or None if failed
            - error_message: Error message if failed, None if successful
        """
        if self.client is None:
            logger.warning(f"Mock mode: skipping embedding batch {batch_num}")
            return None, "Mock mode: API not configured"

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Batch {batch_num}: embedding {len(texts)} texts (attempt {attempt + 1}/{self.max_retries})")
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                logger.info(f"Batch {batch_num}: successfully embedded {len(response.data)} texts")
                return response.data, None

            except RateLimitError as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"Rate limit exceeded after {self.max_retries} retries: {str(e)}"
                    logger.error(error_msg)
                    return None, error_msg

                wait_seconds = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                logger.warning(
                    f"Batch {batch_num}: Rate limited. Retrying after {wait_seconds}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                time.sleep(wait_seconds)

            except APIError as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"API error after {self.max_retries} retries: {str(e)}"
                    logger.error(error_msg)
                    return None, error_msg

                wait_seconds = 2 ** attempt
                logger.warning(
                    f"Batch {batch_num}: API error. Retrying after {wait_seconds}s: {str(e)}"
                )
                time.sleep(wait_seconds)

            except Exception as e:
                error_msg = f"Unexpected error in batch {batch_num}: {str(e)}"
                logger.error(error_msg)
                return None, error_msg

        return None, f"Failed to embed batch {batch_num} after {self.max_retries} attempts"

    def embed_chunks(
        self,
        chunks: List[Chunk],
        skip_existing: bool = True,
    ) -> RunSummary:
        """
        Embed a list of chunks in batches.

        Args:
            chunks: List of Chunk objects to embed
            skip_existing: If True, skip chunks that already have embeddings

        Returns:
            RunSummary with statistics about the embedding run
        """
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat()

        total_chunks = len(chunks)
        existing_ids = self.get_existing_embedding_ids() if skip_existing else set()
        pending_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = total_chunks - len(pending_chunks)

        logger.info(f"Starting embedding run: {total_chunks} total, {skipped} skipped, {len(pending_chunks)} pending")

        summary = RunSummary(
            timestamp=timestamp,
            total_chunks=total_chunks,
            skipped_existing=skipped,
            embedded=0,
            failed=0,
            input_tokens_used=0,
            estimated_cost_usd=0.0,
            batch_size=self.batch_size,
            total_batches=0,
            successful_batches=0,
            failed_batches=0,
        )

        if not pending_chunks:
            logger.info("No chunks to embed. All chunks already have embeddings.")
            summary.duration_seconds = time.time() - start_time
            return summary

        # Batch the pending chunks
        batches = self._batches(pending_chunks, self.batch_size)
        summary.total_batches = len(batches)

        for batch_num, batch in enumerate(batches, start=1):
            # Prepare texts and keep chunk references
            texts = [chunk.text for chunk in batch]
            chunk_by_idx = {i: chunk for i, chunk in enumerate(batch)}

            # Embed batch with retry
            embeddings_data, error = self._embed_batch_with_retry(texts, batch_num)

            if error:
                logger.error(f"Batch {batch_num} failed: {error}")
                summary.failed += len(batch)
                summary.failed_batches += 1
                summary.failed_chunk_ids.extend([c.chunk_id for c in batch])
                summary.run_errors.append(error)
                continue

            summary.successful_batches += 1

            # Process embeddings and save records
            for embedding_obj in embeddings_data:
                idx = embedding_obj.index
                if idx not in chunk_by_idx:
                    logger.warning(f"Received embedding for unknown index {idx}")
                    continue

                chunk = chunk_by_idx[idx]
                record = EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding_obj.embedding,
                    source_id=chunk.metadata.source_id,
                    section=chunk.metadata.section,
                    page=chunk.metadata.page,
                    position=chunk.metadata.position,
                )
                self._embeddings[chunk.chunk_id] = record
                summary.embedded += 1

            # Update token count (rough estimate based on text length)
            batch_tokens = sum(count_tokens(text) for text in texts)
            summary.input_tokens_used += batch_tokens

            logger.info(f"Batch {batch_num}: embedded {len(embeddings_data)} chunks, ~{batch_tokens} tokens")

        # Calculate cost
        summary.estimated_cost_usd = round(
            (summary.input_tokens_used / 1000) * PRICE_PER_1K_INPUT_TOKENS, 6
        )

        # Save embeddings to disk
        self._save_embeddings()

        summary.duration_seconds = time.time() - start_time
        logger.info(f"Embedding run complete: {summary.embedded} embedded, {summary.failed} failed in {summary.duration_seconds:.2f}s")

        return summary

    def embed_from_store(self, store: ChunkStore, skip_existing: bool = True) -> RunSummary:
        """Embed all chunks from a ChunkStore."""
        chunks = store.list_chunks()
        return self.embed_chunks(chunks, skip_existing=skip_existing)

    def get_embeddings(self) -> Dict[str, EmbeddingRecord]:
        """Get all stored embeddings."""
        return self._embeddings.copy()

    def get_embedding(self, chunk_id: str) -> Optional[EmbeddingRecord]:
        """Get embedding for a specific chunk ID."""
        return self._embeddings.get(chunk_id)


def main():
    """Demo: embed sample chunks with batch pipeline."""
    try:
        from src.sample_chunks import build_sample_chunks
    except (ModuleNotFoundError, ImportError):  # pragma: no cover
        from sample_chunks import build_sample_chunks

    print("\n" + "=" * 70)
    print(" 📦 BATCH EMBEDDING PIPELINE DEMO")
    print("=" * 70 + "\n")

    # Build sample chunks
    store = build_sample_chunks()
    chunks = store.list_chunks()
    print(f"📚 Loaded {len(chunks)} sample chunks\n")

    # Initialize pipeline
    pipeline = BatchEmbeddingPipeline(batch_size=2)
    print(f"🔧 Pipeline initialized with batch_size={pipeline.batch_size}\n")

    # Show existing embeddings
    existing = pipeline.get_existing_embedding_ids()
    print(f"📁 Found {len(existing)} existing embeddings on disk\n")

    # Run embedding
    print("🚀 Starting embedding run...\n")
    summary = pipeline.embed_from_store(store, skip_existing=True)

    # Print summary
    print("\n" + summary.to_markdown())

    print(f"\n💾 Embeddings saved to: {pipeline.embeddings_file}\n")

    # Show sample embeddings
    if summary.embedded > 0:
        print("📊 Sample embeddings (first 100 dimensions of first embedding):\n")
        embeddings = pipeline.get_embeddings()
        first_embedding = next(iter(embeddings.values()))
        first_100_dims = first_embedding.embedding[:100]
        print(f"Chunk: {first_embedding.chunk_id}")
        print(f"Dimensions: {first_100_dims}\n")


if __name__ == "__main__":
    main()
