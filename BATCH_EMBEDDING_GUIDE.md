# Batch Embedding Pipeline - Complete Guide

A production-ready Python module for efficiently embedding large corpora of text chunks using OpenAI's embedding API. Features batching, automatic retries with exponential backoff, cost tracking, and intelligent deduplication to avoid wasteful re-embeddings.

## Features

### ✅ Batch Processing
- **Efficient API usage**: Send multiple chunks per request instead of one-per-request
- **Configurable batch size**: Default 64 (max 2048 for text-embedding-3-small)
- **Automatic batching**: Chunks are transparently batched before sending to API

### ♻️ Resumable & Safe
- **Skip already-embedded chunks**: On rerun, detect and skip previously embedded chunks
- **Crash-safe**: Embeddings saved to disk after each batch
- **Resume capability**: Interrupted runs can safely resume from last completed batch
- **Zero duplicate cost**: Rerun shows 100% cost savings via deduplication

### 🔄 Robust Error Handling
- **Exponential backoff**: Configurable retry strategy (1s, 2s, 4s, 8s, 16s default)
- **Rate limit handling**: Specifically handles OpenAI 429 errors
- **Transient failure recovery**: Automatic retry for API timeouts
- **Error visibility**: All failed chunks logged and reported in summary

### 💰 Cost Tracking
- **Real-time token counting**: Accurate cost estimation using tiktoken
- **Per-run reporting**: Know exactly how much each run costs
- **Model-aware pricing**: Built-in pricing for text-embedding-3-small
- **Fallback estimation**: Works without tiktoken (uses character approximation)

---

## Quick Start

### 1. Installation

```bash
# Install requirements
pip install -r requirements.txt

# Optional: install tiktoken for accurate token counting
pip install tiktoken
```

### 2. Configuration

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
# Optional: for Azure OpenAI or custom base URL
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. Basic Usage

```python
from src.batch_embedding import BatchEmbeddingPipeline
from src.sample_chunks import build_sample_chunks

# Initialize pipeline
pipeline = BatchEmbeddingPipeline(batch_size=64)

# Load chunks
store = build_sample_chunks()

# Embed all chunks (skips already-embedded by default)
summary = pipeline.embed_from_store(store, skip_existing=True)

# View results
print(summary.to_markdown())
```

### 4. Run the Demo

```bash
cd src
python demo_batch_embedding.py
```

This will:
1. **Run #1**: Embed all sample chunks (first time)
2. **Run #2**: Rerun to show deduplication (0 cost, skips all)
3. Generate comprehensive report with cost analysis

---

## API Reference

### BatchEmbeddingPipeline

Main class for batch embedding operations.

#### Constructor

```python
pipeline = BatchEmbeddingPipeline(
    api_key=None,              # Optional: override OPENAI_API_KEY env var
    base_url=None,             # Optional: override OPENAI_BASE_URL env var
    model="text-embedding-3-small",  # Embedding model name
    batch_size=64,             # Chunks per API request (1-2048)
    max_retries=5,             # Retry attempts per batch
    embeddings_file=None,      # Save location (default: outputs/embeddings.json)
)
```

#### Methods

**`embed_chunks(chunks: List[Chunk], skip_existing=True) -> RunSummary`**
- Embed a list of Chunk objects
- `skip_existing`: Skip chunks already in embeddings.json
- Returns: RunSummary with statistics

**`embed_from_store(store: ChunkStore, skip_existing=True) -> RunSummary`**
- Embed all chunks from a ChunkStore
- Convenience wrapper around embed_chunks

**`get_existing_embedding_ids() -> set[str]`**
- Get IDs of chunks already embedded
- Used to detect and skip duplicates

**`get_embeddings() -> Dict[str, EmbeddingRecord]`**
- Get all stored embeddings
- Returns dict mapping chunk_id to EmbeddingRecord

**`get_embedding(chunk_id: str) -> Optional[EmbeddingRecord]`**
- Get embedding for a single chunk

### RunSummary

Statistics about an embedding run.

**Key attributes:**
- `timestamp`: ISO format timestamp
- `total_chunks`: Total chunks processed
- `skipped_existing`: Chunks skipped (already embedded)
- `embedded`: Successfully embedded
- `failed`: Failed to embed
- `input_tokens_used`: Total input tokens sent to API
- `estimated_cost_usd`: Estimated cost in USD
- `batch_size`: Batch size used
- `total_batches`: Number of batches
- `successful_batches`: Batches completed successfully
- `failed_batches`: Batches that failed
- `duration_seconds`: Wall-clock time for run

**Methods:**
- `to_dict()`: Convert to dictionary (for JSON)
- `to_markdown()`: Generate markdown report

### EmbeddingRecord

Single embedding record with metadata.

**Attributes:**
- `chunk_id`: Unique chunk identifier
- `text`: Original text
- `embedding`: Vector (List[float])
- `source_id`: Source document ID
- `section`: Document section
- `page`: Page number
- `position`: Position in source
- `created_at`: ISO timestamp

---

## Detailed Batching Explanation

### Why Batching is More Efficient

**Without batching** (1 request per chunk):
```
3 chunks = 3 API requests
Total latency = 3 × (API overhead + processing)
Example: 3 × 500ms = 1.5s
```

**With batching** (64 chunks per request):
```
1000 chunks = ~16 API requests (vs 1000)
Total latency ≈ 16 × (API overhead + processing)
Example: 16 × 500ms = 8s (vs 500s)
60× faster!
```

The embedding API is optimized for batch processing—sending 64 texts is only slightly slower than sending 1.

### Configuring Batch Size

```python
# Small batches: lower latency, more requests
pipeline = BatchEmbeddingPipeline(batch_size=16)  # ~63 requests per 1000 chunks

# Large batches: higher throughput, fewer requests (recommended)
pipeline = BatchEmbeddingPipeline(batch_size=64)  # ~16 requests per 1000 chunks

# Max batch size (model limit)
pipeline = BatchEmbeddingPipeline(batch_size=2048)  # ~1 request per 2000 chunks
```

**Recommendation**: Start with 64, adjust based on your text length and latency requirements.

---

## Rate Limiting & Retry Strategy

### How Retries Work

The pipeline automatically retries failed batches with exponential backoff:

```
Batch fails with rate limit error (429)
  ↓
Wait 1 second, retry
  ↓
Still fails, wait 2 seconds, retry
  ↓
Still fails, wait 4 seconds, retry
  ↓
Still fails, wait 8 seconds, retry
  ↓
Still fails, wait 16 seconds, retry
  ↓
5 retries exhausted → log error → continue to next batch
```

**Why exponential backoff?**
- Gradual increase respects API rate limits
- Doesn't hammer the server on failure
- Gives transient issues time to resolve
- Standard industry practice

### Catching Specific Errors

```python
from openai import RateLimitError, APIError

# The pipeline catches these automatically:
# - RateLimitError (429): Rate limit exceeded
# - APIError: Transient API errors
# - Generic Exception: Unexpected errors (logged and skipped)
```

### Monitoring Failures

```python
pipeline = BatchEmbeddingPipeline()
summary = pipeline.embed_from_store(store)

# Check for failures
if summary.failed > 0:
    print(f"Failed chunks: {summary.failed_chunk_ids}")
    print(f"Errors: {summary.run_errors}")
```

---

## Cost Estimation & Tracking

### Model Pricing

The pipeline is configured for **text-embedding-3-small**:
- **Cost**: $0.02 per 1 million input tokens
- **Speed**: Fast (good for large corpora)
- **Quality**: Excellent for semantic search

Alternative models (adjust `PRICE_PER_1K_INPUT_TOKENS`):
- `text-embedding-3-large`: $0.13 per 1M tokens (higher quality)
- Custom model: Update pricing constant

### Token Counting

The pipeline uses **tiktoken** for accurate token counting:

```python
from src.token_counter import count_tokens

tokens = count_tokens("Your text here")
# Uses OpenAI's actual tokenizer
```

**Fallback mode** (if tiktoken not installed):
```python
# Rough approximation: ~4 characters per token in English
tokens = max(1, int(len(text) / 4))
```

### Cost Calculation Example

**100 clinical chunks, ~200 tokens each:**
- Total tokens: 100 × 200 = 20,000
- Cost: (20,000 / 1,000,000) × $0.02 = **$0.0004**

**1,000,000 chunks, ~100 tokens each:**
- Total tokens: 1M × 100 = 100M
- Cost: (100M / 1,000,000) × $0.02 = **$2,000**

---

## Skip-on-Rerun: Deduplication Logic

### How It Works

**First run:**
```
All 1000 chunks are new
  → All 1000 sent to API
  → Embeddings saved to embeddings.json
  → Cost: $X
```

**Second run (same chunks):**
```
Load existing embeddings from embeddings.json
  → 1000 chunk IDs found in file
  → Filter query: pending = all_chunks - existing_ids = []
  → Zero chunks sent to API
  → Cost: $0 (free rerun)
```

### Implementation Details

```python
# On initialization, load existing embeddings
self._load_existing_embeddings()  # Reads embeddings.json

# Get set of known chunk IDs (O(1) lookup)
existing_ids = self.get_existing_embedding_ids()  # Returns set[str]

# Filter to only pending chunks
pending = [c for c in chunks if c.chunk_id not in existing_ids]

# Process only pending
for batch in self._batches(pending, batch_size):
    # Send to API only if not already embedded
```

### Safety Guarantees

- **Atomic writes**: Embeddings saved to disk after each batch
- **Crash-safe**: If interrupted, just rerun—skips completed batches
- **No duplicates**: Chunk IDs are UUIDs (globally unique)
- **Resumable**: Progressive processing is 100% safe

---

## Scaling to Large Corpora

### Strategy 1: Single-Process with Checkpointing

For corpora up to **10-100k chunks**:

```python
# Day 1: Embed 50k chunks
for day in range(10):
    pipeline = BatchEmbeddingPipeline()
    daily_chunks = load_chunks_for_day(day)
    summary = pipeline.embed_chunks(daily_chunks, skip_existing=True)
    print(f"Day {day}: {summary.embedded} new, {summary.skipped_existing} skipped")

# Run summary shows rerun cost savings
```

**Why this works:**
- Embeddings persist to disk
- Each run skips already-done chunks
- No duplicate costs
- Progress tracked in run summaries

### Strategy 2: Partition-Based Parallelization

For corpora **100k+ chunks**:

```python
# Split into logical partitions
partitions = [
    ("hospital_a_protocols", hospital_a_chunks),
    ("hospital_b_protocols", hospital_b_chunks),
    ("drug_database", drug_chunks),
]

# Run parallel embeddings (respecting rate limits)
import concurrent.futures
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    futures = []
    for name, chunks in partitions:
        future = executor.submit(embed_partition, name, chunks)
        futures.append(future)
    
    for future in concurrent.futures.as_completed(futures):
        summary = future.result()
        print(summary.to_markdown())
```

**Benefits:**
- Process multiple sources in parallel
- Respects OpenAI rate limits (~3,500 requests/min)
- Each partition maintains own embedding file
- Failures isolated to single partition

### Strategy 3: Streaming with Progress Checkpoints

For **massive corpora** (1M+ chunks):

```python
def embed_with_checkpoints(chunks_generator, checkpoint_every=10000):
    pipeline = BatchEmbeddingPipeline()
    
    for batch_num, batch in enumerate(batches(chunks_generator, 100)):
        summary = pipeline.embed_chunks(batch, skip_existing=True)
        
        # Save checkpoint every N batches
        if batch_num % checkpoint_every == 0:
            save_checkpoint(batch_num, summary)
            print(f"Checkpoint: {batch_num} batches processed")
```

**Advantages:**
- Process one batch at a time from generator
- Constant memory usage (no loading all chunks)
- Checkpoint every N batches for recovery
- Resume from last checkpoint if interrupted

---

## File Structure & Outputs

```
project/
├── outputs/
│   ├── embeddings.json              # Main storage: all embeddings + metadata
│   ├── sample_run_summary.json       # Example summary output
│   ├── run_summary_first.json        # First run summary
│   ├── run_summary_second.json       # Second run summary
│   └── BATCH_EMBEDDING_RUN_REPORT.md # Comprehensive report
├── src/
│   ├── batch_embedding.py           # Main pipeline module
│   ├── demo_batch_embedding.py       # Demo with two runs
│   ├── chunk_metadata.py            # Chunk storage
│   ├── sample_chunks.py             # Sample data
│   └── token_counter.py             # Token utilities
└── .env                             # API configuration
```

### embeddings.json Structure

```json
[
  {
    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
    "text": "Clinical protocol text...",
    "embedding": [-0.0234, 0.0892, 0.1234, ...],
    "source_id": "README.md",
    "section": "Clinical Protocols",
    "page": 42,
    "position": 100,
    "created_at": "2026-08-31T10:15:30.123456"
  },
  ...
]
```

---

## Testing & Validation

### Test the Pipeline Locally

```bash
# Test with mock mode (no API calls)
MOCK_MODE=true python src/demo_batch_embedding.py

# Test with real API (requires key in .env)
python src/demo_batch_embedding.py
```

### Verify Embeddings

```python
from src.batch_embedding import BatchEmbeddingPipeline

pipeline = BatchEmbeddingPipeline()
embeddings = pipeline.get_embeddings()

# Check structure
for chunk_id, record in embeddings.items():
    assert len(record.embedding) == 1536  # text-embedding-3-small size
    assert record.chunk_id == chunk_id
    assert isinstance(record.text, str)
    
print(f"✓ Verified {len(embeddings)} embeddings")
```

---

## Troubleshooting

### Issue: "No API key found. Using mock mode."

**Solution**: Set `OPENAI_API_KEY` in `.env` or environment:
```bash
export OPENAI_API_KEY=sk-your-key
```

### Issue: "Rate limit exceeded after N retries"

**Solution**: Reduce batch size or add delays between runs:
```python
# Smaller batches = fewer requests = less likely to hit rate limit
pipeline = BatchEmbeddingPipeline(batch_size=32)  # Default is 64

# Or wait between retries in your calling code
```

### Issue: "tiktoken module not found"

**Solution**: Install optional dependency:
```bash
pip install tiktoken
```

Falls back to character approximation if not available (less accurate).

### Issue: "embeddings.json not found"

**Solution**: First run creates it automatically. If missing:
```python
pipeline = BatchEmbeddingPipeline()
# Initialize empty embeddings
pipeline._save_embeddings()
```

---

## Production Deployment

### Environment Variables

Set in production environment:

```env
OPENAI_API_KEY=sk-prod-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
BATCH_SIZE=64
MAX_RETRIES=5
EMBEDDINGS_FILE=/data/embeddings.json
```

### Monitoring & Logging

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embedding_run.log"),
        logging.StreamHandler(),
    ]
)

pipeline = BatchEmbeddingPipeline()
summary = pipeline.embed_from_store(store)

# Log to external service
send_to_monitoring(summary.to_dict())
```

### Cloud Storage Integration

For multi-worker scenarios, store embeddings.json in cloud:

```python
import boto3

# Save to S3 after each batch
s3_client = boto3.client('s3')

def save_embeddings_to_s3(pipeline):
    with open(pipeline.embeddings_file) as f:
        s3_client.upload_fileobj(
            f, 'bucket-name', 'embeddings/embeddings.json'
        )
```

---

## Contributing

Found a bug or want to improve the pipeline? See our contribution guidelines (if applicable to your team).

---

## License

See LICENSE file in repository.

---

## Questions?

Review the **demo output** in `outputs/BATCH_EMBEDDING_RUN_REPORT.md` for a complete worked example.
