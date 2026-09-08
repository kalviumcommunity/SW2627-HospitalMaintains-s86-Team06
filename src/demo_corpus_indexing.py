"""Demonstration script for Corpus Record Indexing & Verification.

Demonstrates:
1. Preparing vector records from embedded corpus chunks (stable ID, vector, text, metadata)
2. Bulk batch indexing into a vector database collection
3. Confirming indexed count matches expected chunk count
4. Spot-checking stored record integrity against source chunks
5. Saving a markdown report of indexing metrics
"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.corpus_indexer import CorpusIndexer, to_vector_record
    from src.vector_db import COLLECTION_NAME, VECTOR_DIMENSION, VectorDBStore
except (ModuleNotFoundError, ImportError):
    from corpus_indexer import CorpusIndexer, to_vector_record
    from vector_db import COLLECTION_NAME, VECTOR_DIMENSION, VectorDBStore


def create_sample_embedded_chunks(count: int = 25) -> List[Dict[str, Any]]:
    """Create a sample dataset of embedded corpus chunks for demonstration."""
    clinical_sources = [
        "protocol_cardiology.md",
        "emergency_guidelines.pdf",
        "drug_interactions.html",
        "policy.txt",
    ]

    sample_texts = [
        "Initial Assessment: Complete 12-lead ECG within 10 minutes of patient arrival for chest pain triage.",
        "Emergency Resuscitation: Maintain airway patency and administer high-flow oxygen at 15L/min via non-rebreather mask.",
        "Drug Contraindication: Concurrent administration of nitrate therapy with PDE5 inhibitors is strictly contraindicated.",
        "Hospital Circular: All clinical staff must renew mandatory infection control certification annually.",
        "Pharmacology Note: High-sensitivity Troponin I samples drawn at 0h and 3h post-presentation.",
    ]

    chunks: List[Dict[str, Any]] = []

    for i in range(count):
        source = clinical_sources[i % len(clinical_sources)]
        text = sample_texts[i % len(sample_texts)]
        chunk_id = f"{source}:{i}"

        # Generate 1536-dimensional synthetic embedding vector
        vector = [round(0.01 * ((i + j) % 100), 4) for j in range(VECTOR_DIMENSION)]

        chunks.append(
            {
                "id": chunk_id,
                "embedding": vector,
                "text": f"Chunk [{i}]: {text}",
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "section": f"Section {1 + (i % 3)}",
                },
            }
        )

    return chunks


def run_corpus_indexing_demo() -> Dict[str, Any]:
    """Execute corpus record indexing, verification, and spot-check demo."""
    print("=" * 70)
    print(" CORPUS RECORD INDEXING & VERIFICATION DEMO")
    print("=" * 70)

    # 1. Generate embedded chunks
    embedded_chunks = create_sample_embedded_chunks(count=25)
    expected_count = len(embedded_chunks)
    print(f"\n1. Generated {expected_count} embedded corpus chunks for indexing.")

    # 2. Connect to vector DB and create collection
    db_store = VectorDBStore(in_memory=True)
    collection = db_store.create_collection(
        name=COLLECTION_NAME,
        dimension=VECTOR_DIMENSION,
        metric="cosine",
    )
    print(f"2. Connected to Vector DB & created collection '{COLLECTION_NAME}' (dim={VECTOR_DIMENSION}).")

    # 3. Index corpus records in batches
    indexer = CorpusIndexer(collection=collection)
    print(f"\n3. Indexing records in batches (batch_size=10)...")
    summary = indexer.index_chunks(embedded_chunks, batch_size=10)

    print(f"   expected chunks: {summary.expected_count}")
    print(f"   inserted this run: {summary.inserted_this_run}")
    print(f"   indexed count: {summary.indexed_count}")
    print(f"   failures: {summary.failures}")

    # 4. Count assertion
    assert summary.indexed_count == summary.expected_count, (
        f"indexed count ({summary.indexed_count}) does not match chunk count ({summary.expected_count})"
    )
    print("\n[SUCCESS] Count verification passed: Indexed count matches chunk count!")

    # 5. Spot-check integrity on first chunk
    print(f"\n4. Spot-checking record integrity...")
    sample = embedded_chunks[0]
    spot_result = indexer.spot_check(sample)

    print(f"   spot check passed: {spot_result['id']}")
    print(f"   source: {spot_result['source']}")
    print(f"   text preview: {spot_result['text_preview']}")

    assert spot_result["spot_check_passed"], "Spot check failed!"
    print("\n[SUCCESS] Spot check passed: Stored text, source, and vector length match original chunk!")

    # 6. Save Markdown report
    report_content = f"""# Corpus Record Indexing & Verification Report

## Summary
- **Target Collection:** `{COLLECTION_NAME}`
- **Vector Dimension:** `{VECTOR_DIMENSION}`
- **Expected Chunks:** `{summary.expected_count}`
- **Inserted Records:** `{summary.inserted_this_run}`
- **Final Indexed Count:** `{summary.indexed_count}`
- **Batch Size:** `{summary.batch_size}`
- **Total Batches:** `{summary.total_batches}`
- **Failure Count:** `{len(summary.failures)}`
- **Count Validation:** `PASSED` (`indexed_count == expected_count`)

## Spot Check Details
- **Spot-Checked ID:** `{spot_result['id']}`
- **Source Document:** `{spot_result['source']}`
- **Vector Dimension Verified:** `{spot_result['vector_length']}`
- **Text Preview:** `{spot_result['text_preview']}`
- **Integrity Validation:** `PASSED`
"""

    output_path = Path("outputs/corpus_indexing_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    print(f"\n[INFO] Saved indexing report to: {output_path.resolve()}")

    return summary.to_dict()


def main() -> None:
    run_corpus_indexing_demo()


if __name__ == "__main__":
    main()
