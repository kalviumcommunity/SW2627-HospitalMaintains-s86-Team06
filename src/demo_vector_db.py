"""Demonstration script for Vector Database Setup & Collection Design.

Demonstrates:
1. Connecting to ChromaDB vector store
2. Creating a collection with schema and dimension (VECTOR_DIMENSION = 1536)
3. Upserting vector, source text, and metadata together
4. Reading back stored record and verifying ID, vector length, text, metadata
5. Early error handling for vector dimension mismatches
"""

import json
from pathlib import Path
try:
    from src.vector_db import COLLECTION_NAME, VECTOR_DIMENSION, VectorDBStore
except (ModuleNotFoundError, ImportError):
    from vector_db import COLLECTION_NAME, VECTOR_DIMENSION, VectorDBStore


def run_vector_db_demo() -> dict:
    """Execute the vector database collection setup and readback demo."""
    print("=" * 70)
    print(" VECTOR DATABASE SETUP & COLLECTION DESIGN DEMO")
    print("=" * 70)

    # 1. Connect application to vector database
    db_store = VectorDBStore(in_memory=True)
    print(f"\n1. Connected to Vector Database (ChromaDB in-memory)")

    # 2. Create collection with specific dimension and metric
    collection = db_store.create_collection(
        name=COLLECTION_NAME,
        dimension=VECTOR_DIMENSION,
        metric="cosine",
    )
    print(f"2. Collection created: '{COLLECTION_NAME}'")
    print(f"   - Vector dimension: {VECTOR_DIMENSION}")
    print(f"   - Distance metric: cosine")

    # 3. Create test record matching standard schema
    # Generate 1536-dimensional synthetic embedding vector
    test_embedding = [0.01 * (i % 100) for i in range(VECTOR_DIMENSION)]

    test_record = {
        "id": "account-guide.md:0",
        "vector": test_embedding,
        "text": "Password reset instructions for learner accounts.",
        "metadata": {
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Account access",
        },
    }

    # 4. Upsert record into collection
    print(f"\n3. Upserting record into collection...")
    collection.upsert([test_record])
    print(f"   - Inserted record ID: {test_record['id']}")

    # 5. Read back stored record and verify
    print(f"\n4. Reading back record from database...")
    stored = collection.get("account-guide.md:0")

    if not stored:
        raise RuntimeError("Failed to retrieve stored record from vector database!")

    print(f"   readback id: {stored['id']}")
    print(f"   vector length: {len(stored['vector'])}")
    print(f"   text: {stored['text']}")
    print(f"   metadata: {json.dumps(stored['metadata'])}")

    # Assertions for verification
    assert stored["id"] == test_record["id"], "ID mismatch!"
    assert len(stored["vector"]) == VECTOR_DIMENSION, "Vector dimension mismatch!"
    assert stored["text"] == test_record["text"], "Text content mismatch!"
    assert stored["metadata"] == test_record["metadata"], "Metadata mismatch!"
    print("\n[SUCCESS] Verification passed: Readback record matches input record perfectly!")

    # 6. Verify early failure on dimension mismatch
    print(f"\n5. Verifying early failure on incorrect vector dimension...")
    mismatched_vector = [0.1, 0.2, 0.3]  # Only 3 elements instead of 1536
    dimension_check_passed = False
    try:
        collection.upsert(
            [
                {
                    "id": "invalid-chunk:0",
                    "vector": mismatched_vector,
                    "text": "Mismatched dimension test.",
                    "metadata": {"source": "invalid.md"},
                }
            ]
        )
    except ValueError as err:
        dimension_check_passed = True
        print(f"   Caught expected ValueError: {err}")

    assert dimension_check_passed, "Collection failed to reject mismatched vector dimension!"
    print("[SUCCESS] Early dimension failure verification passed!")

    # Generate Markdown Report
    report_content = f"""# Vector Database Setup & Collection Design Report

## Overview
- **Database Backend:** ChromaDB (In-Memory / Persistent)
- **Collection Name:** `{COLLECTION_NAME}`
- **Vector Dimension:** `{VECTOR_DIMENSION}`
- **Distance Metric:** `cosine`

## Schema Design
- **Record ID:** `string` (stable chunk identifier)
- **Vector:** `List[float]` (dimension={VECTOR_DIMENSION})
- **Text:** `string` (original source text chunk)
- **Metadata:** `Dict[str, Any]` (`source`, `chunk_index`, `section`)

## Readback Verification
- **Readback ID:** `{stored['id']}`
- **Vector Length:** `{len(stored['vector'])}`
- **Text:** `{stored['text']}`
- **Metadata:** `{json.dumps(stored['metadata'])}`
- **Verification Status:** `PASSED`

## Dimension Safety
- **Mismatched Vector Test:** Rejection verified with early `ValueError`.
- **Dimension Safety Status:** `PASSED`
"""

    output_path = Path("outputs/vector_db_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    print(f"\n[INFO] Saved demo summary report to: {output_path.resolve()}")

    return {
        "status": "success",
        "readback_id": stored["id"],
        "vector_length": len(stored["vector"]),
        "text": stored["text"],
        "metadata": stored["metadata"],
    }


def main() -> None:
    run_vector_db_demo()


if __name__ == "__main__":
    main()
