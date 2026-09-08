# Vector Database Setup & Collection Design Report

## Overview
- **Database Backend:** ChromaDB (In-Memory / Persistent)
- **Collection Name:** `rag_chunks`
- **Vector Dimension:** `1536`
- **Distance Metric:** `cosine`

## Schema Design
- **Record ID:** `string` (stable chunk identifier)
- **Vector:** `List[float]` (dimension=1536)
- **Text:** `string` (original source text chunk)
- **Metadata:** `Dict[str, Any]` (`source`, `chunk_index`, `section`)

## Readback Verification
- **Readback ID:** `account-guide.md:0`
- **Vector Length:** `1536`
- **Text:** `Password reset instructions for learner accounts.`
- **Metadata:** `{"source": "account-guide.md", "chunk_index": 0, "section": "Account access"}`
- **Verification Status:** `PASSED`

## Dimension Safety
- **Mismatched Vector Test:** Rejection verified with early `ValueError`.
- **Dimension Safety Status:** `PASSED`
