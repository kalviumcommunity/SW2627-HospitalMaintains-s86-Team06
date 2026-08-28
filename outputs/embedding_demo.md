# Embedding Demonstration

Provider: `offline semantic fixture`
Chunks embedded: `3`
Vector dimension: `8`
All chunks have the same vector length: `True`

## Stored chunks and vectors

- **Text:** The patient should take the prescribed medication with water.
  **Metadata:** `{"chunk_index": 1, "section": "Dosage", "source_document": "medication-guideline.pdf"}`
  **Vector length:** `8`
  **Vector values (first 5):** `[0.91, 0.08, 0.02, 0.12, 0.04]`
- **Text:** Patients need to use their recommended medicine with water.
  **Metadata:** `{"chunk_index": 2, "section": "Administration", "source_document": "medication-guideline.pdf"}`
  **Vector length:** `8`
  **Vector values (first 5):** `[0.89, 0.1, 0.03, 0.11, 0.05]`
- **Text:** The help desk can reset an employee password.
  **Metadata:** `{"chunk_index": 1, "section": "Account access", "source_document": "it-support-handbook.pdf"}`
  **Vector length:** `8`
  **Vector values (first 5):** `[0.04, 0.02, 0.91, 0.03, 0.08]`

## Cosine similarity

- Similar pair (samples 1 and 2): `0.999421`
- Unrelated pair (samples 1 and 3): `0.076448`
- Similar pair scores higher: `True`

## Query ranking

Query: **What medication instructions should the patient follow?**

Metric: cosine similarity, because it compares the direction of embedding vectors and is less affected by their magnitude.

1. **Score:** `0.999884`
   **Text:** The patient should take the prescribed medication with water.
   **Metadata:** `{"chunk_index": 1, "section": "Dosage", "source_document": "medication-guideline.pdf"}`
2. **Score:** `0.999676`
   **Text:** Patients need to use their recommended medicine with water.
   **Metadata:** `{"chunk_index": 2, "section": "Administration", "source_document": "medication-guideline.pdf"}`
3. **Score:** `0.076703`
   **Text:** The help desk can reset an employee password.
   **Metadata:** `{"chunk_index": 1, "section": "Account access", "source_document": "it-support-handbook.pdf"}`

Most similar: **The patient should take the prescribed medication with water.** (`0.999884`)
Least similar: **The help desk can reset an employee password.** (`0.076703`)

## What vectors represent

Embedding vectors are numeric representations of a text's meaning in a learned semantic space. They are not random IDs and they are not keyword counts; texts with related meaning tend to be near one another in that space, which is why cosine similarity is useful for semantic search.
