# Similarity Search & Top-k Retrieval

Provider: `offline semantic fixture`
Embedding model: `same as document chunks`
Query: **What medication instructions should the patient follow?**

## Query embedding

The user query is embedded with the same model and vector space as the document chunks, so cosine similarity compares compatible representations.

Query embedding length: `8`

## Top-k similarity search

### k = 1

1. **Score:** `0.999884`
   **Text:** The patient should take the prescribed medication with water.
   **Metadata:** `{"chunk_index": 1, "section": "Dosage", "source_document": "medication-guideline.pdf"}`

### k = 3

1. **Score:** `0.999884`
   **Text:** The patient should take the prescribed medication with water.
   **Metadata:** `{"chunk_index": 1, "section": "Dosage", "source_document": "medication-guideline.pdf"}`
2. **Score:** `0.999676`
   **Text:** Patients need to use their recommended medicine with water.
   **Metadata:** `{"chunk_index": 2, "section": "Administration", "source_document": "medication-guideline.pdf"}`
3. **Score:** `0.076703`
   **Text:** The help desk can reset an employee password.
   **Metadata:** `{"chunk_index": 1, "section": "Account access", "source_document": "it-support-handbook.pdf"}`

## Interpretation

- A larger k returns more candidate chunks, which improves recall but may dilute the answer with less relevant context.
- A smaller k gives tighter, higher-precision context but risks missing useful supporting passages.
- The full query answer should later use the retrieved chunks as grounded context for the model.
