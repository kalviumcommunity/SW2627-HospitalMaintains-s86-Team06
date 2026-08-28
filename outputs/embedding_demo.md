# Embedding Demonstration

Provider: `offline semantic fixture`
Vector dimension: `8`
All 3 sample texts have the same vector length: `True`

## Sample vectors

- **Text:** The patient should take the prescribed medication with water.
  **Vector:** `[0.91, 0.08, 0.02, 0.12, 0.04, 0.03, 0.01, 0.02]`
- **Text:** Patients need to use their recommended medicine with water.
  **Vector:** `[0.89, 0.1, 0.03, 0.11, 0.05, 0.02, 0.01, 0.03]`
- **Text:** The help desk can reset an employee password.
  **Vector:** `[0.04, 0.02, 0.91, 0.03, 0.08, 0.02, 0.01, 0.05]`

## Cosine similarity

- Similar pair (samples 1 and 2): `0.999421`
- Unrelated pair (samples 1 and 3): `0.076448`
- Similar pair scores higher: `True`

## What vectors represent

Embedding vectors are numeric representations of a text's meaning in a learned semantic space. They are not random IDs and they are not keyword counts; texts with related meaning tend to be near one another in that space, which is why cosine similarity is useful for semantic search.
