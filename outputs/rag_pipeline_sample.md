# RAG pipeline sample output

Query: `What medication instructions should the patient follow?`

Embedding provider: `offline semantic fixture`

Answer:

> Based on the indexed documents, The patient should take the prescribed medication with water. Patients need to use their recommended medicine with water. ([Source 1], [Source 2]).

Retrieved sources:

1. **Score:** `0.999884`  
   **Source:** `medication-guideline.pdf`  
   **Section:** `Dosage`  
   **Text:** The patient should take the prescribed medication with water.
2. **Score:** `0.999676`  
   **Source:** `medication-guideline.pdf`  
   **Section:** `Administration`  
   **Text:** Patients need to use their recommended medicine with water.

The answer is generated from the two retrieved passages and each passage is
returned with its source metadata for inspection.