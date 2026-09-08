# Quantitative Retrieval Quality Evaluation Report

**Queries Evaluated**: `5`  
**Top-k Cutoff**: `k = 5`  
**Recall@5**: `1.0000` (`100.0%`)  
**Precision@5**: `0.2400` (`24.0%`)  
**Failure Count (Recall < 1.0)**: `0`

---

## 1. Query-Level Evaluation Results

| Query | Expected Chunk IDs | Retrieved Chunk IDs | Hits | Recall | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `How can a learner reset their password?` | account-guide.md:0<br>account-guide.md:1 | account-guide.md:0<br>account-guide.md:1<br>submission-rubric.md:2... | `account-guide.md:0, account-guide.md:1` | 1.00 | 0.40 |
| `What evidence is required for project submission?` | submission-rubric.md:2 | submission-rubric.md:2<br>account-guide.md:0<br>account-guide.md:1... | `submission-rubric.md:2` | 1.00 | 0.20 |
| `What medication instructions should the patient follow?` | medication-guideline.pdf:0 | medication-guideline.pdf:0<br>account-guide.md:0<br>triage-protocol.pdf:0... | `medication-guideline.pdf:0` | 1.00 | 0.20 |
| `What emergency triage verification procedures exist?` | triage-protocol.pdf:0 | triage-protocol.pdf:0<br>account-guide.md:0<br>account-guide.md:1... | `triage-protocol.pdf:0` | 1.00 | 0.20 |
| `What are the rules for campus wireless network account provisioning?` | campus-it-guide.pdf:0 | campus-it-guide.pdf:0<br>security-policy.pdf:5<br>account-guide.md:0... | `campus-it-guide.pdf:0` | 1.00 | 0.20 |

---

## 2. Failure Inspection & Root Cause Diagnostics

✅ **No failures observed!** All labelled queries achieved 100% recall@k.

---

## 3. Actionable Strategies to Improve Recall

- **Increase K**: Increase top-k candidate parameter (e.g. from 5 to 10 or 20) so borderline relevant chunks enter the candidate set.
- **Hybrid Search**: Combine sparse BM25 keyword matching with dense vector search to bridge vocabulary gaps (e.g., specific ID or code queries).
- **Metadata Filtering**: Apply strict section or source metadata filters to eliminate out-of-domain noise from the vector search space.
- **Re Ranking**: Use a two-stage pipeline: retrieve a large candidate set (k=20) and apply a cross-encoder / LLM re-ranker to boost relevant chunks to top ranks.
- **Chunking Tuning**: Adjust chunk sizes and overlap to preserve complete context windows without fragmenting key facts across chunk boundaries.
- **Query Rewriting**: Use LLM query expansion / HyDE (Hypothetical Document Embeddings) to expand short or ambiguous user queries.
