# RAG Retrieval & Re-ranking Analysis Report

**Query**: `What evidence is required for project submission?`  
**Initial Candidate Pool Size (k)**: `10`  
**Final Context Size (final_k)**: `3`

---

## 1. Initial Vector Retrieval (Before Re-ranking)

| Rank | Vector Score | Source Document | Section | Text Preview |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.8800 | `project-submission-overview.pdf` | `Overview` | Project submission guidelines: Overview of project lifecycle, milestones, and te... |
| 2 | 0.8200 | `submission-requirements.pdf` | `Submission Evidence` | Required evidence for project submission includes signed security audit reports,... |
| 3 | 0.7900 | `doc-standards.pdf` | `Formatting` | General documentation rules: Use markdown format for README files and document a... |

---

## 2. Re-ranked Results (After Re-ranking)

| Rank | Rerank Score (0-10) | Vector Score | Source Document | Section | Text Preview |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 10.00 | 0.8200 | `submission-requirements.pdf` | `Submission Evidence` | Required evidence for project submission includes signed security audit reports,... |
| 2 | 7.62 | 0.5200 | `finance-policy.pdf` | `Finances` | Budget approval evidence: Financial receipts and purchase order approvals must b... |
| 3 | 5.75 | 0.7600 | `clinical-compliance.pdf` | `Verification` | Verification evidence for clinical software submission must include validated te... |

---

## 3. Cost & Latency Trade-off Analysis

### Stage Comparison

- **Initial Retrieval**: Fast (1–10 ms for 100k+ chunks) latency | Low (Fixed vector index search, query embedding calculated once) cost | Moderate (Finds semantically similar chunks, but can rank broad overviews above specific answers) precision
- **Re-ranking Stage**: Higher (50–300 ms for 10–20 candidates depending on LLM / cross-encoder model) latency | Higher (Requires processing query + candidate tokens for every candidate chunk) cost | High (Directly evaluates exact relevance, evidence completeness, and query alignment) precision

### Strategy & Recommendations

- **Candidate Set Sizing**: Retrieve `10 to 20 candidates retrieved in Stage 1`.
- **Final Context Selection**: Select `3 to 5 top re-ranked chunks passed to LLM generation`.
- **Decision Rule**: Use re-ranking when precision is critical and initial vector search returns mixed-quality candidates. Measure whether answer quality improves relative to the added latency and token cost.
