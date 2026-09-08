# Retrieval Relevance Tuning Results

## Test queries and expected sources

| Query | Expected source |
| --- | --- |
| password reset steps | it-support-handbook.pdf |
| password expiry policy | security-policy-2026.pdf |
| wireless infrastructure | campus-it-guide.pdf |
| medication instructions | medication-guideline.pdf |
| emergency identity check | triage-protocol.pdf |

## Compared settings and relevance results

Metrics are averaged over the five queries. A hit means the expected source appears in the returned list.

| Setting | Top-1 hit rate | Top-k hit rate | MRR |
| --- | ---: | ---: | ---: |
| baseline: k=3, no filter | 1.00 | 1.00 | 1.00 |
| filtered: k=2, section filter, threshold=0.80 | 0.20 | 0.20 | 0.20 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | 1.00 | 1.00 | 1.00 |

## Query-level results

| Setting | Query | Expected | Retrieved | Top-k hit |
| --- | --- | --- | --- | ---: |
| baseline: k=3, no filter | password reset steps | it-support-handbook.pdf | it-support-handbook.pdf, security-policy-2026.pdf, campus-it-guide.pdf | 1 |
| baseline: k=3, no filter | password expiry policy | security-policy-2026.pdf | security-policy-2026.pdf, it-support-handbook.pdf, campus-it-guide.pdf | 1 |
| baseline: k=3, no filter | wireless infrastructure | campus-it-guide.pdf | campus-it-guide.pdf, security-policy-2026.pdf, it-support-handbook.pdf | 1 |
| baseline: k=3, no filter | medication instructions | medication-guideline.pdf | medication-guideline.pdf, triage-protocol.pdf, campus-it-guide.pdf | 1 |
| baseline: k=3, no filter | emergency identity check | triage-protocol.pdf | triage-protocol.pdf, medication-guideline.pdf, campus-it-guide.pdf | 1 |
| filtered: k=2, section filter, threshold=0.80 | password reset steps | it-support-handbook.pdf | it-support-handbook.pdf | 1 |
| filtered: k=2, section filter, threshold=0.80 | password expiry policy | security-policy-2026.pdf | it-support-handbook.pdf | 0 |
| filtered: k=2, section filter, threshold=0.80 | wireless infrastructure | campus-it-guide.pdf | it-support-handbook.pdf | 0 |
| filtered: k=2, section filter, threshold=0.80 | medication instructions | medication-guideline.pdf | (no result) | 0 |
| filtered: k=2, section filter, threshold=0.80 | emergency identity check | triage-protocol.pdf | (no result) | 0 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | password reset steps | it-support-handbook.pdf | it-support-handbook.pdf | 1 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | password expiry policy | security-policy-2026.pdf | security-policy-2026.pdf | 1 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | wireless infrastructure | campus-it-guide.pdf | campus-it-guide.pdf | 1 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | medication instructions | medication-guideline.pdf | medication-guideline.pdf | 1 |
| adaptive filter: k=2, query-specific section, threshold=0.80 | emergency identity check | triage-protocol.pdf | triage-protocol.pdf | 1 |

## Chosen settings

**adaptive filter: k=2, query-specific section, threshold=0.80** is the best-performing setup: top-1 hit rate 1.00, top-k hit rate 1.00, and MRR 1.00. The adaptive section filter removes unrelated sections before ranking while retaining the expected source for every test query. The fixed Account access filter is intentionally included as a caution: a wrong metadata filter can hide all relevant results.

The vectors and corpus are deterministic, so the report can be regenerated with `python -m src.retrieval_tuning`.
